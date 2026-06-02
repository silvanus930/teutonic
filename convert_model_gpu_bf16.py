#!/usr/bin/env python3
"""
Convert a Hugging Face model checkpoint from FP32 to BF16 on GPU.

Supports kings that ship bundled remote code (e.g. configuration_qwen3_5.py /
modeling_qwen3_5.py for Quasar) by adding the model directory to sys.path and
applying the transformers mask compat patch used elsewhere in teutonic.

Usage:
    python convert_model_gpu_bf16.py --src /path/to/fp32_model --dst /path/to/bf16_model
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("convert_model_gpu_bf16")

CUSTOM_CODE_GLOBS = (
    "modeling_*.py",
    "configuration_*.py",
    "tokenization_*.py",
    "processing_*.py",
    "modular_*.py",
)

CUSTOM_CODE_MARKERS = (
    "configuration_qwen3_5.py",
    "modeling_qwen3_5.py",
    "configuration_quasar.py",
    "modeling_quasar.py",
)

SAVE_KWARGS = {
    "safe_serialization": True,
    "max_shard_size": "4GB",
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def patch_transformers_quasar_compat() -> None:
    """Kings with bundled modeling_qwen3_5.py use a newer create_causal_mask API."""
    import transformers.masking_utils as masking

    if getattr(masking, "_teutonic_quasar_compat", False):
        return

    original = masking.create_causal_mask
    allowed = set(inspect.signature(original).parameters)

    def compat_create_causal_mask(*args, **kwargs):
        if "cache_position" in kwargs:
            return None
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return original(*args, **filtered)

    masking.create_causal_mask = compat_create_causal_mask
    masking._teutonic_quasar_compat = True
    LOGGER.debug("Applied transformers quasar mask compat patch.")


def has_bundled_remote_code(model_dir: Path) -> bool:
    if any((model_dir / name).is_file() for name in CUSTOM_CODE_MARKERS):
        return True
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        return bool(json.loads(cfg_path.read_text()).get("auto_map"))
    except (json.JSONDecodeError, OSError):
        return False


def prepare_remote_code(model_dir: Path) -> None:
    """Make in-repo configuration_*/modeling_* modules importable."""
    resolved = model_dir.resolve()
    if not has_bundled_remote_code(resolved):
        return
    patch_transformers_quasar_compat()
    path = str(resolved)
    if path not in sys.path:
        sys.path.insert(0, path)
        LOGGER.info("Added %s to sys.path for bundled remote code.", resolved)


def load_dtype_kwarg(dtype: torch.dtype) -> dict:
    """Prefer `dtype` (transformers >=5.9); fall back to `torch_dtype`."""
    try:
        from transformers import AutoModelForCausalLM as _cls

        params = inspect.signature(_cls.from_pretrained).parameters
        if "dtype" in params:
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        LOGGER.info("CUDA available: %s (%.1f GiB)", name, total_gb)
        return device
    LOGGER.warning("CUDA not available; using CPU (conversion will be slower).")
    return torch.device("cpu")


def validate_paths(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src}")
    if not any(src.glob("*.safetensors")) and not (src / "model.safetensors").exists():
        index = src / "model.safetensors.index.json"
        if not index.is_file():
            bin_files = list(src.glob("*.bin")) + list(src.glob("pytorch_model*.bin"))
            if not bin_files:
                LOGGER.warning(
                    "No weight files found in %s; continuing anyway.", src
                )
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(
            f"Destination already exists and is not empty: {dst}. "
            "Remove it or choose a different path."
        )


def finalize_config_from_source(src: Path, dst: Path) -> None:
    """Keep the source config intact; only force BF16 dtype fields."""
    src_cfg_path = src / "config.json"
    dst_cfg_path = dst / "config.json"
    if not src_cfg_path.is_file() or not dst_cfg_path.is_file():
        LOGGER.warning("Skipping config merge; config.json missing in src or dst.")
        return

    src_cfg = json.loads(src_cfg_path.read_text())
    merged = dict(src_cfg)
    merged["dtype"] = "bfloat16"
    merged["torch_dtype"] = "bfloat16"
    dst_cfg_path.write_text(json.dumps(merged, indent=2) + "\n")
    LOGGER.info("Merged source config.json into destination (dtype=bfloat16).")

    for extra in ("generation_config.json",):
        extra_src = src / extra
        if extra_src.is_file():
            shutil.copy2(extra_src, dst / extra)
            LOGGER.info("Copied %s from source.", extra)


def copy_custom_code_files(src: Path, dst: Path) -> list[str]:
    """Copy custom Python modules required by trust_remote_code models."""
    dst.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    seen: set[str] = set()

    for pattern in CUSTOM_CODE_GLOBS:
        for src_file in sorted(src.glob(pattern)):
            name = src_file.name
            if name in seen:
                continue
            seen.add(name)
            shutil.copy2(src_file, dst / name)
            copied.append(name)
            LOGGER.info("Copied custom code: %s", name)

    init_src = src / "__init__.py"
    if init_src.is_file():
        shutil.copy2(init_src, dst / "__init__.py")
        copied.append("__init__.py")
        LOGGER.info("Copied custom code: __init__.py")

    return copied


def build_load_kwargs(device: torch.device, dtype: torch.dtype) -> dict:
    kwargs: dict = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        **load_dtype_kwarg(dtype),
    }
    if device.type == "cuda":
        kwargs["device_map"] = {"": device}
    return kwargs


def load_model(src: Path, device: torch.device):
    """Load full model with remote code enabled."""
    prepare_remote_code(src)
    LOGGER.info("Loading model from %s (trust_remote_code=True)...", src)
    model = AutoModelForCausalLM.from_pretrained(
        str(src), **build_load_kwargs(device, torch.float32)
    )
    param_count = sum(p.numel() for p in model.parameters())
    LOGGER.info("Loaded %s with %s parameters.", type(model).__name__, f"{param_count:,}")
    return model


def convert_weights_to_bf16(model, device: torch.device):
    """Cast floating-point parameters and buffers to bfloat16."""
    LOGGER.info("Converting weights to torch.bfloat16 on %s...", device)
    model = model.to(dtype=torch.bfloat16)
    if device.type == "cuda":
        model = model.to(device)
    return model


def set_config_dtype(model) -> None:
    LOGGER.info("Setting config dtype to bfloat16.")
    model.config.torch_dtype = "bfloat16"
    if hasattr(model.config, "dtype"):
        model.config.dtype = "bfloat16"


def save_model_and_tokenizer(model, src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    copy_custom_code_files(src, dst)

    LOGGER.info(
        "Saving BF16 model to %s (safetensors, max_shard_size=4GB)...", dst
    )
    model.save_pretrained(str(dst), **SAVE_KWARGS)
    finalize_config_from_source(src, dst)

    LOGGER.info("Saving tokenizer...")
    try:
        prepare_remote_code(src)
        tokenizer = AutoTokenizer.from_pretrained(str(src), trust_remote_code=True)
        tokenizer.save_pretrained(str(dst))
        LOGGER.info("Tokenizer saved.")
    except Exception as exc:
        LOGGER.warning("Could not load/save tokenizer: %s", exc)


def count_non_bf16_float_params(model) -> tuple[int, int]:
    """Count learnable floating weights only (exclude RoPE buffers kept in FP32)."""
    non_bf16 = 0
    float_total = 0
    for tensor in model.parameters():
        if not tensor.is_floating_point():
            continue
        float_total += 1
        if tensor.dtype != torch.bfloat16:
            non_bf16 += 1
    return non_bf16, float_total


def config_reports_bf16(config) -> bool:
    for attr in ("torch_dtype", "dtype"):
        value = getattr(config, attr, None)
        if value in ("bfloat16", torch.bfloat16):
            return True
    return False


def verify_saved_model(dst: Path, device: torch.device) -> None:
    """Reload saved checkpoint and confirm BF16 floating weights."""
    LOGGER.info("Verification: reloading model from %s...", dst)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    prepare_remote_code(dst)
    model = AutoModelForCausalLM.from_pretrained(
        str(dst), **build_load_kwargs(device, torch.bfloat16)
    )
    config = AutoConfig.from_pretrained(str(dst), trust_remote_code=True)
    if not config_reports_bf16(config):
        raise RuntimeError(
            "config.torch_dtype/dtype is not bfloat16 after save: "
            f"torch_dtype={getattr(config, 'torch_dtype', None)!r}, "
            f"dtype={getattr(config, 'dtype', None)!r}."
        )

    non_bf16, float_total = count_non_bf16_float_params(model)
    if non_bf16 > 0:
        raise RuntimeError(
            f"Verification failed: {non_bf16}/{float_total} floating tensors "
            "are not bfloat16."
        )

    safetensors_files = list(dst.glob("*.safetensors"))
    if not safetensors_files:
        raise RuntimeError(
            f"Verification failed: no .safetensors files in {dst}."
        )

    LOGGER.info(
        "Verification passed: %d safetensors shard(s), all %d floating tensors "
        "are bfloat16.",
        len(safetensors_files),
        float_total,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face FP32 model checkpoint to BF16 on GPU.",
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the source FP32 model directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Path to write the converted BF16 model directory.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    src = args.src.resolve()
    dst = args.dst.resolve()

    LOGGER.info("Source: %s", src)
    LOGGER.info("Destination: %s", dst)

    try:
        validate_paths(src, dst)
        device = resolve_device()
        model = load_model(src, device)
        model = convert_weights_to_bf16(model, device)
        set_config_dtype(model)
        save_model_and_tokenizer(model, src, dst)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        verify_saved_model(dst, device)
        LOGGER.info("Conversion complete: %s", dst)
        return 0
    except Exception as exc:
        LOGGER.error("Conversion failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
