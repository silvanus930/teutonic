"""HuggingFace load helpers for Quasar kings with bundled modeling_qwen3_5.py."""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path


def ensure_quasar_arch_registered() -> None:
    """Register vendored Quasar for quasar_text checkpoints (no trust_remote_code)."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        from archs.quasar.configuration_quasar import QuasarConfig
        from archs.quasar.modeling_quasar import QuasarForCausalLM
        import archs.quasar  # noqa: F401 — side effect registers "quasar"
    except ImportError:
        return
    for model_type, cfg_cls, model_cls in (
        ("quasar_text", QuasarConfig, QuasarForCausalLM),
    ):
        try:
            AutoConfig.register(model_type, cfg_cls)
        except ValueError:
            pass
        try:
            AutoModelForCausalLM.register(cfg_cls, model_cls)
        except ValueError:
            pass


def patch_transformers_quasar_compat() -> None:
    """Kings with auto_map use modeling_qwen3_5 against unreleased transformers APIs."""
    import transformers.masking_utils as _masking

    if getattr(_masking, "_teutonic_quasar_compat", False):
        return
    _orig = _masking.create_causal_mask
    _allowed = set(inspect.signature(_orig).parameters)

    def _compat_create_causal_mask(*args, **kwargs):
        if "cache_position" in kwargs:
            return None
        filt = {k: v for k, v in kwargs.items() if k in _allowed}
        return _orig(*args, **filt)

    _masking.create_causal_mask = _compat_create_causal_mask
    _masking._teutonic_quasar_compat = True
    for _name, _mod in list(sys.modules.items()):
        if _name.endswith("modeling_qwen3_5") and hasattr(_mod, "create_causal_mask"):
            _mod.create_causal_mask = _compat_create_causal_mask


def prepare_quasar_model(model) -> int:
    """Disable causal-conv1d CUDA kernels that fail on some Quasar weight layouts.

    Bundled ``modeling_qwen3_5.py`` calls ``causal_conv1d_fn`` with
    ``conv1d.weight.squeeze(1)`` views that can trip
    ``Cannot access data pointer of Tensor that doesn't have storage`` on
    torch 2.12 + cu130. The model already has a PyTorch conv fallback when
    ``causal_conv1d_fn`` is None.
    """
    n = 0
    for module in model.modules():
        if getattr(module, "causal_conv1d_fn", None) is not None:
            module.causal_conv1d_fn = None
            n += 1
        if getattr(module, "causal_conv1d_update", None) is not None:
            module.causal_conv1d_update = None
    return n


def _is_quasar_king_dir(model_path: Path) -> bool:
    cfg_path = model_path / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    model_type = str(cfg.get("model_type", ""))
    archs = cfg.get("architectures") or []
    layer_types = cfg.get("layer_types") or []
    return (
        model_type in ("quasar", "quasar_text")
        or any("Quasar" in str(a) for a in archs)
        or "linear_attention" in layer_types
    )


def ensure_quasar_runtime_deps(model_path: str = "") -> None:
    """Quasar kings with bundled modeling_qwen3_5.py need fla + causal-conv1d."""
    p = Path(model_path).expanduser().resolve() if model_path else None
    if p and p.is_dir():
        has_custom = any((p / n).is_file() for n in (
            "configuration_qwen3_5.py", "modeling_qwen3_5.py",
        ))
        if not has_custom:
            return
    elif p and p.is_dir() and not _is_quasar_king_dir(p):
        return
    missing: list[str] = []
    try:
        import causal_conv1d  # noqa: F401
    except ImportError:
        missing.append("causal-conv1d")
    try:
        import fla  # noqa: F401
    except ImportError:
        missing.append("flash-linear-attention (fla)")
    if not missing:
        return
    raise ImportError(
        "Quasar king load requires: " + ", ".join(missing) + ".\n"
        "See scripts/mining/requirements.txt for install commands."
    )


def hf_remote_code_kwargs(model_path: str) -> dict:
    """Return from_pretrained kwargs for kings that ship local Python modules."""
    p = Path(model_path).expanduser().resolve()
    if not p.is_dir():
        return {}
    has_custom = any((p / name).is_file() for name in (
        "configuration_qwen3_5.py", "modeling_qwen3_5.py",
    ))
    if not has_custom:
        return {}
    ensure_quasar_runtime_deps(str(p))
    patch_transformers_quasar_compat()
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return {"trust_remote_code": True}


_QUASAR_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)


def default_lora_target_modules_for_king(model_path: str) -> str | None:
    """Comma-separated LoRA suffixes for kings with bundled Quasar/hybrid layers."""
    cfg_path = Path(model_path).expanduser() / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    model_type = str(cfg.get("model_type", ""))
    archs = cfg.get("architectures") or []
    layer_types = cfg.get("layer_types") or []
    if (
        model_type in ("quasar", "quasar_text")
        or any("Quasar" in str(a) for a in archs)
        or "linear_attention" in layer_types
    ):
        return ",".join(_QUASAR_LORA_TARGETS)
    return None


def king_subprocess_env(model_path: str, env: dict | None = None) -> dict:
    """Environment for torchrun / subprocesses that load the same king."""
    out = dict(env if env is not None else os.environ)
    p = Path(model_path).expanduser().resolve()
    if not p.is_dir():
        return out
    patch_transformers_quasar_compat()
    if any((p / n).is_file() for n in ("configuration_qwen3_5.py", "modeling_qwen3_5.py")):
        ensure_quasar_runtime_deps(str(p))
        prev = out.get("PYTHONPATH", "")
        s = str(p)
        out["PYTHONPATH"] = f"{s}:{prev}" if prev else s
        out["TRUST_REMOTE_CODE"] = "true"
    return out
