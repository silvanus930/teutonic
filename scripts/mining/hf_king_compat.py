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


def hf_remote_code_kwargs(model_path: str) -> dict:
    """Return from_pretrained kwargs for kings that ship local Python modules."""
    p = Path(model_path).expanduser().resolve()
    if not p.is_dir():
        return {}
    # Merged challengers may copy auto_map in config.json without shipping *.py;
    # only enable remote code when the modeling files are actually present.
    has_custom = any((p / name).is_file() for name in (
        "configuration_qwen3_5.py", "modeling_qwen3_5.py",
    ))
    if not has_custom:
        return {}
    patch_transformers_quasar_compat()
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
    return {"trust_remote_code": True}


def king_subprocess_env(model_path: str, env: dict | None = None) -> dict:
    """Environment for torchrun / subprocesses that load the same king."""
    out = dict(env if env is not None else os.environ)
    p = Path(model_path).expanduser().resolve()
    if not p.is_dir():
        return out
    patch_transformers_quasar_compat()
    if any((p / n).is_file() for n in ("configuration_qwen3_5.py", "modeling_qwen3_5.py")):
        prev = out.get("PYTHONPATH", "")
        s = str(p)
        out["PYTHONPATH"] = f"{s}:{prev}" if prev else s
        out["TRUST_REMOTE_CODE"] = "true"
    return out
