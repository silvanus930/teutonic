"""Paired cross-entropy evaluation: king vs merged model or LoRA adapter."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM

log = logging.getLogger(__name__)

EVAL_ALPHA = 0.001
EVAL_DELTA = 0.0025
LM_HEAD_CHUNK = 256


@torch.no_grad()
def compute_per_seq_loss(model, token_batches, device, chunk: int = LM_HEAD_CHUNK) -> list[float]:
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=device).contiguous()
    if hasattr(model, "reset_state"):
        model.reset_state()
    out = model.model(input_ids)
    hidden = out.last_hidden_state
    lm_head = model.lm_head
    n_pos = input_ids.size(1) - 1
    total = torch.zeros(len(token_batches), device=device)
    for i in range(0, n_pos, chunk):
        end = min(i + chunk, n_pos)
        logits = lm_head(hidden[:, i:end, :])
        labels = input_ids[:, i + 1:end + 1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        total += loss.reshape(len(token_batches), -1).sum(dim=1)
        del logits, loss
    return (total / n_pos).cpu().tolist()


def _bootstrap_lcb(diffs: np.ndarray, n_bootstrap: int, alpha: float) -> tuple[float, float]:
    mu_hat = float(diffs.mean())
    boot = np.empty(n_bootstrap)
    rng = np.random.default_rng(0xB007)
    for b in range(n_bootstrap):
        boot[b] = diffs[rng.integers(0, len(diffs), size=len(diffs))].mean()
    return mu_hat, float(np.quantile(boot, alpha))


def _eval_acceptance(
    mu_hat: float,
    lcb: float,
    acceptance_lcb_floor: float,
    mean_delta_floor: float,
) -> tuple[bool, list[str]]:
    accepted = lcb > acceptance_lcb_floor and mu_hat >= mean_delta_floor
    rejection_reasons: list[str] = []
    if lcb <= acceptance_lcb_floor:
        rejection_reasons.append(
            f"lcb={lcb:.6f} <= acceptance_lcb_floor={acceptance_lcb_floor:.6f}"
        )
    if mean_delta_floor > 0 and mu_hat < mean_delta_floor:
        rejection_reasons.append(
            f"mu_hat={mu_hat:.6f} < mean_delta_floor={mean_delta_floor:.6f}"
        )
    return accepted, rejection_reasons


def _load_model(path: str, device: str, hf_remote_code_kwargs, prepare_quasar_model):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        use_safetensors=True,
        **hf_remote_code_kwargs(path),
    )
    prepare_quasar_model(model)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    return model


def paired_eval_merged(
    king_dir: str,
    chall_dir: str,
    shard: np.ndarray,
    indices: list[int],
    device: str,
    *,
    batch_size: int = 8,
    n_bootstrap: int = 10000,
    alpha: float = EVAL_ALPHA,
    acceptance_lcb_floor: float = EVAL_DELTA,
    mean_delta_floor: float = 0.0,
    hf_remote_code_kwargs,
    prepare_quasar_model,
) -> dict:
    """Paired eval with two full merged models."""
    log.info("paired_eval_merged: king=%s challenger=%s device=%s", king_dir, chall_dir, device)
    king = _load_model(king_dir, device, hf_remote_code_kwargs, prepare_quasar_model)
    chall = _load_model(chall_dir, device, hf_remote_code_kwargs, prepare_quasar_model)
    return _run_paired_loop(
        king, chall, shard, indices, device,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        acceptance_lcb_floor=acceptance_lcb_floor,
        mean_delta_floor=mean_delta_floor,
        eval_mode="merged",
        challenger_path=chall_dir,
    )


def paired_eval_adapter(
    king_dir: str,
    adapter_dir: str,
    shard: np.ndarray,
    indices: list[int],
    device: str,
    *,
    batch_size: int = 8,
    n_bootstrap: int = 10000,
    alpha: float = EVAL_ALPHA,
    acceptance_lcb_floor: float = EVAL_DELTA,
    mean_delta_floor: float = 0.0,
    hf_remote_code_kwargs,
    prepare_quasar_model,
) -> dict:
    """Paired eval: king vs base+LoRA adapter (no merge)."""
    adapter_path = Path(adapter_dir)
    log.info("paired_eval_adapter: king=%s adapter=%s device=%s", king_dir, adapter_dir, device)
    king = _load_model(king_dir, device, hf_remote_code_kwargs, prepare_quasar_model)
    base = AutoModelForCausalLM.from_pretrained(
        king_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        use_safetensors=True,
        **hf_remote_code_kwargs(king_dir),
    )
    prepare_quasar_model(base)
    if hasattr(base, "config"):
        base.config.use_cache = True
    chall = PeftModel.from_pretrained(base, str(adapter_path))
    chall.eval()
    return _run_paired_loop(
        king, chall, shard, indices, device,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        acceptance_lcb_floor=acceptance_lcb_floor,
        mean_delta_floor=mean_delta_floor,
        eval_mode="adapter",
        challenger_path=str(adapter_path),
    )


def _run_paired_loop(
    king,
    chall,
    shard: np.ndarray,
    indices: list[int],
    device: str,
    *,
    batch_size: int,
    n_bootstrap: int,
    alpha: float,
    acceptance_lcb_floor: float,
    mean_delta_floor: float,
    eval_mode: str,
    challenger_path: str,
) -> dict:
    diffs: list[float] = []
    king_sum = chall_sum = 0.0
    n_done = 0
    t0 = time.time()
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        toks = [shard[j].tolist() for j in batch_idx]
        kl = compute_per_seq_loss(king, toks, device)
        cl = compute_per_seq_loss(chall, toks, device)
        for k, c in zip(kl, cl):
            diffs.append(k - c)
            king_sum += k
            chall_sum += c
            n_done += 1
        if (i // batch_size) % 5 == 0:
            mu = float(np.mean(diffs))
            log.info(
                "eval %d/%d | mu_hat=%.6f | king=%.4f chall=%.4f | %.1fs",
                n_done, len(indices), mu,
                king_sum / n_done, chall_sum / n_done, time.time() - t0,
            )

    diffs_arr = np.asarray(diffs, dtype=np.float64)
    mu_hat, lcb = _bootstrap_lcb(diffs_arr, n_bootstrap, alpha)
    accepted, rejection_reasons = _eval_acceptance(
        mu_hat, lcb, acceptance_lcb_floor, mean_delta_floor,
    )
    log.info(
        "paired_eval (%s): mu_hat=%.6f lcb=%.6f accepted=%s",
        eval_mode, mu_hat, lcb, accepted,
    )
    del king, chall
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "eval_mode": eval_mode,
        "challenger_path": challenger_path,
        "n_eval": n_done,
        "mu_hat": mu_hat,
        "mean_delta": mu_hat,
        "lcb": lcb,
        "delta": EVAL_DELTA,
        "delta_threshold": EVAL_DELTA,
        "alpha": alpha,
        "acceptance_lcb_floor": acceptance_lcb_floor,
        "mean_delta_floor": mean_delta_floor,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "avg_king_loss": king_sum / max(1, n_done),
        "avg_chall_loss": chall_sum / max(1, n_done),
        "avg_challenger_loss": chall_sum / max(1, n_done),
        "elapsed_s": time.time() - t0,
    }


def gpu_memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    return {
        "gpu_available": True,
        "max_memory_allocated_gb": round(
            torch.cuda.max_memory_allocated() / 1e9, 3,
        ),
        "max_memory_reserved_gb": round(
            torch.cuda.max_memory_reserved() / 1e9, 3,
        ),
    }


def mixture_weighted_paired_eval(
    king_dir: str,
    chall_dir: str,
    eval_pools: dict[str, tuple[np.ndarray, list[int]]],
    weights: dict[str, float],
    device: str,
    *,
    batch_size: int = 8,
    n_bootstrap: int = 10000,
    alpha: float = EVAL_ALPHA,
    acceptance_lcb_floor: float = EVAL_DELTA,
    mean_delta_floor: float = 0.0,
    hf_remote_code_kwargs,
    prepare_quasar_model,
    regression_datasets: tuple[str, ...] = ("automathtext-v2", "ultradata-math", "finewebedu"),
    regression_mu_floor: float = -0.001,
    regression_lcb_floor: float = -0.001,
) -> dict:
    """Per-dataset paired eval plus mixture-weighted pooled bootstrap."""
    log.info(
        "mixture_weighted_paired_eval: %d datasets, challenger=%s",
        len(eval_pools), chall_dir,
    )
    king = _load_model(king_dir, device, hf_remote_code_kwargs, prepare_quasar_model)
    chall = _load_model(chall_dir, device, hf_remote_code_kwargs, prepare_quasar_model)

    per_dataset: dict[str, dict] = {}
    pooled_diffs: list[float] = []
    pooled_weights: list[float] = []
    regression_warnings: list[str] = []

    for ds_name, (shard, indices) in eval_pools.items():
        if not indices:
            continue
        result = _run_paired_loop(
            king, chall, shard, indices, device,
            batch_size=batch_size,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            acceptance_lcb_floor=acceptance_lcb_floor,
            mean_delta_floor=mean_delta_floor,
            eval_mode="merged",
            challenger_path=chall_dir,
        )
        result["dataset"] = ds_name
        result["weight"] = weights.get(ds_name, 0.0)
        per_dataset[ds_name] = result

        w = float(weights.get(ds_name, 0.0))
        n = int(result.get("n_eval", 0))
        if n > 0 and w > 0:
            # Recompute diffs for pooling — approximate via stored stats is wrong;
            # re-run lightweight diff collection
            diffs = _collect_diffs(king, chall, shard, indices, device, batch_size)
            pooled_diffs.extend(diffs)
            pooled_weights.extend([w] * len(diffs))

        if ds_name in regression_datasets:
            mu = float(result.get("mu_hat", 0))
            lcb = float(result.get("lcb", 0))
            if mu < regression_mu_floor or lcb < regression_lcb_floor:
                msg = (
                    f"REGRESSION WARNING: {ds_name} mu_hat={mu:.6f} lcb={lcb:.6f} "
                    f"(floors mu>={regression_mu_floor}, lcb>={regression_lcb_floor})"
                )
                regression_warnings.append(msg)
                log.warning(msg)

    del king, chall
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not pooled_diffs:
        return {
            "eval_mode": "mixture",
            "per_dataset": per_dataset,
            "mixture_mu_hat": 0.0,
            "mixture_lcb": 0.0,
            "mu_hat": 0.0,
            "lcb": 0.0,
            "n_eval": 0,
            "accepted": False,
            "regression_warnings": regression_warnings,
        }

    diffs_arr = np.asarray(pooled_diffs, dtype=np.float64)
    weights_arr = np.asarray(pooled_weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    mixture_mu = float(np.average(diffs_arr, weights=weights_arr))

    boot = np.empty(n_bootstrap)
    rng = np.random.default_rng(0xB007)
    n = len(diffs_arr)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True, p=weights_arr)
        boot[b] = diffs_arr[idx].mean()
    mixture_lcb = float(np.quantile(boot, alpha))

    accepted, rejection_reasons = _eval_acceptance(
        mixture_mu, mixture_lcb, acceptance_lcb_floor, mean_delta_floor,
    )
    if regression_warnings:
        rejection_reasons.extend(regression_warnings)

    return {
        "eval_mode": "mixture",
        "per_dataset": per_dataset,
        "mixture_mu_hat": mixture_mu,
        "mixture_lcb": mixture_lcb,
        "mu_hat": mixture_mu,
        "lcb": mixture_lcb,
        "mean_delta": mixture_mu,
        "n_eval": int(sum(r.get("n_eval", 0) for r in per_dataset.values())),
        "delta": EVAL_DELTA,
        "delta_threshold": EVAL_DELTA,
        "alpha": alpha,
        "acceptance_lcb_floor": acceptance_lcb_floor,
        "mean_delta_floor": mean_delta_floor,
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "regression_warnings": regression_warnings,
        "weights": weights,
    }


def _collect_diffs(king, chall, shard, indices, device, batch_size) -> list[float]:
    diffs: list[float] = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        toks = [shard[j].tolist() for j in batch_idx]
        kl = compute_per_seq_loss(king, toks, device)
        cl = compute_per_seq_loss(chall, toks, device)
        diffs.extend(k - c for k, c in zip(kl, cl))
    return diffs


def mixture_weighted_paired_eval_adapter(
    king_dir: str,
    adapter_dir: str,
    eval_pools: dict[str, tuple[np.ndarray, list[int]]],
    weights: dict[str, float],
    device: str,
    *,
    batch_size: int = 8,
    n_bootstrap: int = 10000,
    alpha: float = EVAL_ALPHA,
    acceptance_lcb_floor: float = EVAL_DELTA,
    mean_delta_floor: float = 0.0,
    hf_remote_code_kwargs,
    prepare_quasar_model,
    regression_datasets: tuple[str, ...] = ("automathtext-v2", "ultradata-math", "finewebedu"),
    regression_mu_floor: float = -0.001,
    regression_lcb_floor: float = -0.001,
) -> dict:
    """Mixture eval with LoRA adapter (no merge)."""
    from peft import PeftModel

    king = _load_model(king_dir, device, hf_remote_code_kwargs, prepare_quasar_model)
    base = AutoModelForCausalLM.from_pretrained(
        king_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        use_safetensors=True,
        **hf_remote_code_kwargs(king_dir),
    )
    prepare_quasar_model(base)
    chall = PeftModel.from_pretrained(base, str(adapter_dir))
    chall.eval()

    per_dataset: dict[str, dict] = {}
    pooled_diffs: list[float] = []
    pooled_weights: list[float] = []
    regression_warnings: list[str] = []

    for ds_name, (shard, indices) in eval_pools.items():
        if not indices:
            continue
        result = _run_paired_loop(
            king, chall, shard, indices, device,
            batch_size=batch_size,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            acceptance_lcb_floor=acceptance_lcb_floor,
            mean_delta_floor=mean_delta_floor,
            eval_mode="adapter",
            challenger_path=str(adapter_dir),
        )
        result["dataset"] = ds_name
        result["weight"] = weights.get(ds_name, 0.0)
        per_dataset[ds_name] = result
        w = float(weights.get(ds_name, 0.0))
        diffs = _collect_diffs(king, chall, shard, indices, device, batch_size)
        if diffs and w > 0:
            pooled_diffs.extend(diffs)
            pooled_weights.extend([w] * len(diffs))
        if ds_name in regression_datasets:
            mu = float(result.get("mu_hat", 0))
            lcb = float(result.get("lcb", 0))
            if mu < regression_mu_floor or lcb < regression_lcb_floor:
                msg = (
                    f"REGRESSION WARNING: {ds_name} mu_hat={mu:.6f} lcb={lcb:.6f}"
                )
                regression_warnings.append(msg)
                log.warning(msg)

    del king, chall, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not pooled_diffs:
        return {
            "eval_mode": "mixture_adapter",
            "per_dataset": per_dataset,
            "mixture_mu_hat": 0.0,
            "mixture_lcb": 0.0,
            "mu_hat": 0.0,
            "lcb": 0.0,
            "n_eval": 0,
            "accepted": False,
            "regression_warnings": regression_warnings,
        }

    diffs_arr = np.asarray(pooled_diffs, dtype=np.float64)
    weights_arr = np.asarray(pooled_weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    mixture_mu = float(np.average(diffs_arr, weights=weights_arr))
    boot = np.empty(n_bootstrap)
    rng = np.random.default_rng(0xB007)
    n = len(diffs_arr)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True, p=weights_arr)
        boot[b] = diffs_arr[idx].mean()
    mixture_lcb = float(np.quantile(boot, alpha))
    accepted, rejection_reasons = _eval_acceptance(
        mixture_mu, mixture_lcb, acceptance_lcb_floor, mean_delta_floor,
    )
    if regression_warnings:
        rejection_reasons.extend(regression_warnings)

    return {
        "eval_mode": "mixture_adapter",
        "per_dataset": per_dataset,
        "mixture_mu_hat": mixture_mu,
        "mixture_lcb": mixture_lcb,
        "mu_hat": mixture_mu,
        "lcb": mixture_lcb,
        "n_eval": int(sum(r.get("n_eval", 0) for r in per_dataset.values())),
        "accepted": accepted,
        "rejection_reasons": rejection_reasons,
        "regression_warnings": regression_warnings,
        "weights": weights,
    }
