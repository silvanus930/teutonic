#!/usr/bin/env python3
"""LoRA fine-tune a causal LM on pretokenized JSONL data.

Improvements over baseline:
  - WSD (Warmup-Stable-Decay) scheduler in addition to cosine/linear/constant
  - Custom adam_beta2 (default 0.95, vs HF default 0.999)
  - Post-training checkpoint selection: parses trainer_state.json to find
    the best checkpoint by eval_loss — does NOT re-run inference
  - Optional LoRA checkpoint averaging: averages top-K checkpoint adapters
    and uses the averaged model if it beats the best single checkpoint
  - Writes checkpoint_scores.json and train_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import sys

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model

_mining = Path(__file__).resolve().parents[1] / "mining"
if str(_mining) not in sys.path:
    sys.path.insert(0, str(_mining))
from hf_king_compat import (  # noqa: E402
    default_lora_target_modules_for_king,
    hf_remote_code_kwargs,
    load_king_tokenizer,
    patch_transformers_quasar_compat,
    prepare_quasar_model,
)

patch_transformers_quasar_compat()


# ---------------------------------------------------------------------------
# Dataset / collator
# ---------------------------------------------------------------------------

class TokenIdsDataset(Dataset):
    def __init__(self, path: str, seq_len: int, prefilter_nonfinite: bool = False):
        self.rows: list[list[int]] = []
        n_skipped = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids = obj["input_ids"][:seq_len]
                if len(ids) < seq_len:
                    n_skipped += 1
                    continue
                self.rows.append(ids)
        self.seq_len = seq_len
        if n_skipped:
            print(f"[dataset] {path}: skipped {n_skipped} short sequences (< {seq_len} tokens)")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        ids = self.rows[idx]
        x = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


@dataclass
class Collator:
    def __call__(self, features: list[dict]) -> dict:
        return {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }


# ---------------------------------------------------------------------------
# WSD scheduler
# ---------------------------------------------------------------------------

def get_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    stable_ratio: float = 0.0,
    min_lr_ratio: float = 0.10,
) -> LambdaLR:
    """Warmup → Stable → Linear-decay scheduler.

    Phases:
      warmup: steps 0 .. num_warmup_steps           (0 → lr)
      stable: next stable_ratio * total steps       (lr constant)
      decay:  remaining steps                       (lr → min_lr_ratio * lr, linear)
    """
    num_stable_steps = int(num_training_steps * stable_ratio)
    num_decay_steps = max(1, num_training_steps - num_warmup_steps - num_stable_steps)

    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        step -= num_warmup_steps
        if step < num_stable_steps:
            return 1.0
        step -= num_stable_steps
        progress = min(1.0, step / num_decay_steps)
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Post-training checkpoint helpers
# ---------------------------------------------------------------------------

def extract_checkpoint_evals(output_dir: Path) -> dict[str, float]:
    """Parse trainer_state.json for per-checkpoint eval_loss.
    Returns {checkpoint_dir_str: best_eval_loss}. No re-inference."""
    state_path = output_dir / "trainer_state.json"
    if not state_path.is_file():
        ckpts = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        for ckpt in reversed(ckpts):
            candidate = ckpt / "trainer_state.json"
            if candidate.is_file():
                state_path = candidate
                break
        else:
            return {}
    try:
        state = json.loads(state_path.read_text())
    except Exception:
        return {}
    result: dict[str, float] = {}
    for entry in state.get("log_history", []):
        if "eval_loss" not in entry or "step" not in entry:
            continue
        ckpt = output_dir / f"checkpoint-{int(entry['step'])}"
        if not ckpt.is_dir():
            continue
        key = str(ckpt)
        loss = float(entry["eval_loss"])
        if key not in result or loss < result[key]:
            result[key] = loss
    return result


def average_lora_adapters(checkpoint_dirs: list[Path], output_dir: Path) -> Path:
    """Average LoRA adapter tensors from multiple checkpoint dirs."""
    try:
        from safetensors.torch import load_file as st_load, save_file as st_save
        has_st = True
    except ImportError:
        has_st = False

    tensors_list: list[dict] = []
    for ckpt in checkpoint_dirs:
        st_path = ckpt / "adapter_model.safetensors"
        bin_path = ckpt / "adapter_model.bin"
        if st_path.is_file() and has_st:
            tensors_list.append(st_load(str(st_path), device="cpu"))
        elif bin_path.is_file():
            tensors_list.append(torch.load(str(bin_path), map_location="cpu", weights_only=True))
        else:
            print(f"[avg] warning: no adapter found in {ckpt.name}, skipping")

    if not tensors_list:
        raise RuntimeError("No adapter weights found in any of the provided checkpoint dirs")

    print(f"[avg] averaging {len(tensors_list)} adapters")
    averaged: dict = {}
    for key in tensors_list[0].keys():
        stacked = torch.stack([t[key].float() for t in tensors_list])
        averaged[key] = stacked.mean(dim=0).to(tensors_list[0][key].dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    if has_st:
        from safetensors.torch import save_file as st_save
        st_save(averaged, str(output_dir / "adapter_model.safetensors"))
    else:
        torch.save(averaged, str(output_dir / "adapter_model.bin"))

    # Copy adapter_config.json from first valid checkpoint
    for ckpt in checkpoint_dirs:
        cfg = ckpt / "adapter_config.json"
        if cfg.is_file():
            shutil.copy(cfg, output_dir / "adapter_config.json")
            break

    print(f"[avg] saved averaged adapter to {output_dir}")
    return output_dir


def eval_adapter_val_loss(
    base_model: str,
    adapter_dir: Path,
    val_ds: TokenIdsDataset,
    dtype: torch.dtype,
    max_batches: int = 64,
    batch_size: int = 4,
) -> float:
    """Evaluate an adapter on val_ds, return mean CE loss. Runs on cuda:0 or cpu."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _king_kw = hf_remote_code_kwargs(base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, use_safetensors=True, **_king_kw,
    )
    prepare_quasar_model(base)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model.eval()

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=Collator())
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=ids, labels=labels)
            total_loss += out.loss.item()
            n += 1

    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return total_loss / max(1, n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LoRA fine-tune a causal LM on pretokenized JSONL data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Core
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.5)
    ap.add_argument(
        "--max-steps", type=int, default=0,
        help="Cap total optimizer steps (overrides epochs when > 0).",
    )
    ap.add_argument(
        "--skip-trainer-eval", action="store_true",
        help="Disable periodic validation during training (faster micro-screens).",
    )

    # LR scheduler
    ap.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "constant_with_warmup", "linear", "cosine", "wsd"),
        default="cosine",
        help="LR scheduler. 'wsd' = Warmup-Stable-Decay (custom implementation).",
    )
    ap.add_argument("--warmup-ratio", type=float, default=0.03,
                    help="Warmup fraction of total steps (used if --warmup-steps=0).")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Absolute warmup steps (overrides --warmup-ratio when > 0).")
    ap.add_argument("--min-lr-ratio", type=float, default=0.10,
                    help="Min LR at end of schedule = lr * min_lr_ratio.")
    ap.add_argument("--stable-ratio", type=float, default=0.0,
                    help="WSD: fraction of total steps for the constant-LR stable phase.")
    ap.add_argument("--decay-ratio", type=float, default=0.0,
                    help="WSD: fraction of total steps for decay (0 = all steps after warmup+stable).")

    # Optimizer
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--adam-beta2", type=float, default=0.95)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lora-target-modules", type=str, default=None,
                    help="Comma-separated module name suffixes. "
                         "Defaults to Quasar/Qwen3-aware set.")

    # Dtype / hardware
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", action="store_false",
                    dest="gradient_checkpointing")

    # Checkpointing
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=300)
    ap.add_argument("--eval-steps", type=int, default=300)
    ap.add_argument("--save-total-limit", type=int, default=0,
                    help="Max checkpoints to keep. 0 = auto (based on --average-top-k).")

    # Post-training checkpoint selection / averaging
    ap.add_argument("--average-top-k-lora-checkpoints", type=int, default=0,
                    help="Average adapter weights from top-K checkpoints by eval_loss. "
                         "0 = disabled. Averaged model replaces best_adapter only if it "
                         "achieves lower val loss than the best single checkpoint.")

    # Misc
    ap.add_argument("--prefilter-nonfinite", action="store_true", default=False)
    ap.add_argument("--no-prefilter-nonfinite", action="store_false", dest="prefilter_nonfinite")
    ap.add_argument("--resume-from-checkpoint", default="")
    args = ap.parse_args()

    t0 = time.time()
    use_cuda = torch.cuda.is_available()
    use_bf16 = args.dtype == "bf16" and use_cuda
    use_fp16 = args.dtype == "fp16" and use_cuda
    eval_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = local_rank == 0

    # ---- Model ----
    _king_kw = hf_remote_code_kwargs(args.base_model)
    tokenizer = load_king_tokenizer(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=eval_dtype,
        use_safetensors=True,
        **_king_kw,
    )
    n_quasar = prepare_quasar_model(model)
    if is_main and n_quasar:
        print(f"[train_lora] quasar compat: PyTorch conv fallback on {n_quasar} layer(s)")
    model.config.use_cache = False

    lora_targets = args.lora_target_modules or default_lora_target_modules_for_king(
        args.base_model,
    )
    target_modules = (
        lora_targets.split(",") if lora_targets
        else ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]
    )
    if is_main:
        print(f"[lora] target_modules={target_modules}")
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )
    resume = (args.resume_from_checkpoint or "").strip()
    resume_ckpt: str | None = None
    adapter_resume: str | None = None
    if resume:
        resume_p = Path(resume)
        if (resume_p / "trainer_state.json").is_file():
            resume_ckpt = str(resume_p)
        elif (resume_p / "adapter_model.safetensors").is_file() or \
                (resume_p / "adapter_model.bin").is_file():
            adapter_resume = str(resume_p)
        elif is_main:
            raise FileNotFoundError(
                f"--resume-from-checkpoint {resume} has no trainer_state.json "
                "or adapter weights"
            )

    if adapter_resume:
        model = PeftModel.from_pretrained(
            model, adapter_resume, is_trainable=True,
        )
        if is_main:
            print(f"[lora] loaded adapter weights from {adapter_resume}")
    else:
        model = get_peft_model(model, lora_cfg)
    if is_main:
        model.print_trainable_parameters()

    # ---- Dataset ----
    train_ds = TokenIdsDataset(args.train_data, args.seq_len, args.prefilter_nonfinite)
    val_ds = TokenIdsDataset(args.val_data, args.seq_len, args.prefilter_nonfinite)
    if is_main:
        print(f"[dataset] train={len(train_ds)} val={len(val_ds)} seq_len={args.seq_len}")

    # ---- Scheduler / step calculation ----
    steps_per_epoch = math.ceil(
        len(train_ds) / (args.micro_batch_size * world_size * args.grad_accum)
    )
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = (
        args.warmup_steps if args.warmup_steps > 0
        else int(total_steps * args.warmup_ratio)
    )
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    if is_main:
        print(
            f"[scheduler] type={args.lr_scheduler_type} total_steps={total_steps} "
            f"warmup={warmup_steps} min_lr_ratio={args.min_lr_ratio} "
            f"stable_ratio={args.stable_ratio} adam_beta2={args.adam_beta2}"
        )

    # ---- Build optimizer + scheduler ----
    use_custom = args.lr_scheduler_type == "wsd"
    optimizers: tuple = (None, None)

    if use_custom:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, args.adam_beta2),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )
        scheduler = get_wsd_scheduler(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            stable_ratio=args.stable_ratio,
            min_lr_ratio=args.min_lr_ratio,
        )
        optimizers = (optimizer, scheduler)
        hf_scheduler = "constant"   # Trainer won't override with its own
        hf_warmup_steps = 0
    else:
        hf_scheduler = args.lr_scheduler_type
        hf_warmup_steps = warmup_steps

    # ---- Checkpoint retention ----
    k_avg = args.average_top_k_lora_checkpoints
    if args.save_total_limit > 0:
        save_total_limit = args.save_total_limit
    elif k_avg > 0:
        save_total_limit = max(6, k_avg + 3)
    else:
        save_total_limit = 3

    # ---- TrainingArguments ----
    train_kw: dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.micro_batch_size,
        "per_device_eval_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "warmup_steps": hf_warmup_steps,
        "warmup_ratio": 0.0,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "adam_beta2": args.adam_beta2,
        "lr_scheduler_type": hf_scheduler,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": save_total_limit,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": "none",
        "ddp_find_unused_parameters": False,
    }
    if args.skip_trainer_eval:
        train_kw["eval_strategy"] = "no"
        train_kw["load_best_model_at_end"] = False
    else:
        train_kw["eval_strategy"] = "steps"
        train_kw["eval_steps"] = args.eval_steps
        train_kw["load_best_model_at_end"] = True
        train_kw["metric_for_best_model"] = "eval_loss"
        train_kw["greater_is_better"] = False
    if args.max_steps > 0:
        train_kw["max_steps"] = total_steps
    else:
        train_kw["num_train_epochs"] = args.epochs
    training_args = TrainingArguments(**train_kw)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=Collator(),
        optimizers=optimizers,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ============================================================
    # POST-TRAINING: checkpoint selection + optional averaging
    #
    # DDP safety: after trainer.train() returns, no barriers are
    # active. All file I/O is confined to rank 0 (is_main).
    # Non-main ranks do nothing — their model state is identical to
    # rank 0's (DDP gradient sync guarantees this).
    # ============================================================
    output_dir = Path(args.output_dir)
    final_adapter_dir = output_dir / "best_adapter"
    checkpoint_scores: dict = {}
    averaged_is_better = False

    if is_main:
        # --- Parse trainer_state.json: no re-inference, just read logs ---
        ckpt_evals = extract_checkpoint_evals(output_dir)
        sorted_ckpts = sorted(ckpt_evals.items(), key=lambda x: x[1])

        best_single_dir: Path | None = None
        best_single_loss = float("inf")
        if sorted_ckpts:
            best_single_dir = Path(sorted_ckpts[0][0])
            best_single_loss = sorted_ckpts[0][1]

        checkpoint_scores = {
            "checkpoints": [{"dir": d, "eval_loss": v} for d, v in sorted_ckpts],
            "best_checkpoint": str(best_single_dir) if best_single_dir else None,
            "best_checkpoint_eval_loss": best_single_loss if best_single_dir else None,
            "n_checkpoints_found": len(ckpt_evals),
        }
        print(
            f"[ckpt] {len(ckpt_evals)} checkpoints in trainer_state | "
            f"best={best_single_dir.name if best_single_dir else 'none'} "
            f"loss={best_single_loss:.4f}"
        )

        # --- Optional LoRA checkpoint averaging (rank 0 only) ---
        avg_loss: float | None = None
        avg_dir = output_dir / "lora_avg"

        if k_avg > 0 and len(sorted_ckpts) >= 2:
            k = min(k_avg, len(sorted_ckpts))
            top_k_dirs = [Path(d) for d, _ in sorted_ckpts[:k]]
            print(f"[avg] averaging top-{k} adapters:")
            for d, v in sorted_ckpts[:k]:
                print(f"  {Path(d).name}: loss={v:.4f}")
            try:
                average_lora_adapters(top_k_dirs, avg_dir)
                avg_loss = eval_adapter_val_loss(
                    args.base_model, avg_dir, val_ds, eval_dtype,
                    max_batches=64, batch_size=args.micro_batch_size,
                )
                print(f"[avg] eval_loss={avg_loss:.4f} vs best_single={best_single_loss:.4f}")
                if avg_loss < best_single_loss - 1e-6:
                    averaged_is_better = True
                    print("[avg] averaged adapter BETTER — using as best_adapter")
                else:
                    print("[avg] averaged not better — keeping best single checkpoint")
            except Exception as exc:
                print(f"[avg] failed: {exc} — falling back to best single checkpoint")

            checkpoint_scores["averaged_eval_loss"] = avg_loss
            checkpoint_scores["averaged_is_better"] = averaged_is_better

        checkpoint_scores["final_adapter_source"] = (
            "averaged" if averaged_is_better else "best_checkpoint"
        )

        # --- Save best_adapter (rank 0 only) ---
        if averaged_is_better and avg_dir.is_dir():
            if final_adapter_dir.exists():
                shutil.rmtree(final_adapter_dir)
            shutil.copytree(str(avg_dir), str(final_adapter_dir))
            print(f"[ckpt] best_adapter <- averaged ({final_adapter_dir})")
        else:
            averaged_is_better = False
            if best_single_dir and best_single_dir.is_dir():
                if final_adapter_dir.exists():
                    shutil.rmtree(final_adapter_dir)
                shutil.copytree(str(best_single_dir), str(final_adapter_dir))
                print(
                    f"[ckpt] best_adapter <- {best_single_dir.name} "
                    f"(eval_loss={best_single_loss:.4f})"
                )
            else:
                trainer.save_model(str(final_adapter_dir))
                print(f"[ckpt] best_adapter <- final trainer state ({final_adapter_dir})")

        tokenizer.save_pretrained(str(final_adapter_dir))

        # Write checkpoint_scores.json atomically
        tmp = output_dir / "checkpoint_scores.json.tmp"
        tmp.write_text(json.dumps(checkpoint_scores, indent=2))
        shutil.move(str(tmp), str(output_dir / "checkpoint_scores.json"))

    # ---- train_summary.json (rank 0 only) ----
    if is_main:
        train_loss: float | None = None
        eval_loss: float | None = None
        if trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if eval_loss is None and "eval_loss" in entry:
                    eval_loss = float(entry["eval_loss"])
                if train_loss is None and "loss" in entry and "eval_loss" not in entry:
                    train_loss = float(entry["loss"])
                if train_loss is not None and eval_loss is not None:
                    break

        summary = {
            "train_rows": len(train_ds),
            "val_rows": len(val_ds),
            "seq_len": args.seq_len,
            "lr": args.learning_rate,
            "epochs": args.epochs,
            "lr_scheduler_type": args.lr_scheduler_type,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "warmup_ratio": args.warmup_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "stable_ratio": args.stable_ratio,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_config_label": f"r{args.lora_r}:a{args.lora_alpha}:lr{args.learning_rate:g}",
            "adam_beta2": args.adam_beta2,
            "micro_batch_size": args.micro_batch_size,
            "grad_accum": args.grad_accum,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "dtype": args.dtype,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_target_modules": target_modules,
            "final_train_loss": train_loss,
            "final_eval_loss": eval_loss,
            "best_checkpoint_eval_loss": checkpoint_scores.get("best_checkpoint_eval_loss"),
            "averaged_eval_loss": checkpoint_scores.get("averaged_eval_loss"),
            "final_adapter_source": checkpoint_scores.get("final_adapter_source", "best_checkpoint"),
            "average_top_k": k_avg,
            "output_dir": args.output_dir,
            "elapsed_s": round(time.time() - t0, 1),
        }
        summary_path = output_dir / "train_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(
            f"[train] done in {summary['elapsed_s']}s | "
            f"train_loss={train_loss} eval_loss={eval_loss} "
            f"source={summary['final_adapter_source']}"
        )
        print(f"[train] summary → {summary_path}")


if __name__ == "__main__":
    main()
