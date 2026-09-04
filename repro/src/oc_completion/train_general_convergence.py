"""Checkpoint-free convergence training for configurable Hugging Face models.

This runner generalizes the BERT-only convergence diagnostic to encoder
sequence classifiers and causal decoder models (for example Llama and Qwen).
It trains one model/task/seed in a single process, selects the best epoch by
observed-label validation loss, retains only the trainable parameters of that
best epoch in host RAM, and evaluates ordinary and matched-pair test sets
before the process exits.

No model, optimizer, scheduler, or RNG checkpoint is written.  This is an
intentional convergence diagnostic rather than a resumable production job.
When Slurm sends SIGUSR1, an incomplete epoch is abandoned at the next batch
boundary and the best previously completed epoch is finalized.  Consequently,
at least one full epoch must complete inside the allocation.

The default learning-rate schedule is linear warmup for 6% of the configured
optimizer steps followed by a constant learning rate.  Unlike the original
``train_hf`` protocol, there is no post-warmup decay.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.oc_completion.eval_pairs import evaluate_artifact
from src.oc_completion.scoring import quick_pair_diag
from src.oc_completion.train_dl import DATA_ROOT, RESULTS_DIR, append_result, load_split
from src.oc_completion.train_hf import (
    HFScorer,
    MAX_LENGTH,
    build_dataset,
    build_texts,
    evaluate,
    forward_batch,
    set_seed,
)


TASKS = ("ocdet", "ocnoisy")
MODEL_KINDS = ("bert", "causal")
DEFAULT_MAX_EPOCHS = 80
DEFAULT_PATIENCE = 3
DEFAULT_LR = 2e-5
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM = 2
MAIN_PAIR_DATASET = "pairs_two_hole_heldout_val.csv"
HF_CACHE = os.environ.get("HF_HUB_CACHE", "/root/hf_cache")

_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None


def atomic_json_dump(payload: Any, path: Path) -> None:
    """Atomically replace a small JSON artifact."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


def append_jsonl(payload: dict[str, Any], path: Path) -> None:
    """Durably append one scalar epoch record."""
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_predictions(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Save predictions as Parquet, with a compressed CSV fallback."""
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        csv_path = parquet_path.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False)
        return csv_path


def safe_model_tag(model_name: str) -> str:
    """Return a filesystem-safe, deterministic label for a model identifier."""
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-._")
    if not tag:
        raise ValueError(f"cannot derive a safe model tag from {model_name!r}")
    return tag


def capture_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy trainable parameters to independent CPU tensors."""
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("model exposes no trainable parameters")
    return state


def restore_trainable_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Restore a state produced by :func:`capture_trainable_state`."""
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if trainable.keys() != state.keys():
        missing = sorted(state.keys() - trainable.keys())
        extra = sorted(trainable.keys() - state.keys())
        raise ValueError(
            f"trainable parameter set changed; missing={missing}, extra={extra}"
        )
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(
                state[name].to(device=parameter.device, dtype=parameter.dtype)
            )


def build_warmup_then_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> tuple[torch.optim.lr_scheduler.LRScheduler, int]:
    """Return linear warmup followed by a constant learning rate."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not math.isfinite(warmup_ratio) or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be finite and in [0, 1)")
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_multiplier(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    return scheduler, warmup_steps


def accumulation_group_size(
    batch_index: int, n_batches: int, grad_accum: int
) -> int:
    """Return the actual size of an accumulation group, including its tail."""
    if n_batches < 1 or grad_accum < 1:
        raise ValueError("n_batches and grad_accum must be positive")
    if not 0 <= batch_index < n_batches:
        raise ValueError("batch_index is outside the loader")
    group_start = (batch_index // grad_accum) * grad_accum
    return min(grad_accum, n_batches - group_start)


@dataclass
class EarlyStopping:
    """Strict validation-loss early stopping."""

    patience: int
    best_loss: float = math.inf
    best_epoch: int = 0
    no_improve: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")

    def observe(self, value: float, epoch: int) -> tuple[bool, bool]:
        if not math.isfinite(value):
            raise ValueError(f"non-finite validation loss at epoch {epoch}: {value}")
        improved = value < self.best_loss
        if improved:
            self.best_loss = value
            self.best_epoch = epoch
            self.no_improve = 0
        else:
            self.no_improve += 1
        return improved, self.no_improve >= self.patience


def request_graceful_stop(signum: int, _frame: Any) -> None:
    """Request finalization of the best previously completed epoch."""
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    print(
        f"[convergence] received signal {signum}; stop at the next batch "
        "boundary and finalize the best completed epoch",
        flush=True,
    )


def default_lora_targets(model_kind: str) -> tuple[str, ...]:
    if model_kind == "bert":
        return ("query", "key", "value")
    return ("q_proj", "k_proj", "v_proj", "o_proj")


def parse_lora_targets(raw: str, model_kind: str) -> tuple[str, ...]:
    if not raw.strip():
        return default_lora_targets(model_kind)
    targets = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not targets:
        raise ValueError("--lora_targets must contain at least one module name")
    if len(set(targets)) != len(targets):
        raise ValueError("--lora_targets contains duplicate module names")
    return targets


def build_tokenizer(
    model_name: str,
    model_revision: str,
    trust_remote_code: bool,
):
    """Load a tokenizer using the same explicit PAD-token policy as train_hf."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        cache_dir=HF_CACHE,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def build_selected_model(
    *,
    model_kind: str,
    model_name: str,
    model_revision: str,
    use_lora: bool,
    lora_targets: Sequence[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    gradient_checkpointing: bool,
    trust_remote_code: bool,
) -> torch.nn.Module:
    """Build an encoder classifier or causal LM plus classification head."""
    from peft import LoraConfig, get_peft_model

    common = {
        "revision": model_revision,
        "cache_dir": HF_CACHE,
        "trust_remote_code": trust_remote_code,
    }

    if model_kind == "bert":
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            **common,
        )
        model.resize_token_embeddings(len(tokenizer))
        if gradient_checkpointing:
            if not getattr(model, "supports_gradient_checkpointing", False):
                raise ValueError(
                    f"{model_name} does not support gradient checkpointing"
                )
            model.gradient_checkpointing_enable()
        if use_lora:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(lora_targets),
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
            )
            model = get_peft_model(model, config)
            if gradient_checkpointing and hasattr(
                model, "enable_input_require_grads"
            ):
                model.enable_input_require_grads()
    else:
        from transformers import AutoModelForCausalLM

        from src.experiments.LLM_fraction_experiment import (
            CausalLMWithClassificationHead,
        )

        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            tie_word_embeddings=True,
            **common,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        base.resize_token_embeddings(len(tokenizer))
        if gradient_checkpointing:
            if not getattr(base, "supports_gradient_checkpointing", False):
                raise ValueError(
                    f"{model_name} does not support gradient checkpointing"
                )
            base.config.use_cache = False
            base.gradient_checkpointing_enable()
        model = CausalLMWithClassificationHead(base, num_classes=2)
        if use_lora:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(lora_targets),
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.backbone = get_peft_model(model.backbone, config)
            if gradient_checkpointing and hasattr(
                model.backbone, "enable_input_require_grads"
            ):
                model.backbone.enable_input_require_grads()

    return model.to(device)


def find_optimal_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Select the validation threshold that maximizes F1."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.10, 0.90, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = float(f1_score(labels, predictions, zero_division=0))
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = score
    return best_threshold, best_f1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--model_kind", choices=MODEL_KINDS, required=True)
    ap.add_argument("--model_revision", default="main")
    ap.add_argument("--model_tag", default=None)
    ap.add_argument(
        "--lora", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    ap.add_argument("--lora_targets", default="")
    ap.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--seed", type=int, default=9550)
    ap.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument(
        "--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO
    )
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--diag_pairs", type=int, default=2_000)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--require_cuda",
        action="store_true",
        help="fail rather than silently run the full diagnostic on CPU",
    )
    args = ap.parse_args()

    positive_ints = {
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "threads": args.threads,
        "diag_pairs": args.diag_pairs,
        "eval_batch": args.eval_batch,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    for name, value in positive_ints.items():
        if value < 1:
            ap.error(f"--{name} must be positive")
    if args.workers < 0:
        ap.error("--workers must be non-negative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        ap.error("--lr must be finite and positive")
    if not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio < 1:
        ap.error("--warmup_ratio must be finite and in [0, 1)")
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        ap.error("--lora_dropout must be finite and in [0, 1)")
    if not args.model_name.strip():
        ap.error("--model_name must not be empty")
    if not args.model_revision.strip():
        ap.error("--model_revision must not be empty")
    if args.model_tag is not None and not re.fullmatch(
        r"[A-Za-z0-9._-]+", args.model_tag
    ):
        ap.error("--model_tag may contain only letters, numbers, '.', '_', '-'")
    try:
        args.lora_targets_parsed = parse_lora_targets(
            args.lora_targets, args.model_kind
        )
    except ValueError as exc:
        ap.error(str(exc))
    return args


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(set(np.asarray(labels).tolist())) < 2:
        return 0.5
    return float(roc_auc_score(labels, probabilities))


def _termination_from_signal() -> str:
    if _STOP_SIGNAL == signal.SIGUSR1:
        return "slurm_signal_usr1"
    return f"signal_{_STOP_SIGNAL}"


def main() -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = False
    _STOP_SIGNAL = None
    args = parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"ERROR: output directory is not empty; refusing overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = output_dir / "history.jsonl"
    summary_path = output_dir / "summary.json"

    signal.signal(signal.SIGUSR1, request_graceful_stop)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("ERROR: this convergence run requires a CUDA GPU")
    dtype = (
        torch.bfloat16
        if args.model_kind == "causal" and device.type == "cuda"
        else torch.float32
    )

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = build_tokenizer(
        args.model_name, args.model_revision, args.trust_remote_code
    )
    model = build_selected_model(
        model_kind=args.model_kind,
        model_name=args.model_name,
        model_revision=args.model_revision,
        use_lora=args.lora,
        lora_targets=args.lora_targets_parsed,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        trust_remote_code=args.trust_remote_code,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    n_trainable = sum(parameter.numel() for parameter in trainable_parameters)
    trainable_state_gib = sum(
        parameter.numel() * parameter.element_size()
        for parameter in trainable_parameters
    ) / 2**30
    resolved_revision = getattr(model.config, "_commit_hash", None)

    X_train, y_train, _ = load_split(args.task, "train", False)
    X_val, y_val, ystar_val = load_split(args.task, "val", False)
    X_test, y_test, ystar_test = load_split(args.task, "test", False)

    length_probe = build_texts(args.model_kind, X_train[:256])
    token_length = max(len(tokenizer(text)["input_ids"]) for text in length_probe)
    if token_length >= args.max_length:
        raise ValueError(
            f"max_length {args.max_length} would truncate an input of "
            f"{token_length} tokens"
        )

    train_dataset = build_dataset(
        args.model_kind,
        build_texts(args.model_kind, X_train),
        y_train,
        tokenizer,
        args.max_length,
    )
    val_dataset = build_dataset(
        args.model_kind,
        build_texts(args.model_kind, X_val),
        y_val,
        tokenizer,
        args.max_length,
    )
    test_dataset = build_dataset(
        args.model_kind,
        build_texts(args.model_kind, X_test),
        y_test,
        tokenizer,
        args.max_length,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch,
        **loader_options,
    )

    diagnostic_path = DATA_ROOT / "pairs" / MAIN_PAIR_DATASET
    diagnostic_pairs = pd.read_csv(diagnostic_path).head(args.diag_pairs)
    scorer = HFScorer(
        args.model_kind,
        model,
        tokenizer,
        args.max_length,
        device,
        args.eval_batch,
    )

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    steps_per_epoch = (
        len(train_loader) + args.grad_accum - 1
    ) // args.grad_accum
    total_steps = args.max_epochs * steps_per_epoch
    scheduler, warmup_steps = build_warmup_then_constant_scheduler(
        optimizer, total_steps, args.warmup_ratio
    )
    stopper = EarlyStopping(args.patience)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    termination_reason = "max_epochs"
    incomplete_epoch: dict[str, Any] | None = None
    model_tag = args.model_tag or safe_model_tag(args.model_name)
    model_id = f"{model_tag}_{'lora' if args.lora else 'full'}"
    training_mode = (
        "lora_checkpoint_free_convergence"
        if args.lora
        else "full_ft_checkpoint_free_convergence"
    )

    config = {
        "purpose": "checkpoint-free configurable HF convergence diagnostic",
        "model": model_id,
        "model_name": args.model_name,
        "model_kind": args.model_kind,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": resolved_revision,
        "trust_remote_code": args.trust_remote_code,
        "task": args.task,
        "seed": args.seed,
        "lora": {
            "enabled": args.lora,
            "r": args.lora_r if args.lora else None,
            "alpha": args.lora_alpha if args.lora else None,
            "dropout": args.lora_dropout if args.lora else None,
            "targets": list(args.lora_targets_parsed) if args.lora else None,
        },
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stopping_metric": "observed-label validation loss",
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "scheduler": "linear warmup then constant learning rate",
        "max_length": args.max_length,
        "eval_batch": args.eval_batch,
        "device": str(device),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "n_params": n_params,
        "n_trainable": n_trainable,
        "trainable_state_gib": trainable_state_gib,
        "checkpoint_policy": "none; best trainable state retained in host RAM",
        "pair_training": False,
        "pair_diagnostic": {
            "dataset": MAIN_PAIR_DATASET,
            "n": len(diagnostic_pairs),
            "affects_model_selection": False,
        },
    }
    atomic_json_dump(config, output_dir / "config.json")

    print("=== CHECKPOINT-FREE GENERAL HF CONVERGENCE RUN ===", flush=True)
    print(
        f"model={model_id} hf={args.model_name}@{args.model_revision} "
        f"kind={args.model_kind} lora={args.lora} task={args.task} "
        f"seed={args.seed} device={device}",
        flush=True,
    )
    print(
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"params={n_params} trainable={n_trainable} "
        f"best_state_gib={trainable_state_gib:.3f}",
        flush=True,
    )
    print(
        f"epochs<={args.max_epochs} patience={args.patience} "
        f"batch={args.batch_size} accum={args.grad_accum} "
        f"effective_batch={args.batch_size * args.grad_accum}",
        flush=True,
    )
    print(
        f"optimizer=AdamW peak_lr={args.lr} warmup_ratio={args.warmup_ratio} "
        f"warmup_steps={warmup_steps} post_warmup_schedule=constant",
        flush=True,
    )
    print("checkpoint_policy=NONE", flush=True)

    started = time.time()
    for epoch_index in range(args.max_epochs):
        epoch = epoch_index + 1
        epoch_started = time.time()
        model.train()
        total_loss, seen, batches_completed = 0.0, 0, 0
        interrupted = False
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            if _STOP_REQUESTED:
                interrupted = True
                break
            loss, _ = forward_batch(
                args.model_kind, model, batch, device, with_loss=True
            )
            group_size = accumulation_group_size(
                batch_index, len(train_loader), args.grad_accum
            )
            (loss / group_size).backward()
            if (
                (batch_index + 1) % args.grad_accum == 0
                or (batch_index + 1) == len(train_loader)
            ):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            batch_rows = batch["input_ids"].shape[0]
            total_loss += loss.item() * batch_rows
            seen += batch_rows
            batches_completed = batch_index + 1
            if _STOP_REQUESTED and batches_completed < len(train_loader):
                interrupted = True
                break

        if interrupted:
            optimizer.zero_grad(set_to_none=True)
            termination_reason = _termination_from_signal()
            incomplete_epoch = {
                "epoch": epoch,
                "batches_completed": batches_completed,
                "batches_total": len(train_loader),
                "rows_seen": seen,
                "fraction_completed": batches_completed / len(train_loader),
            }
            print(
                f"[convergence] abandoning partial epoch {epoch}: "
                f"batches={batches_completed}/{len(train_loader)}; "
                "the best completed epoch will be restored",
                flush=True,
            )
            break

        validation = evaluate(args.model_kind, model, val_loader, device)
        improved, should_stop = stopper.observe(validation["loss"], epoch)
        if improved:
            best_state = capture_trainable_state(model)

        if _STOP_REQUESTED:
            diagnostic = {
                "completion_pair_acc": None,
                "completion_mean_margin": None,
            }
        else:
            diagnostic = quick_pair_diag(scorer, diagnostic_pairs)

        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "val_loss": validation["loss"],
            "val_auc_observed": validation["auc"],
            "val_auc_latent": _auc(ystar_val, validation["probs"]),
            "val_f1_at_0.5": validation["f1"],
            **diagnostic,
            "improved": improved,
            "best_epoch": stopper.best_epoch,
            "best_val_loss": stopper.best_loss,
            "no_improve": stopper.no_improve,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        append_jsonl(record, history_jsonl)
        pair_text = (
            "skipped"
            if record["completion_pair_acc"] is None
            else f"{record['completion_pair_acc']:.4f}"
        )
        print(
            f"[convergence] ep={epoch}/{args.max_epochs} "
            f"train={record['train_loss']:.6f} "
            f"val={record['val_loss']:.6f} "
            f"val_auc_obs={record['val_auc_observed']:.4f} "
            f"pair_acc={pair_text} best_ep={stopper.best_epoch} "
            f"stale={stopper.no_improve} lr={record['learning_rate']:.3e}",
            flush=True,
        )

        if should_stop:
            termination_reason = "early_stopping"
            print(
                f"[convergence] early stopping after epoch {epoch}; "
                f"best epoch was {stopper.best_epoch}",
                flush=True,
            )
            break
        if _STOP_REQUESTED:
            termination_reason = _termination_from_signal()
            print(
                f"[convergence] walltime finalization after completed epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        failure = {
            "status": "incomplete_no_completed_epoch",
            "model": model_id,
            "model_name": args.model_name,
            "task": args.task,
            "seed": args.seed,
            "termination_reason": termination_reason,
            "checkpoint_saved": False,
            "incomplete_epoch": incomplete_epoch,
        }
        atomic_json_dump(failure, summary_path)
        raise RuntimeError(
            "no completed epoch produced an in-memory best state; reduce the "
            "per-epoch workload or use a resumable training protocol"
        )

    restore_trainable_state(model, best_state)
    del best_state, optimizer, scheduler, train_loader, train_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_validation = evaluate(args.model_kind, model, val_loader, device)
    threshold, val_f1_optimal = find_optimal_threshold(
        best_validation["labels"], best_validation["probs"]
    )
    test = evaluate(args.model_kind, model, test_loader, device)
    predictions = (test["probs"] >= threshold).astype(int)
    latent_auc = _auc(ystar_test, test["probs"])
    validation_latent_auc = _auc(ystar_val, best_validation["probs"])
    elapsed = time.time() - started

    validation_predictions_path = save_predictions(
        pd.DataFrame(
            {
                "observed_label": best_validation["labels"],
                "latent_label": np.asarray(ystar_val),
                "probability_positive": best_validation["probs"],
            }
        ),
        output_dir / "validation_predictions.parquet",
    )
    test_predictions_path = save_predictions(
        pd.DataFrame(
            {
                "observed_label": test["labels"],
                "latent_label": np.asarray(ystar_test),
                "probability_positive": test["probs"],
                "predicted_label": predictions,
            }
        ),
        output_dir / "test_predictions.parquet",
    )

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_id,
        "training_mode": training_mode,
        "task": args.task,
        "seed": args.seed,
        "train_rows": len(X_train),
        "epochs_done": len(history),
        "best_epoch": stopper.best_epoch,
        "val_f1": round(float(val_f1_optimal), 6),
        "val_auc": round(float(best_validation["auc"]), 6),
        "val_loss": round(float(best_validation["loss"]), 6),
        "test_auc_obs": round(float(test["auc"]), 6),
        "test_auc_latent": round(latent_auc, 6),
        "test_f1": round(
            float(f1_score(test["labels"], predictions, zero_division=0)), 6
        ),
        "test_precision": round(
            float(
                precision_score(test["labels"], predictions, zero_division=0)
            ),
            6,
        ),
        "test_recall": round(
            float(recall_score(test["labels"], predictions, zero_division=0)),
            6,
        ),
        "threshold": round(float(threshold), 4),
        "n_params": n_params,
        "recipe": (
            f"checkpoint-free {args.model_name}@{args.model_revision} "
            f"kind={args.model_kind} lora={args.lora} "
            f"AdamW lr={args.lr} batch={args.batch_size} "
            f"accum={args.grad_accum} effective_batch="
            f"{args.batch_size * args.grad_accum} warmup="
            f"{args.warmup_ratio:g} then constant epochs<={args.max_epochs} "
            f"patience={args.patience} early=val_loss "
            f"max_len={args.max_length} dtype={dtype}"
        ),
        "wallclock_s": round(elapsed, 1),
        "checkpoint": (
            "NOT_SAVED; best trainable state evaluated in memory; "
            f"run={output_dir.name}"
        ),
        "smoke": False,
        "host": os.uname().nodename,
    }
    append_result(RESULTS_DIR / "training_results.csv", row)
    atomic_json_dump(history, output_dir / "history.json")

    summary: dict[str, Any] = {
        "status": "standard_evaluation_complete",
        "model": model_id,
        "model_name": args.model_name,
        "model_kind": args.model_kind,
        "lora": args.lora,
        "task": args.task,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "termination_reason": termination_reason,
        "converged_by_patience": termination_reason == "early_stopping",
        "checkpoint_saved": False,
        "epochs_completed": len(history),
        "incomplete_epoch": incomplete_epoch,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
        "best_val_auc_observed": float(best_validation["auc"]),
        "best_val_auc_latent": validation_latent_auc,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "standard_metrics": row,
        "validation_predictions": str(validation_predictions_path),
        "test_predictions": str(test_predictions_path),
        "pair_evaluation": None,
    }
    atomic_json_dump(summary, summary_path)

    print("[convergence] evaluating all matched pairs in memory", flush=True)
    pair_rows = evaluate_artifact(
        scorer,
        {
            "model": model_id,
            "seed": args.seed,
            "training_mode": training_mode,
        },
        args.task,
        False,
        ["val", "test"],
        (
            "NOT_SAVED; in-memory best trainable state; "
            f"run={output_dir.name}; best_epoch={stopper.best_epoch}"
        ),
        out_dir=RESULTS_DIR / "pair_predictions",
    )
    summary["status"] = "complete"
    summary["pair_evaluation"] = pair_rows
    summary["total_wallclock_s"] = round(time.time() - started, 1)
    atomic_json_dump(summary, summary_path)

    print("=== GENERAL CONVERGENCE RUN COMPLETE ===", flush=True)
    print(
        f"model={model_id} task={args.task} termination={termination_reason} "
        f"epochs={len(history)} best_epoch={stopper.best_epoch} "
        f"test_auc_obs={test['auc']:.4f} test_auc_latent={latent_auc:.4f}",
        flush=True,
    )
    print("checkpoint_saved=False", flush=True)
    print(f"results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
