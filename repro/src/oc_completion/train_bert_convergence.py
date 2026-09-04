"""Checkpoint-free BERT-LoRA convergence run for an OC task.

This is deliberately a single-process diagnostic, not a resumable production
trainer.  It trains one ``bert_lora`` seed on the full OC-Det or OC-Noisy
classification task, keeps the best trainable parameters in host RAM, and
evaluates that in-memory best model on both the ordinary test set and all
matched pairs.

No model, optimizer, scheduler, or RNG checkpoint is written.  Scalar
configuration/history/summary files are retained so that an early-stopped run
can be distinguished from an epoch-capped or walltime-stopped run.

The model is trained on individual sequences with observed labels (which are
flipped only for OC-Noisy).  The two members of a matched pair are scored
separately only for diagnostics and final mechanism evaluation; this is not
pairwise training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.oc_completion.eval_pairs import evaluate_artifact
from src.oc_completion.scoring import quick_pair_diag
from src.oc_completion.train_dl import DATA_ROOT, RESULTS_DIR, append_result, load_split
from src.oc_completion.train_hf import (
    ARMS,
    HFScorer,
    MAX_LENGTH,
    build_dataset,
    build_model,
    build_texts,
    build_tokenizer,
    evaluate,
    set_seed,
)


ARM = "bert_lora"
TASKS = ("ocdet", "ocnoisy")
LR=2e-05
WARMUP_RATIO=0.06
DEFAULT_MAX_EPOCHS = 80
DEFAULT_PATIENCE = 3
MAIN_PAIR_DATASET = "pairs_two_hole_heldout_val.csv"

_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None


def atomic_json_dump(payload: Any, path: Path) -> None:
    """Atomically replace a small JSON artifact."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def append_jsonl(payload: dict[str, Any], path: Path) -> None:
    """Persist one scalar epoch record without storing model state."""
    with open(path, "a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def save_predictions(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Save auditable predictions, with a CSV fallback when pyarrow is absent."""
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        csv_path = parquet_path.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False)
        return csv_path


def capture_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy only trainable parameters to CPU RAM.

    For BERT-LoRA this is the adapters plus the trainable classification head;
    the frozen pretrained backbone remains unchanged throughout the run.
    """
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
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


@dataclass
class EarlyStopping:
    """Strict validation-loss early stopping with an in-memory best state."""

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
    """Ask the epoch loop to stop after the active epoch."""
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    print(
        f"[convergence] received signal {signum}; finish the active epoch, "
        "then evaluate the in-memory best model",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        choices=TASKS,
        required=True,
        help="OC classification task; launchers must select it explicitly",
    )
    ap.add_argument("--seed", type=int, default=9550)
    ap.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--max_length", type=int, default=MAX_LENGTH)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--diag_pairs", type=int, default=2_000)
    ap.add_argument("--eval_batch", type=int, default=64)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument(
        "--require_cuda",
        action="store_true",
        help="fail instead of silently running the full diagnostic on CPU",
    )
    args = ap.parse_args()
    if args.max_epochs < 1:
        ap.error("--max_epochs must be positive")
    if args.patience < 1:
        ap.error("--patience must be positive")
    if args.threads < 1 or args.workers < 0:
        ap.error("--threads must be positive and --workers non-negative")
    if args.diag_pairs < 1 or args.eval_batch < 1:
        ap.error("--diag_pairs and --eval_batch must be positive")
    return args


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

    signal.signal(signal.SIGUSR1, request_graceful_stop)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("ERROR: the checkpoint-free FOX run requires a CUDA GPU")
    # Match the existing train_hf BERT path.  BERT is loaded in fp32 there;
    # the dtype argument is used by the Llama path only.
    model_dtype = torch.float32
    spec = ARMS[ARM]

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer, _ = build_tokenizer(spec["kind"], spec["model_name"])
    model = build_model(
        spec["kind"], spec["model_name"], spec["peft"], tokenizer, device,
        model_dtype,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    X_train, y_train, _ = load_split(args.task, "train", False)
    X_val, y_val, ystar_val = load_split(args.task, "val", False)
    X_test, y_test, ystar_test = load_split(args.task, "test", False)

    longest = max(build_texts(spec["kind"], X_train[:64]), key=len)
    token_length = len(tokenizer(longest)["input_ids"])
    if token_length >= args.max_length:
        raise ValueError(
            f"max_length {args.max_length} would truncate an input of "
            f"{token_length} tokens"
        )

    train_dataset = build_dataset(
        spec["kind"], build_texts(spec["kind"], X_train), y_train,
        tokenizer, args.max_length,
    )
    val_dataset = build_dataset(
        spec["kind"], build_texts(spec["kind"], X_val), y_val,
        tokenizer, args.max_length,
    )
    test_dataset = build_dataset(
        spec["kind"], build_texts(spec["kind"], X_test), y_test,
        tokenizer, args.max_length,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=spec["micro_batch"],
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch, num_workers=args.workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch, num_workers=args.workers
    )

    diagnostic_path = DATA_ROOT / "pairs" / MAIN_PAIR_DATASET
    diagnostic_pairs = pd.read_csv(diagnostic_path).head(args.diag_pairs)
    scorer = HFScorer(
        spec["kind"], model, tokenizer, args.max_length, device,
        args.eval_batch,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    accumulation = spec["grad_accum"]
    steps_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    total_steps = args.max_epochs * steps_per_epoch

    warmup_steps = int(WARMUP_RATIO * total_steps)
    
    def lr_multiplier(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_multiplier,
    )
    
    stopper = EarlyStopping(args.patience)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    termination_reason = "max_epochs"

    config = {
        "purpose": (
            f"checkpoint-free BERT-LoRA {args.task} convergence diagnostic"
        ),
        "arm": ARM,
        "task": args.task,
        "seed": args.seed,
        "model_name": spec["model_name"],
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.1,
            "targets": ["query", "key", "value"],
        },
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "optimizer": "AdamW",
        "learning_rate": LR,
        "micro_batch": spec["micro_batch"],
        "gradient_accumulation": accumulation,
        "effective_batch": spec["micro_batch"] * accumulation,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stopping_metric": "observed-label validation loss",
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": warmup_steps,
        "scheduler": "linear warmup then constant learning rate",
        "post_warmup_learning_rate": LR,
        "max_length": args.max_length,
        "device": str(device),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "n_params": n_params,
        "n_trainable": n_trainable,
        "checkpoint_policy": "none; best trainable state retained in host RAM",
        "pair_training": False,
        "pair_diagnostic": {
            "dataset": MAIN_PAIR_DATASET,
            "n": len(diagnostic_pairs),
            "affects_model_selection": False,
        },
    }
    atomic_json_dump(config, output_dir / "config.json")

    print("=== CHECKPOINT-FREE BERT CONVERGENCE RUN ===", flush=True)
    print(
        f"arm={ARM} task={args.task} seed={args.seed} device={device} "
        f"max_epochs={args.max_epochs} patience={args.patience}",
        flush=True,
    )
    print(
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"params={n_params} trainable={n_trainable}",
        flush=True,
    )
    print("checkpoint_policy=NONE", flush=True)

    started = time.time()
    for epoch_index in range(args.max_epochs):
        epoch = epoch_index + 1
        epoch_started = time.time()
        model.train()
        total_loss, seen = 0.0, 0
        optimizer.zero_grad()
        for batch_index, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            (loss / accumulation).backward()
            if ((batch_index + 1) % accumulation == 0
                    or (batch_index + 1) == len(train_loader)):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            batch_size = input_ids.shape[0]
            total_loss += loss.item() * batch_size
            seen += batch_size

        validation = evaluate(spec["kind"], model, val_loader, device)
        diagnostic = quick_pair_diag(scorer, diagnostic_pairs)
        improved, should_stop = stopper.observe(validation["loss"], epoch)
        if improved:
            best_state = capture_trainable_state(model)

        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            "val_loss": validation["loss"],
            "val_auc_observed": validation["auc"],
            "val_auc_latent": float(
                roc_auc_score(ystar_val, validation["probs"])
            ),
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
        print(
            f"[convergence] ep={epoch}/{args.max_epochs} "
            f"train={record['train_loss']:.6f} "
            f"val={record['val_loss']:.6f} "
            f"val_auc_obs={record['val_auc_observed']:.4f} "
            f"pair_acc={record['completion_pair_acc']:.4f} "
            f"best_ep={stopper.best_epoch} stale={stopper.no_improve} "
            f"lr={record['learning_rate']:.3e}",
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
            termination_reason = (
                "slurm_signal_usr1"
                if _STOP_SIGNAL == signal.SIGUSR1
                else f"signal_{_STOP_SIGNAL}"
            )
            print(
                f"[convergence] graceful walltime stop after epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("no completed epoch produced an in-memory best state")
    restore_trainable_state(model, best_state)
    del best_state

    from src.experiments.BERT_fraction_experiment import find_optimal_threshold

    best_validation = evaluate(spec["kind"], model, val_loader, device)
    threshold, val_f1_optimal = find_optimal_threshold(
        best_validation["labels"], best_validation["probs"]
    )
    test = evaluate(spec["kind"], model, test_loader, device)
    predictions = (test["probs"] >= threshold).astype(int)
    latent_auc = float(roc_auc_score(ystar_test, test["probs"]))
    validation_latent_auc = float(
        roc_auc_score(ystar_val, best_validation["probs"])
    )
    elapsed = time.time() - started

    validation_predictions_path = save_predictions(pd.DataFrame(
        {
            "observed_label": best_validation["labels"],
            "latent_label": np.asarray(ystar_val),
            "probability_positive": best_validation["probs"],
        }
    ), output_dir / "validation_predictions.parquet")
    test_predictions_path = save_predictions(pd.DataFrame(
        {
            "observed_label": test["labels"],
            "latent_label": np.asarray(ystar_test),
            "probability_positive": test["probs"],
            "predicted_label": predictions,
        }
    ), output_dir / "test_predictions.parquet")

    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": ARM,
        "training_mode": "lora_checkpoint_free_convergence",
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
            float(precision_score(test["labels"], predictions, zero_division=0)), 6
        ),
        "test_recall": round(
            float(recall_score(test["labels"], predictions, zero_division=0)), 6
        ),
        "threshold": round(float(threshold), 4),
        "n_params": n_params,
        "recipe": (
            f"checkpoint-free {ARM} {spec['model_name']} "
            f"LoRA(r=8,a=16,drop=.1) AdamW lr={LR} "
            f"eff_bs={spec['micro_batch'] * accumulation} "
            f"warmup=6% then constant_lr={LR} epochs<={args.max_epochs} "
            f"patience={args.patience} early=val_loss max_len={args.max_length}"
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
        "model": ARM,
        "task": args.task,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "termination_reason": termination_reason,
        "converged_by_patience": termination_reason == "early_stopping",
        "checkpoint_saved": False,
        "epochs_completed": len(history),
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
    atomic_json_dump(summary, output_dir / "summary.json")

    print("[convergence] evaluating all matched pairs in memory", flush=True)
    pair_rows = evaluate_artifact(
        scorer,
        {
            "model": ARM,
            "seed": args.seed,
            "training_mode": "lora_checkpoint_free_convergence",
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
    atomic_json_dump(summary, output_dir / "summary.json")

    print("=== CONVERGENCE RUN COMPLETE ===", flush=True)
    print(
        f"termination={termination_reason} epochs={len(history)} "
        f"best_epoch={stopper.best_epoch} "
        f"test_auc_obs={test['auc']:.4f} "
        f"test_auc_latent={latent_auc:.4f}",
        flush=True,
    )
    print("checkpoint_saved=False", flush=True)
    print(f"results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
