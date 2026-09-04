"""Checkpoint-free convergence training for the small OC sequence models.

This runner trains one of ``LSTM``, ``Transformer`` or ``RNNTransformer`` on
one OC task and one seed.  It deliberately differs from the reproducibility
trainer in :mod:`src.oc_completion.train_dl` in two ways required by the
convergence diagnostic:

* the best epoch is selected by observed-label validation loss; and
* no model, optimizer, scheduler or RNG checkpoint is written.

The best completed model state is retained only in host RAM.  A SIGUSR1
request abandons an incomplete epoch at the next batch boundary, restores the
best fully completed epoch and proceeds to ordinary and matched-pair
evaluation.  At least one full epoch must therefore finish before the signal.
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
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.oc_completion.eval_pairs import evaluate_artifact
from src.oc_completion.scoring import quick_pair_diag
from src.oc_completion.train_dl import (
    DATA_ROOT,
    OC_ENCODING,
    RECIPES,
    RESULTS_DIR,
    append_result,
    create_model,
    evaluate,
    load_split,
    make_loader,
    model_logits,
)


MODELS = ("LSTM", "Transformer", "RNNTransformer")
TASKS = ("ocdet", "ocnoisy")
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_EVAL_BATCH = 1024
DEFAULT_DIAG_PAIRS = 2_000
MAIN_PAIR_DATASET = "pairs_two_hole_heldout_val.csv"

_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None


def atomic_json_dump(payload: Any, path: Path) -> None:
    """Atomically replace a small JSON artifact."""
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(temporary, path)


def append_jsonl(payload: dict[str, Any], path: Path) -> None:
    """Durably append one epoch record."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def save_predictions(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Save predictions as Parquet, with compressed CSV as a fallback."""
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        csv_path = parquet_path.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False)
        return csv_path


def capture_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy the complete model state to independent CPU tensors."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def restore_model_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    """Restore a state produced by :func:`capture_model_state`."""
    model.load_state_dict(state, strict=True)


@dataclass
class EarlyStopping:
    """Strict observed-label validation-loss early stopping."""

    patience: int
    best_loss: float = math.inf
    best_epoch: int = 0
    no_improve: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be positive")

    def observe(self, value: float, epoch: int) -> tuple[bool, bool]:
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite validation loss at epoch {epoch}: {value}"
            )
        improved = value < self.best_loss
        if improved:
            self.best_loss = value
            self.best_epoch = epoch
            self.no_improve = 0
        else:
            self.no_improve += 1
        return improved, self.no_improve >= self.patience


def request_graceful_stop(signum: int, _frame: Any) -> None:
    """Ask the training loop to finalize its best completed epoch."""
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    print(
        f"[dl-convergence] received signal {signum}; stopping at the next "
        "batch boundary and finalizing the best completed epoch",
        flush=True,
    )


def termination_from_signal() -> str:
    if _STOP_SIGNAL == signal.SIGUSR1:
        return "slurm_signal_usr1"
    return f"signal_{_STOP_SIGNAL}"


def auc(labels: list[int] | np.ndarray, probabilities: np.ndarray) -> float:
    labels_array = np.asarray(labels, dtype=int)
    if len(np.unique(labels_array)) < 2:
        return 0.5
    return float(roc_auc_score(labels_array, probabilities))


def classification_metrics(
    labels: list[int] | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=int)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auc": auc(labels_array, probabilities),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "precision": float(
            precision_score(labels_array, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
    }


def find_optimal_threshold(
    labels: list[int] | np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Select the validation threshold that maximizes observed-label F1."""
    best_threshold = 0.5
    best_f1 = -1.0
    labels_array = np.asarray(labels, dtype=int)
    for threshold in np.arange(0.10, 0.90, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = float(f1_score(labels_array, predictions, zero_division=0))
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = score
    return best_threshold, best_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seed", type=int, default=9550)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="constant Adam learning rate (default: model RECIPES value)",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--diag_pairs", type=int, default=DEFAULT_DIAG_PAIRS)
    parser.add_argument("--eval_batch", type=int, default=DEFAULT_EVAL_BATCH)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--require_cuda",
        action="store_true",
        help="fail instead of silently running this FOX diagnostic on CPU",
    )
    args = parser.parse_args()

    positive_values = {
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "diag_pairs": args.diag_pairs,
        "eval_batch": args.eval_batch,
    }
    for name, value in positive_values.items():
        if value < 1:
            parser.error(f"--{name} must be positive")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("--lr must be finite and positive")
    return args


def main() -> None:
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = False
    _STOP_SIGNAL = None
    args = parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"ERROR: output directory is not empty; refusing overwrite: "
            f"{output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_jsonl = output_dir / "history.jsonl"
    summary_path = output_dir / "summary.json"

    signal.signal(signal.SIGUSR1, request_graceful_stop)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise SystemExit("ERROR: this convergence run requires a CUDA GPU")

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    recipe = RECIPES[args.model]
    learning_rate = recipe["lr"] if args.lr is None else args.lr
    learning_rate_source = "RECIPES" if args.lr is None else "command_line"
    model = create_model(args.model, recipe["config"]).to(device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    state_gib = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    ) / 2**30

    X_train, y_train, _ = load_split(args.task, "train", False)
    X_val, y_val, ystar_val = load_split(args.task, "val", False)
    X_test, y_test, ystar_test = load_split(args.task, "test", False)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = make_loader(
        X_train,
        y_train,
        True,
        batch_size=args.batch_size,
        generator=generator,
    )
    val_loader = make_loader(
        X_val, y_val, False, batch_size=args.eval_batch
    )
    test_loader = make_loader(
        X_test, y_test, False, batch_size=args.eval_batch
    )

    diagnostic_path = DATA_ROOT / "pairs" / MAIN_PAIR_DATASET
    diagnostic_pairs = pd.read_csv(diagnostic_path).head(args.diag_pairs)

    def score_sequences(sequences: list[str]) -> np.ndarray:
        return model_logits(
            model, sequences, device, batch_size=args.eval_batch
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(args.patience)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    termination_reason = "max_epochs"
    incomplete_epoch: dict[str, Any] | None = None
    training_mode = "scratch_checkpoint_free_convergence"

    config = {
        "purpose": "checkpoint-free configurable DL convergence diagnostic",
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "recipe": recipe,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "learning_rate_source": learning_rate_source,
        "scheduler": "none; learning rate constant from the first update",
        "batch_size": args.batch_size,
        "eval_batch": args.eval_batch,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "early_stopping_metric": "observed-label validation loss",
        "threads": args.threads,
        "device": str(device),
        "n_params": n_params,
        "best_state_gib": state_gib,
        "checkpoint_policy": "none; best model state retained in host RAM",
        "pair_training": False,
        "pair_diagnostic": {
            "dataset": MAIN_PAIR_DATASET,
            "n": len(diagnostic_pairs),
            "affects_model_selection": False,
        },
    }
    atomic_json_dump(config, output_dir / "config.json")

    print("=== CHECKPOINT-FREE DL CONVERGENCE RUN ===", flush=True)
    print(
        f"model={args.model} task={args.task} seed={args.seed} "
        f"device={device} params={n_params}",
        flush=True,
    )
    print(
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"epochs<={args.max_epochs} patience={args.patience} "
        f"batch={args.batch_size}",
        flush=True,
    )
    print(
        f"optimizer=Adam constant_lr={learning_rate:g} "
        f"lr_source={learning_rate_source}",
        flush=True,
    )
    print("checkpoint_policy=NONE", flush=True)

    started = time.time()
    for epoch_index in range(args.max_epochs):
        epoch = epoch_index + 1
        epoch_started = time.time()
        model.train()
        total_loss = 0.0
        rows_seen = 0
        batches_completed = 0
        interrupted = False

        for batch_index, (inputs, labels) in enumerate(train_loader):
            if _STOP_REQUESTED:
                interrupted = True
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_rows = len(labels)
            total_loss += loss.item() * batch_rows
            rows_seen += batch_rows
            batches_completed = batch_index + 1
            if _STOP_REQUESTED and batches_completed < len(train_loader):
                interrupted = True
                break

        if interrupted:
            optimizer.zero_grad(set_to_none=True)
            termination_reason = termination_from_signal()
            incomplete_epoch = {
                "epoch": epoch,
                "batches_completed": batches_completed,
                "batches_total": len(train_loader),
                "rows_seen": rows_seen,
                "fraction_completed": batches_completed / len(train_loader),
            }
            print(
                f"[dl-convergence] abandoning partial epoch {epoch}: "
                f"batches={batches_completed}/{len(train_loader)}; restoring "
                "the best completed epoch",
                flush=True,
            )
            break

        validation = evaluate(model, val_loader, device, criterion)
        improved, should_stop = stopper.observe(validation["loss"], epoch)
        if improved:
            best_state = capture_model_state(model)

        if _STOP_REQUESTED:
            diagnostic = {
                "completion_pair_acc": None,
                "completion_mean_margin": None,
            }
        else:
            diagnostic = quick_pair_diag(score_sequences, diagnostic_pairs)

        record = {
            "epoch": epoch,
            "train_loss": total_loss / rows_seen,
            "val_loss": validation["loss"],
            "val_auc_observed": validation["auc"],
            "val_auc_latent": auc(ystar_val, validation["probs"]),
            "val_f1_at_0.5": validation["f1"],
            **diagnostic,
            "improved": improved,
            "best_epoch": stopper.best_epoch,
            "best_val_loss": stopper.best_loss,
            "no_improve": stopper.no_improve,
            "learning_rate": learning_rate,
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
            f"[dl-convergence] ep={epoch}/{args.max_epochs} "
            f"train={record['train_loss']:.6f} "
            f"val={record['val_loss']:.6f} "
            f"val_auc_obs={record['val_auc_observed']:.4f} "
            f"pair_acc={pair_text} best_ep={stopper.best_epoch} "
            f"stale={stopper.no_improve} lr={learning_rate:.3e}",
            flush=True,
        )

        if should_stop:
            termination_reason = "early_stopping"
            print(
                f"[dl-convergence] early stopping after epoch {epoch}; "
                f"best epoch was {stopper.best_epoch}",
                flush=True,
            )
            break
        if _STOP_REQUESTED:
            termination_reason = termination_from_signal()
            print(
                f"[dl-convergence] walltime finalization after completed "
                f"epoch {epoch}",
                flush=True,
            )
            break

    if best_state is None:
        failure = {
            "status": "incomplete_no_completed_epoch",
            "model": args.model,
            "task": args.task,
            "seed": args.seed,
            "termination_reason": termination_reason,
            "checkpoint_saved": False,
            "incomplete_epoch": incomplete_epoch,
        }
        atomic_json_dump(failure, summary_path)
        raise RuntimeError(
            "no completed epoch produced an in-memory best state; reduce the "
            "per-epoch workload or use the resumable train_dl protocol"
        )

    restore_model_state(model, best_state)
    del best_state, optimizer, train_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    best_validation = evaluate(model, val_loader, device, criterion)
    test = evaluate(model, test_loader, device, criterion)
    threshold, _ = find_optimal_threshold(y_val, best_validation["probs"])

    val_observed_default = classification_metrics(
        y_val, best_validation["probs"], 0.5
    )
    val_latent_default = classification_metrics(
        ystar_val, best_validation["probs"], 0.5
    )
    test_observed_default = classification_metrics(y_test, test["probs"], 0.5)
    test_latent_default = classification_metrics(
        ystar_test, test["probs"], 0.5
    )
    val_observed_selected = classification_metrics(
        y_val, best_validation["probs"], threshold
    )
    val_latent_selected = classification_metrics(
        ystar_val, best_validation["probs"], threshold
    )
    test_observed_selected = classification_metrics(
        y_test, test["probs"], threshold
    )
    test_latent_selected = classification_metrics(
        ystar_test, test["probs"], threshold
    )

    validation_predictions_path = save_predictions(
        pd.DataFrame(
            {
                "observed_label": np.asarray(y_val),
                "latent_label": np.asarray(ystar_val),
                "probability_positive": best_validation["probs"],
                "predicted_at_0.5": (
                    best_validation["probs"] >= 0.5
                ).astype(int),
                "predicted_at_val_threshold": (
                    best_validation["probs"] >= threshold
                ).astype(int),
            }
        ),
        output_dir / "validation_predictions.parquet",
    )
    test_predictions_path = save_predictions(
        pd.DataFrame(
            {
                "observed_label": np.asarray(y_test),
                "latent_label": np.asarray(ystar_test),
                "probability_positive": test["probs"],
                "predicted_at_0.5": (test["probs"] >= 0.5).astype(int),
                "predicted_at_val_threshold": (
                    test["probs"] >= threshold
                ).astype(int),
            }
        ),
        output_dir / "test_predictions.parquet",
    )

    elapsed = time.time() - started
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "training_mode": training_mode,
        "task": args.task,
        "seed": args.seed,
        "train_rows": len(X_train),
        "epochs_done": len(history),
        "best_epoch": stopper.best_epoch,
        "val_f1": round(val_observed_selected["f1"], 6),
        "val_auc": round(best_validation["auc"], 6),
        "val_loss": round(best_validation["loss"], 6),
        "test_auc_obs": round(test["auc"], 6),
        "test_auc_latent": round(test_latent_default["auc"], 6),
        "test_f1": round(test_observed_selected["f1"], 6),
        "test_precision": round(test_observed_selected["precision"], 6),
        "test_recall": round(test_observed_selected["recall"], 6),
        "threshold": round(threshold, 4),
        "n_params": n_params,
        "recipe": (
            f"checkpoint-free {args.model} {recipe['config']} Adam "
            f"constant_lr={learning_rate} batch={args.batch_size} "
            f"epochs<={args.max_epochs} patience={args.patience} "
            f"early=observed_val_loss [{recipe['provenance']}]"
        ),
        "wallclock_s": round(elapsed, 1),
        "checkpoint": (
            "NOT_SAVED; best model state evaluated in memory; "
            f"run={output_dir.name}"
        ),
        "smoke": False,
        "host": os.uname().nodename,
    }
    append_result(RESULTS_DIR / "training_results.csv", row)
    atomic_json_dump(history, output_dir / "history.json")

    summary: dict[str, Any] = {
        "status": "standard_evaluation_complete",
        "model": args.model,
        "task": args.task,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "learning_rate_source": learning_rate_source,
        "scheduler": "none_constant",
        "termination_reason": termination_reason,
        "converged_by_patience": termination_reason == "early_stopping",
        "checkpoint_saved": False,
        "epochs_completed": len(history),
        "incomplete_epoch": incomplete_epoch,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best_loss,
        "selected_threshold": threshold,
        "ordinary_metrics": {
            "validation": {
                "observed_at_0.5": val_observed_default,
                "latent_at_0.5": val_latent_default,
                "observed_at_selected_threshold": val_observed_selected,
                "latent_at_selected_threshold": val_latent_selected,
                "observed_loss": best_validation["loss"],
            },
            "test": {
                "observed_at_0.5": test_observed_default,
                "latent_at_0.5": test_latent_default,
                "observed_at_selected_threshold": test_observed_selected,
                "latent_at_selected_threshold": test_latent_selected,
                "observed_loss": test["loss"],
            },
        },
        "n_params": n_params,
        "standard_metrics": row,
        "validation_predictions": str(validation_predictions_path),
        "test_predictions": str(test_predictions_path),
        "pair_evaluation": None,
    }
    atomic_json_dump(summary, summary_path)

    print("[dl-convergence] evaluating all matched pairs in memory", flush=True)
    pair_rows = evaluate_artifact(
        score_sequences,
        {
            "model": args.model,
            "seed": args.seed,
            "training_mode": training_mode,
        },
        args.task,
        False,
        ["val", "test"],
        (
            "NOT_SAVED; in-memory best model state; "
            f"run={output_dir.name}; best_epoch={stopper.best_epoch}"
        ),
        out_dir=output_dir / "pair_predictions",
    )
    summary["status"] = "complete"
    summary["pair_evaluation"] = pair_rows
    summary["total_wallclock_s"] = round(time.time() - started, 1)
    atomic_json_dump(summary, summary_path)

    print("=== DL CONVERGENCE RUN COMPLETE ===", flush=True)
    print(
        f"model={args.model} task={args.task} "
        f"termination={termination_reason} epochs={len(history)} "
        f"best_epoch={stopper.best_epoch} test_auc_obs={test['auc']:.4f} "
        f"test_auc_latent={test_latent_default['auc']:.4f}",
        flush=True,
    )
    print("checkpoint_saved=False", flush=True)
    print(f"results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
