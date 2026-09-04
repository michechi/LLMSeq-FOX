"""Checkpoint-free XGBoost convergence diagnostic for OC classification.

This runner fits one task and one seed at a time using the existing
``all_offset_pair`` sparse feature representation.  The maximum number of
boosting rounds is the XGBoost analogue of a neural model's maximum epochs;
early stopping selects the best round by observed-label validation log-loss.

The fitted booster is kept only in process memory.  No joblib file, XGBoost
model, or resumable checkpoint is written.  After restoring the best
iteration logically through ``iteration_range``, the runner evaluates the
ordinary validation/test splits and every matched one-hole/two-hole pair set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from src.oc_completion.eval_pairs import evaluate_artifact
from src.oc_completion.train_baselines import (
    all_offset_sparse_row,
    sparse_matrix,
)
from src.oc_completion.train_dl import DATA_ROOT, RESULTS_DIR, append_result


TASKS = ("ocdet", "ocnoisy")
DEFAULT_SEED = 9550
DEFAULT_MAX_ROUNDS = 2_000
DEFAULT_PATIENCE = 50
DEFAULT_LR = 0.05
DEFAULT_MAX_DEPTH = 6
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_THREADS = 16
MODEL_ID = "all_offset_pair_xgb"
TRAINING_MODE = "xgb_checkpoint_free_convergence"

_STOP_REQUESTED = False
_STOP_SIGNAL: int | None = None


def request_graceful_stop(signum: int, _frame: Any) -> None:
    """Request an in-memory finalization after the current boosting round."""
    global _STOP_REQUESTED, _STOP_SIGNAL
    _STOP_REQUESTED = True
    _STOP_SIGNAL = signum
    print(
        f"[xgb] received signal {signum}; stop after the current boosting "
        "round and finalize the best completed round",
        flush=True,
    )


class GracefulStopCallback(TrainingCallback):
    """Stop ``fit`` at an iteration boundary after a Slurm warning signal."""

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        del model, epoch, evals_log
        return _STOP_REQUESTED


def atomic_json_dump(payload: Any, path: Path) -> None:
    """Atomically replace a small JSON artifact."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def save_predictions(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Save predictions as Parquet, with a compressed CSV fallback."""
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        csv_path = parquet_path.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False)
        return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit checkpoint-free CPU XGBoost on all-offset OC pair-count "
            "features and evaluate ordinary and matched-pair splits."
        )
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--max_depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--subsample", type=float, default=DEFAULT_SUBSAMPLE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    positive_integers = {
        "max_rounds": args.max_rounds,
        "patience": args.patience,
        "max_depth": args.max_depth,
        "threads": args.threads,
    }
    for name, value in positive_integers.items():
        if value < 1:
            parser.error(f"--{name} must be positive")
    if not 0 <= args.seed <= 2**31 - 1:
        parser.error("--seed must be in [0, 2147483647]")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not math.isfinite(args.subsample) or not 0 < args.subsample <= 1:
        parser.error("--subsample must be finite and in (0, 1]")
    return args


def load_split(task: str, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load one generated OC split without changing its sequence encoding."""
    x_path = DATA_ROOT / f"X_{split}_{task}.csv"
    y_path = DATA_ROOT / f"y_{split}_{task}.csv"
    if not x_path.is_file() or not y_path.is_file():
        missing = [str(path) for path in (x_path, y_path) if not path.is_file()]
        raise FileNotFoundError(f"missing OC split input(s): {missing}")

    x_frame = pd.read_csv(x_path, usecols=["Sequences"])
    y_frame = pd.read_csv(y_path, usecols=["Outcome", "Latent"])
    if len(x_frame) != len(y_frame):
        raise ValueError(
            f"row-count mismatch for {task}/{split}: "
            f"X={len(x_frame)}, y={len(y_frame)}"
        )
    if x_frame["Sequences"].isna().any():
        raise ValueError(f"missing sequence in {x_path}")

    observed = y_frame["Outcome"].to_numpy(dtype=np.int8, copy=True)
    latent = y_frame["Latent"].to_numpy(dtype=np.int8, copy=True)
    for label_name, labels in (("Outcome", observed), ("Latent", latent)):
        values = set(np.unique(labels).tolist())
        if not values <= {0, 1} or len(values) < 2:
            raise ValueError(
                f"{task}/{split} {label_name} must contain both binary "
                f"classes; found {sorted(values)}"
            )
    return x_frame["Sequences"].astype(str).tolist(), observed, latent


def auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Binary ROC-AUC with a defensive single-class fallback."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, probabilities))


def find_optimal_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> tuple[float, float]:
    """Select the observed-validation threshold that maximizes F1."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.10, 0.90, 0.01):
        predictions = (probabilities >= threshold).astype(np.int8)
        score = float(f1_score(labels, predictions, zero_division=0))
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score
    return best_threshold, best_f1


def ordinary_metrics(
    observed: np.ndarray,
    latent: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Return parallel observed-label and latent-label split metrics."""
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12
    )
    predictions = (probabilities >= threshold).astype(np.int8)
    metrics: dict[str, float] = {}
    for label_name, labels in (("observed", observed), ("latent", latent)):
        metrics[f"logloss_{label_name}"] = float(
            log_loss(labels, probabilities, labels=[0, 1])
        )
        metrics[f"auc_{label_name}"] = auc(labels, probabilities)
        metrics[f"f1_{label_name}"] = float(
            f1_score(labels, predictions, zero_division=0)
        )
        metrics[f"precision_{label_name}"] = float(
            precision_score(labels, predictions, zero_division=0)
        )
        metrics[f"recall_{label_name}"] = float(
            recall_score(labels, predictions, zero_division=0)
        )
    return metrics


def build_features(sequences: list[str], split: str):
    """Build the canonical sparse all-offset pair-count representation."""
    started = time.time()
    matrix = sparse_matrix(sequences, all_offset_sparse_row)
    elapsed = time.time() - started
    print(
        f"[xgb] features split={split} rows={matrix.shape[0]} "
        f"cols={matrix.shape[1]} nnz={matrix.nnz} seconds={elapsed:.1f}",
        flush=True,
    )
    return matrix, elapsed


def make_pair_scorer(
    model: XGBClassifier, best_iteration: int
) -> Callable[[list[str]], np.ndarray]:
    """Return the logit scorer expected by matched-pair evaluation."""
    iteration_range = (0, best_iteration + 1)

    def score(sequences: list[str]) -> np.ndarray:
        features = sparse_matrix(sequences, all_offset_sparse_row)
        probabilities = model.predict_proba(
            features, iteration_range=iteration_range
        )[:, 1]
        probabilities = np.clip(probabilities.astype(float), 1e-12, 1 - 1e-12)
        log_odds = np.log(probabilities / (1.0 - probabilities))
        return np.stack([np.zeros(len(log_odds)), log_odds], axis=1)

    return score


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
    summary_path = output_dir / "summary.json"
    started = time.time()
    signal.signal(signal.SIGUSR1, request_graceful_stop)

    X_train, y_train, _ = load_split(args.task, "train")
    X_val, y_val, ystar_val = load_split(args.task, "val")
    X_test, y_test, ystar_test = load_split(args.task, "test")

    config = {
        "purpose": "checkpoint-free XGBoost convergence diagnostic",
        "model": MODEL_ID,
        "feature_family": "all_offset_pair",
        "feature_definition": (
            "sparse stacked ordered-letter pair counts at every offset 1..19"
        ),
        "task": args.task,
        "seed": args.seed,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cpu",
        "max_rounds": args.max_rounds,
        "patience": args.patience,
        "early_stopping_metric": "observed-label validation logloss",
        "learning_rate": args.lr,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "threads": args.threads,
        "checkpoint_policy": "none; fitted booster retained only in process memory",
        "pair_training": False,
        "python": os.sys.version,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
    }
    atomic_json_dump(config, output_dir / "config.json")

    print("=== CHECKPOINT-FREE XGBOOST CONVERGENCE RUN ===", flush=True)
    print(
        f"model={MODEL_ID} task={args.task} seed={args.seed} "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)}",
        flush=True,
    )
    print(
        f"rounds<={args.max_rounds} patience={args.patience} lr={args.lr} "
        f"depth={args.max_depth} subsample={args.subsample} "
        f"threads={args.threads} device=cpu",
        flush=True,
    )
    print("checkpoint_policy=NONE", flush=True)

    train_features, train_feature_seconds = build_features(X_train, "train")
    val_features, val_feature_seconds = build_features(X_val, "val")
    test_features, test_feature_seconds = build_features(X_test, "test")

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=args.max_rounds,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        tree_method="hist",
        device="cpu",
        n_jobs=args.threads,
        random_state=args.seed,
        eval_metric="logloss",
        early_stopping_rounds=args.patience,
        callbacks=[GracefulStopCallback()],
        verbosity=1,
    )
    training_started = time.time()
    model.fit(
        train_features,
        y_train,
        eval_set=[(train_features, y_train), (val_features, y_val)],
        verbose=10,
    )
    training_seconds = time.time() - training_started

    evaluation_history = model.evals_result()
    train_losses = evaluation_history["validation_0"]["logloss"]
    val_losses = evaluation_history["validation_1"]["logloss"]
    if len(train_losses) != len(val_losses) or not val_losses:
        raise RuntimeError("XGBoost returned an invalid evaluation history")

    # The validation-loss history is authoritative.  In particular, if the
    # wall-time callback is the first callback to return True, some XGBoost
    # releases do not let the built-in callback attach ``best_iteration``.
    reported_best_iteration = getattr(model, "best_iteration", None)
    val_loss_array = np.asarray(val_losses, dtype=float)
    if not np.isfinite(val_loss_array).all():
        raise RuntimeError("XGBoost returned a non-finite validation logloss")
    if reported_best_iteration is None:
        best_iteration = int(np.argmin(val_loss_array))
    else:
        best_iteration = int(reported_best_iteration)
    if not 0 <= best_iteration < len(val_losses):
        raise RuntimeError(
            f"invalid XGBoost best_iteration={best_iteration} for "
            f"{len(val_losses)} trained rounds"
        )
    best_round = best_iteration + 1
    best_val_logloss = float(val_losses[best_iteration])
    rounds_trained = len(val_losses)
    if _STOP_REQUESTED:
        termination_reason = (
            "slurm_signal_usr1"
            if _STOP_SIGNAL == signal.SIGUSR1
            else f"signal_{_STOP_SIGNAL}"
        )
    elif rounds_trained < args.max_rounds:
        termination_reason = "early_stopping"
    else:
        termination_reason = "max_rounds"

    history: list[dict[str, Any]] = []
    running_best = math.inf
    running_best_round = 0
    stale = 0
    for index, (train_value, val_value) in enumerate(
        zip(train_losses, val_losses), start=1
    ):
        improved = float(val_value) < running_best
        if improved:
            running_best = float(val_value)
            running_best_round = index
            stale = 0
        else:
            stale += 1
        history.append(
            {
                "round": index,
                "train_logloss_observed": float(train_value),
                "val_logloss_observed": float(val_value),
                "improved": improved,
                "best_round": running_best_round,
                "best_val_logloss": running_best,
                "no_improve": stale,
            }
        )
    atomic_json_dump(history, output_dir / "history.json")
    with open(output_dir / "history.jsonl", "w", encoding="utf-8") as handle:
        for record in history:
            handle.write(json.dumps(record) + "\n")

    iteration_range = (0, best_iteration + 1)
    val_probabilities = model.predict_proba(
        val_features, iteration_range=iteration_range
    )[:, 1]
    test_probabilities = model.predict_proba(
        test_features, iteration_range=iteration_range
    )[:, 1]
    threshold, validation_f1_optimal = find_optimal_threshold(
        y_val, val_probabilities
    )
    validation_metrics = ordinary_metrics(
        y_val, ystar_val, val_probabilities, threshold
    )
    test_metrics = ordinary_metrics(
        y_test, ystar_test, test_probabilities, threshold
    )

    validation_predictions_path = save_predictions(
        pd.DataFrame(
            {
                "observed_label": y_val,
                "latent_label": ystar_val,
                "probability_positive": val_probabilities,
                "predicted_label": (val_probabilities >= threshold).astype(np.int8),
            }
        ),
        output_dir / "validation_predictions.parquet",
    )
    test_predictions_path = save_predictions(
        pd.DataFrame(
            {
                "observed_label": y_test,
                "latent_label": ystar_test,
                "probability_positive": test_probabilities,
                "predicted_label": (test_probabilities >= threshold).astype(np.int8),
            }
        ),
        output_dir / "test_predictions.parquet",
    )

    standard_row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": MODEL_ID,
        "training_mode": TRAINING_MODE,
        "task": args.task,
        "seed": args.seed,
        "train_rows": len(X_train),
        "epochs_done": rounds_trained,
        "best_epoch": best_round,
        "val_f1": round(validation_f1_optimal, 6),
        "val_auc": round(validation_metrics["auc_observed"], 6),
        "val_loss": round(validation_metrics["logloss_observed"], 6),
        "test_auc_obs": round(test_metrics["auc_observed"], 6),
        "test_auc_latent": round(test_metrics["auc_latent"], 6),
        "test_f1": round(test_metrics["f1_observed"], 6),
        "test_precision": round(test_metrics["precision_observed"], 6),
        "test_recall": round(test_metrics["recall_observed"], 6),
        "threshold": round(threshold, 4),
        "n_params": "n/a",
        "recipe": (
            f"checkpoint-free XGBoost CPU hist all_offset_pair "
            f"rounds<={args.max_rounds} patience={args.patience} "
            f"early=val_logloss lr={args.lr} depth={args.max_depth} "
            f"subsample={args.subsample} threads={args.threads}"
        ),
        "wallclock_s": round(time.time() - started, 1),
        "checkpoint": (
            "NOT_SAVED; best iteration evaluated in memory; "
            f"run={output_dir.name}"
        ),
        "smoke": False,
        "host": os.uname().nodename,
    }
    append_result(RESULTS_DIR / "training_results.csv", standard_row)

    summary: dict[str, Any] = {
        "status": "standard_evaluation_complete",
        "model": MODEL_ID,
        "training_mode": TRAINING_MODE,
        "task": args.task,
        "seed": args.seed,
        "checkpoint_saved": False,
        "max_rounds": args.max_rounds,
        "patience": args.patience,
        "rounds_trained": rounds_trained,
        "best_iteration_zero_based": best_iteration,
        "xgboost_reported_best_iteration": reported_best_iteration,
        "best_round": best_round,
        "best_val_logloss_observed": best_val_logloss,
        "termination_reason": termination_reason,
        "converged_by_patience": termination_reason == "early_stopping",
        "termination_signal": _STOP_SIGNAL,
        "threshold_selected_on_observed_validation_f1": threshold,
        "ordinary_validation": validation_metrics,
        "ordinary_test": test_metrics,
        "standard_metrics": standard_row,
        "feature_build_seconds": {
            "train": train_feature_seconds,
            "val": val_feature_seconds,
            "test": test_feature_seconds,
        },
        "training_seconds": training_seconds,
        "validation_predictions": str(validation_predictions_path),
        "test_predictions": str(test_predictions_path),
        "pair_evaluation": None,
    }
    atomic_json_dump(summary, summary_path)

    print(
        f"[xgb] trained_rounds={rounds_trained} best_round={best_round} "
        f"best_val_logloss={best_val_logloss:.6f} "
        f"termination={termination_reason}",
        flush=True,
    )
    print("[xgb] evaluating every matched pair set in memory", flush=True)
    pair_rows = evaluate_artifact(
        make_pair_scorer(model, best_iteration),
        {
            "model": MODEL_ID,
            "seed": args.seed,
            "training_mode": TRAINING_MODE,
        },
        args.task,
        False,
        ["val", "test"],
        (
            "NOT_SAVED; in-memory XGBoost best iteration; "
            f"run={output_dir.name}; best_round={best_round}"
        ),
        out_dir=output_dir / "pair_predictions",
    )
    summary["status"] = "complete"
    summary["pair_evaluation"] = pair_rows
    summary["total_wallclock_s"] = round(time.time() - started, 1)
    atomic_json_dump(summary, summary_path)

    # Make the no-checkpoint contract explicit and release large matrices before
    # normal process shutdown.  The booster has never been serialized.
    del model, train_features, val_features, test_features
    print("=== XGBOOST CONVERGENCE RUN COMPLETE ===", flush=True)
    print(
        f"task={args.task} seed={args.seed} rounds={rounds_trained} "
        f"best_round={best_round} test_auc_obs="
        f"{test_metrics['auc_observed']:.4f} test_auc_latent="
        f"{test_metrics['auc_latent']:.4f}",
        flush=True,
    )
    print("checkpoint_saved=False", flush=True)
    print(f"results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
