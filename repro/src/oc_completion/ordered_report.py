"""Idempotent aggregation and paper outputs for the Ordered Compliance audit.

Only runs with a valid ``done.json`` whose status is ``complete`` contribute
result rows.  Missing and incomplete configurations are inventoried rather
than imputed.  Prediction shards are merged with PyArrow's streaming parquet
writer, and figures are emitted as dependency-free SVG files so reporting does
not depend on matplotlib being installed on a compute node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
from collections import defaultdict
from html import escape as xml_escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.oc_completion.ordered_baselines import ALL_FAMILIES
from src.oc_completion.ordered_io import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    MODEL_SEEDS,
    PI_LEVELS,
    RESULT_ROOT,
)


SCRATCH_MODELS = ("LSTM", "Transformer", "RNNTransformer")
HF_ENDPOINT_MODELS = ("bert_lora", "llama_lora")
HF_MODEL_SEEDS = (int(os.environ.get("OC_HF_MODEL_SEED", "9950")),)
RANDOM_CONTROLS = tuple(f"random_untrained_{name}" for name in SCRATCH_MODELS)

MODEL_LABELS = {
    "LSTM": "LSTM",
    "Transformer": "Transformer",
    "RNNTransformer": "RNNTransformer",
    "bert_lora": "BERT-base + LoRA",
    "llama_lora": "Llama-3.2-1B + LoRA",
    "oracle": "Oracle",
    "letter_count_logreg": "Letter-count LR",
    "lag_pair_logreg": "Lag-7 pair-count LR",
    "chain_occupancy_logreg": "Chain-occupancy LR",
    "chain_occupancy_xgb": "Per-chain-count XGBoost",
    "position_lag_pair_logreg": "Position-aware lag-pair LR",
    "lag_trigram_logreg": "Lag-trigram LR",
    "occupancy_max_count": "Occupancy max-count score",
    "occupancy_n_chains_ge2": "Occupancy chains>=2 score",
    "occupancy_max_run": "Occupancy max-run score",
}

PALETTE = (
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#4b5563", "#65a30d", "#9333ea",
)

MAIN_METRICS = (
    "standard_observed_auc",
    "standard_latent_auc",
    "decisive_position_auroc",
    "repair_macro_auc",
    "top1_valid_filling_accuracy",
    "fixed_position_flip_rate",
    "strict_pair_accuracy",
)

PAPER_ENDPOINT_AUC = {
    ("LSTM", 0.0): 0.999,
    ("Transformer", 0.0): 0.999,
    ("RNNTransformer", 0.0): 0.997,
    ("bert_lora", 0.0): 0.999,
    ("llama_lora", 0.0): 0.846,
    ("LSTM", 0.3): 0.665,
    ("Transformer", 0.3): 0.671,
    ("RNNTransformer", 0.3): 0.671,
    ("bert_lora", 0.3): 0.669,
    ("llama_lora", 0.3): 0.587,
}

TRAINING_COLUMNS = (
    "model", "noise_level", "model_seed", "training_mode", "train_rows",
    "best_epoch", "epochs_completed", "validation_selected_threshold",
    "best_validation_f1", "standard_observed_auc", "standard_latent_auc",
    "standard_observed_f1", "standard_observed_precision",
    "standard_observed_recall", "standard_latent_f1",
    "standard_latent_precision", "standard_latent_recall",
    "training_time_seconds", "hardware", "best_checkpoint", "last_checkpoint",
    "run_directory", "source_done", "paper_expected_observed_auc",
    "paper_auc_absolute_deviation", "paper_auc_deviation_gt_0p02",
)

HOLE_PREDICTION_COLUMNS = (
    "base_sequence_id", "base_Y_star", "position", "original_letter",
    "candidate_letter", "candidate_Y_star", "positive_logit",
    "positive_probability", "model", "noise_level", "model_seed",
    "checkpoint_epoch",
)

STRICT_PREDICTION_COLUMNS = (
    "pair_id", "base_sequence_id", "background", "split", "positive_index",
    "positive_logit", "negative_logit", "margin", "strict_win", "tie", "loss",
    "model", "noise_level", "model_seed", "checkpoint_epoch",
)


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _valid_done(path: Path) -> dict[str, Any] | None:
    payload = _read_json(path)
    if isinstance(payload, dict) and payload.get("status") == "complete":
        return payload
    return None


def _is_smoke(path: Path, payload: Mapping[str, Any] | None = None) -> bool:
    return (
        any("smoke" in part.lower() for part in path.parts)
        or bool((payload or {}).get("smoke", False))
    )


def _leaf_values(payload: Any, key: str) -> Iterable[Any]:
    if isinstance(payload, Mapping):
        if key in payload:
            yield payload[key]
        for value in payload.values():
            yield from _leaf_values(value, key)
    elif isinstance(payload, list):
        for value in payload:
            yield from _leaf_values(value, key)


def _first(payload: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        for value in _leaf_values(payload, key):
            if value is not None and not isinstance(value, (dict, list)):
                return value
    return default


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_default(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_model(value: Any) -> str:
    raw = str(value or "unknown").strip()
    compact = (raw.lower().replace("-", "_").replace(" ", "_")
               .replace("/", "_").replace(".", "_"))
    aliases = {
        "lstm": "LSTM",
        "transformer": "Transformer",
        "rnntransformer": "RNNTransformer",
        "rnn_transformer": "RNNTransformer",
        "bert": "bert_lora",
        "bert_base_lora": "bert_lora",
        "bert_base_uncased": "bert_lora",
        "bert_lora": "bert_lora",
        "llama": "llama_lora",
        "llama_3.2_1b_lora": "llama_lora",
        "llama_3_2_1b_lora": "llama_lora",
        "meta_llama_llama_3_2_1b": "llama_lora",
        "llama_lora": "llama_lora",
    }
    if compact.startswith("random_untrained_"):
        tail = compact.removeprefix("random_untrained_")
        return f"random_untrained_{aliases.get(tail, raw.split('_')[-1])}"
    return aliases.get(compact, raw)


def _model_label(model: str) -> str:
    if model.startswith("random_untrained_"):
        return f"Random {model.removeprefix('random_untrained_')}"
    return MODEL_LABELS.get(model, model)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _training_row(done_path: Path, done: Mapping[str, Any]) -> dict[str, Any]:
    config = _read_json(done_path.parent / "config.json") or {}
    combined = {"done": done, "config": config}
    model = _normalise_model(
        _first(combined, "model_name", "model", "arm", default=done_path.parent.parent.parent.name)
    )
    pi = _float_or_nan(_first(combined, "noise_pi", "noise_level", "pi"))
    seed = _int_or_default(_first(combined, "model_seed", "seed"), 0)
    hardware = _first(combined, "hardware", default="")
    if isinstance(hardware, (dict, list)):
        hardware = json.dumps(hardware, sort_keys=True)
    checkpoints = done.get("checkpoints", {}) if isinstance(done, Mapping) else {}
    expected = PAPER_ENDPOINT_AUC.get((model, round(pi, 1))) if math.isfinite(pi) else None
    observed_auc = _float_or_nan(_first(
        done, "test_observed_auc", "standard_observed_auc", "test_auc_obs",
        "observed_label_test_auc",
    ))
    deviation = (
        abs(observed_auc - expected)
        if expected is not None and math.isfinite(observed_auc) else float("nan")
    )
    return {
        "model": model,
        "noise_level": pi,
        "model_seed": seed,
        "training_mode": _first(combined, "training_mode", default=(
            "scratch" if model in SCRATCH_MODELS else "lora"
        )),
        "train_rows": _int_or_default(_first(combined, "train_rows", default=(
            (config.get("split_rows") or {}).get("train") if isinstance(config, dict) else None
        ))),
        "best_epoch": _int_or_default(_first(done, "best_epoch")),
        "epochs_completed": _int_or_default(_first(done, "epochs_completed", "epochs_done")),
        "validation_selected_threshold": _float_or_nan(_first(
            done, "validation_selected_threshold", "optimal_threshold", "threshold"
        )),
        "best_validation_f1": _float_or_nan(_first(
            done, "best_validation_f1", "val_f1", "validation_f1",
            "validation_f1_at_selected_threshold",
        )),
        "standard_observed_auc": observed_auc,
        "standard_latent_auc": _float_or_nan(_first(
            done, "test_latent_auc", "standard_latent_auc", "test_auc_latent",
            "latent_label_test_auc",
        )),
        "standard_observed_f1": _float_or_nan(_first(
            done, "test_observed_f1", "standard_observed_f1", "test_f1",
            "observed_label_f1",
        )),
        "standard_observed_precision": _float_or_nan(_first(
            done, "test_observed_precision", "standard_observed_precision", "test_precision",
            "observed_label_precision",
        )),
        "standard_observed_recall": _float_or_nan(_first(
            done, "test_observed_recall", "standard_observed_recall", "test_recall",
            "observed_label_recall",
        )),
        "standard_latent_f1": _float_or_nan(_first(done, "test_latent_f1")),
        "standard_latent_precision": _float_or_nan(_first(done, "test_latent_precision")),
        "standard_latent_recall": _float_or_nan(_first(done, "test_latent_recall")),
        "training_time_seconds": _float_or_nan(_first(
            done, "training_time_seconds", "training_time_s", "training_seconds",
            "wallclock_s",
        )),
        "hardware": hardware,
        "best_checkpoint": str(
            checkpoints.get("best_standard", done_path.parent / "best_standard.pt")
        ),
        "last_checkpoint": str(checkpoints.get("last", done_path.parent / "last.pt")),
        "run_directory": str(done_path.parent),
        "source_done": str(done_path),
        "paper_expected_observed_auc": expected,
        "paper_auc_absolute_deviation": deviation,
        "paper_auc_deviation_gt_0p02": (
            bool(deviation > 0.02) if math.isfinite(deviation) else pd.NA
        ),
        "_mtime": done_path.stat().st_mtime,
    }


def discover_training_runs(
    checkpoint_root: Path, *, include_smoke: bool = False
) -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, Any]] = []
    completed_dirs: list[Path] = []
    if not checkpoint_root.exists():
        return pd.DataFrame(columns=TRAINING_COLUMNS), completed_dirs
    for done_path in checkpoint_root.rglob("done.json"):
        if "baselines" in done_path.parts:
            continue
        done = _valid_done(done_path)
        if done is None or (not include_smoke and _is_smoke(done_path, done)):
            continue
        row = _training_row(done_path, done)
        if not math.isfinite(row["noise_level"]):
            continue
        rows.append(row)
        completed_dirs.append(done_path.parent)
    if not rows:
        return pd.DataFrame(columns=TRAINING_COLUMNS), completed_dirs
    frame = pd.DataFrame(rows).sort_values("_mtime")
    frame = frame.drop_duplicates(
        ["model", "noise_level", "model_seed"], keep="last"
    ).drop(columns="_mtime")
    return frame.reindex(columns=TRAINING_COLUMNS), completed_dirs


def discover_baseline_runs(
    checkpoint_root: Path, *, include_smoke: bool = False
) -> tuple[pd.DataFrame, list[Path]]:
    rows: list[dict[str, Any]] = []
    completed_dirs: list[Path] = []
    baseline_root = checkpoint_root / "baselines"
    if not baseline_root.exists():
        return pd.DataFrame(), completed_dirs
    for done_path in baseline_root.rglob("done.json"):
        done = _valid_done(done_path)
        if done is None or (not include_smoke and _is_smoke(done_path, done)):
            continue
        result = _read_json(done_path.parent / "result.json")
        if not isinstance(result, dict):
            continue
        model = _normalise_model(result.get("model", result.get("family")))
        rows.append({
            "baseline": model,
            "model": model,
            "noise_level": _float_or_nan(result.get("noise_pi")),
            "model_seed": _int_or_default(result.get("seed"), 0),
            "training_mode": result.get("training_mode", "baseline"),
            "train_rows": result.get("train_rows"),
            "validation_selected_threshold": result.get("validation_threshold"),
            "validation_f1": result.get("validation_f1"),
            "standard_observed_auc": result.get("standard_observed_auc"),
            "standard_latent_auc": result.get("standard_latent_auc"),
            "standard_observed_f1": result.get("standard_observed_f1"),
            "training_time_seconds": result.get("training_time_s"),
            "hardware": result.get("hardware"),
            "checkpoint": result.get("checkpoint"),
            "source_done": str(done_path),
            "_mtime": done_path.stat().st_mtime,
        })
        completed_dirs.append(done_path.parent)
    if not rows:
        return pd.DataFrame(), completed_dirs
    frame = pd.DataFrame(rows).sort_values("_mtime").drop_duplicates(
        ["model", "noise_level", "model_seed"], keep="last"
    )
    return frame.drop(columns="_mtime"), completed_dirs


def _eval_identity(done_path: Path, done: Mapping[str, Any]) -> tuple[str, float, int, int]:
    metadata = done.get("metadata", {}) if isinstance(done, Mapping) else {}
    # Older baseline artifacts accidentally serialized ``str(estimator)`` as
    # the model name.  The explicit family is the stable scientific identity
    # and remains authoritative for baseline runs.
    family = metadata.get("family")
    raw_model = (
        family
        if family in ALL_FAMILIES
        else metadata.get("model", metadata.get("model_name", "unknown"))
    )
    model = _normalise_model(raw_model)
    pi = _float_or_nan(metadata.get("noise_pi", metadata.get("noise_level")))
    if not math.isfinite(pi):
        # Artifact IDs include ``pi0p1``; use it only as a last-resort identity.
        for level in PI_LEVELS:
            if f"pi{str(level).replace('.', 'p')}" in done_path.parent.name:
                pi = level
                break
    seed = _int_or_default(metadata.get("seed", metadata.get("model_seed")), 0)
    # The direct oracle scorer has no fitted parameters and advertises seed 0;
    # align it with the single baseline-grid bookkeeping seed so its standard
    # and audit metrics form one row instead of two artificial pseudo-seeds.
    if model == "oracle" and seed == 0:
        seed = 9550
    epoch = _int_or_default(metadata.get("checkpoint_epoch", metadata.get("epoch")))
    return model, pi, seed, epoch


def discover_evaluation_runs(
    results_root: Path, *, include_smoke: bool = False
) -> list[dict[str, Any]]:
    evaluation_root = results_root / "runs" / "evaluation"
    records: list[dict[str, Any]] = []
    if not evaluation_root.exists():
        return records
    for done_path in evaluation_root.rglob("done.json"):
        done = _valid_done(done_path)
        if done is None or (not include_smoke and _is_smoke(done_path, done)):
            continue
        model, pi, seed, epoch = _eval_identity(done_path, done)
        records.append({
            "model": model, "noise_level": pi, "model_seed": seed,
            "checkpoint_epoch": epoch, "directory": done_path.parent,
            "done": done, "mtime": done_path.stat().st_mtime,
        })
    # There is one main-test result per model/noise/seed.  A newer evaluation
    # supersedes an older artifact even if an earlier schema encoded the
    # oracle or selected epoch differently; epoch trajectories live in their
    # dedicated diagnostics table and are never merged here.
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record["model"], record["noise_level"], record["model_seed"],
        )
        if key not in latest or record["mtime"] > latest[key]["mtime"]:
            latest[key] = record
    return sorted(latest.values(), key=lambda row: (
        str(row["model"]), row["noise_level"], row["model_seed"]
    ))


def _read_metric_files(
    evaluation_runs: Sequence[Mapping[str, Any]], pattern: str,
    expected_columns: Sequence[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in evaluation_runs:
        directory = Path(record["directory"])
        for path in sorted(directory.glob(pattern)):
            try:
                frame = pd.read_csv(path)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            if frame.empty:
                continue
            defaults = {
                "model": record["model"],
                "noise_level": record["noise_level"],
                "model_seed": record["model_seed"],
                "checkpoint_epoch": record["checkpoint_epoch"],
                "source_file": str(path),
            }
            for key, value in defaults.items():
                # The valid run marker is the source of truth for identity.
                # Overwrite stale identity columns from pre-schema prediction
                # files, while preserving every metric value.
                frame[key] = value
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=expected_columns)
    result = pd.concat(frames, ignore_index=True, sort=False)
    for column in ("model",):
        if column in result:
            result[column] = result[column].map(_normalise_model)
    return result


def _merge_parquet_streaming(
    sources: Sequence[Path], target: Path, empty_columns: Sequence[str],
    *,
    source_defaults: Mapping[Path, Mapping[str, Any]] | None = None,
) -> int:
    """Merge source files by row group without materializing all predictions."""

    target.parent.mkdir(parents=True, exist_ok=True)
    unique_sources = [path for path in dict.fromkeys(map(Path, sources))
                      if path.is_file() and path.resolve() != target.resolve()]
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if not unique_sources:
        pd.DataFrame(columns=list(empty_columns)).to_parquet(temporary, index=False)
        os.replace(temporary, target)
        return 0

    import pyarrow as pa
    import pyarrow.parquet as pq

    defaults_by_path = {
        Path(path).resolve(): dict(values)
        for path, values in (source_defaults or {}).items()
    }
    schemas = []
    for path in unique_sources:
        schema = pq.read_schema(path)
        defaults = defaults_by_path.get(path.resolve(), {})
        for name, value in defaults.items():
            if name not in schema.names:
                schema = schema.append(pa.field(name, pa.scalar(value).type))
        schemas.append(schema)
    try:
        unified = pa.unify_schemas(schemas)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        # Evaluator versions should agree.  If a harmless dtype drift exists,
        # pandas' common-type concatenation is the conservative fallback.
        frames = [pd.read_parquet(path) for path in unique_sources]
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        return int(len(combined))

    writer = pq.ParquetWriter(temporary, unified, compression="zstd")
    rows = 0
    try:
        for path in unique_sources:
            parquet = pq.ParquetFile(path)
            for index in range(parquet.num_row_groups):
                table = parquet.read_row_group(index)
                arrays = []
                for field in unified:
                    if field.name in table.column_names:
                        column = table[field.name]
                        if column.type != field.type:
                            column = column.cast(field.type, safe=False)
                    else:
                        default = defaults_by_path.get(path.resolve(), {}).get(
                            field.name
                        )
                        column = (
                            pa.array([default] * len(table), type=field.type)
                            if default is not None
                            else pa.nulls(len(table), type=field.type)
                        )
                    arrays.append(column)
                aligned = pa.Table.from_arrays(arrays, schema=unified)
                writer.write_table(aligned)
                rows += len(aligned)
    finally:
        writer.close()
    os.replace(temporary, target)
    return rows


def merge_prediction_artifacts(
    evaluation_runs: Sequence[Mapping[str, Any]], results_root: Path
) -> dict[str, Any]:
    hole_sources: list[Path] = []
    strict_sources: list[Path] = []
    strict_defaults: dict[Path, dict[str, Any]] = {}
    for record in evaluation_runs:
        directory = Path(record["directory"])
        hole_sources.extend(sorted((directory / "hole_prediction_shards").glob("*.parquet")))
        for path in sorted(directory.glob("strict_pair_predictions_*.parquet")):
            strict_sources.append(path)
            strict_defaults[path] = {
                "model": record["model"],
                "noise_level": float(record["noise_level"]),
                "model_seed": int(record["model_seed"]),
                "checkpoint_epoch": int(record["checkpoint_epoch"]),
            }
    hole_target = results_root / "hole_predictions.parquet"
    strict_target = results_root / "strict_pair_predictions.parquet"
    return {
        "hole_prediction_sources": len(hole_sources),
        "hole_prediction_rows": _merge_parquet_streaming(
            hole_sources, hole_target, HOLE_PREDICTION_COLUMNS
        ),
        "strict_prediction_sources": len(strict_sources),
        "strict_prediction_rows": _merge_parquet_streaming(
            strict_sources, strict_target, STRICT_PREDICTION_COLUMNS,
            source_defaults=strict_defaults,
        ),
    }


def _keys(frame: pd.DataFrame) -> list[str]:
    return [name for name in ("model", "noise_level", "model_seed") if name in frame]


def _primary_position(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*_keys(frame), "decisive_position_auroc"])
    selected = frame.copy()
    if "split" in selected:
        selected = selected[selected["split"].astype(str).eq("test")]
    if "segment" in selected:
        selected = selected[selected["segment"].astype(str).eq("all")]
    columns = _keys(selected)
    keep = {"position_auroc": "decisive_position_auroc",
            "position_auprc": "decisive_position_auprc"}
    for name in selected.columns:
        if name.startswith("position_auroc_ci_"):
            keep[name] = name.replace("position_auroc", "decisive_position_auroc")
    available = [name for name in keep if name in selected]
    if not available:
        return pd.DataFrame(columns=[*columns, "decisive_position_auroc"])
    return selected.groupby(columns, as_index=False)[available].mean().rename(columns=keep)


def _primary_repair(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*_keys(frame), "repair_macro_auc"])
    selected = frame.copy()
    if "split" in selected:
        selected = selected[selected["split"].astype(str).eq("test")]
    if "segment" in selected:
        all_targets = selected["segment"].astype(str).eq("all_targets")
        if all_targets.any():
            selected = selected[all_targets]
        else:
            selected = selected[selected["segment"].astype(str).isin(("target_0", "target_1"))]
    keys = _keys(selected)
    mapping = {
        "macro_candidate_auc": "repair_macro_auc",
        "pooled_candidate_auc": "repair_pooled_auc",
        "top1_valid_filling_accuracy": "top1_valid_filling_accuracy",
        "tie_aware_top1_accuracy": "tie_aware_top1_accuracy",
        "random_choice_reference": "random_choice_reference",
    }
    for name in selected.columns:
        if name.startswith("macro_candidate_auc_ci_"):
            mapping[name] = name.replace(
                "macro_candidate_auc", "repair_macro_auc"
            )
    available = [name for name in mapping if name in selected]
    if not available:
        return pd.DataFrame(columns=[*keys, "repair_macro_auc"])
    return selected.groupby(keys, as_index=False)[available].mean().rename(columns=mapping)


def _primary_stability(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*_keys(frame), "fixed_position_flip_rate"])
    selected = frame.copy()
    if "split" in selected:
        selected = selected[selected["split"].astype(str).eq("test")]
    if "threshold_name" in selected:
        selected = selected[selected["threshold_name"].astype(str).eq("validation_selected")]
    if "category" in selected:
        selected = selected[selected["category"].astype(str).isin(("fixed_zero", "fixed_one"))]
    keys = _keys(selected)
    rows = []
    for key_values, group in selected.groupby(keys, dropna=False):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        weights = (
            group["n_positions"].to_numpy(dtype=float)
            if "n_positions" in group else np.ones(len(group))
        )
        total = weights.sum()
        row = dict(zip(keys, key_values))
        for source, target in (
            ("prediction_flip_rate", "fixed_position_flip_rate"),
            ("mean_logit_range", "fixed_mean_logit_range"),
            ("mean_probability_range", "fixed_mean_probability_range"),
        ):
            if source in group:
                row[target] = float(np.average(group[source], weights=weights)) if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _primary_strict(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*_keys(frame), "strict_pair_accuracy"])
    selected = frame.copy()
    if "split" in selected:
        selected = selected[selected["split"].astype(str).eq("test")]
    if "background" in selected:
        selected = selected[selected["background"].astype(str).eq("heldout")]
    keys = _keys(selected)
    mapping = {
        "pair_accuracy": "strict_pair_accuracy",
        "pair_accuracy_ci_lo": "strict_pair_accuracy_ci_lo",
        "pair_accuracy_ci_hi": "strict_pair_accuracy_ci_hi",
        "strict_win_rate": "strict_win_rate",
        "tie_rate": "strict_tie_rate",
        "mean_margin": "strict_mean_margin",
        "median_margin": "strict_median_margin",
        "flattened_candidate_auc": "strict_flattened_candidate_auc",
        "tie_tolerance": "strict_tie_tolerance",
    }
    available = [name for name in mapping if name in selected]
    if not available:
        return pd.DataFrame(columns=[*keys, "strict_pair_accuracy"])
    return selected.groupby(keys, as_index=False)[available].mean().rename(columns=mapping)


def _outer_metric_summary(
    base: pd.DataFrame,
    position: pd.DataFrame,
    repair: pd.DataFrame,
    stability: pd.DataFrame,
    strict: pd.DataFrame,
) -> pd.DataFrame:
    pieces = [base.copy(), _primary_position(position), _primary_repair(repair),
              _primary_stability(stability), _primary_strict(strict)]
    result: pd.DataFrame | None = None
    for piece in pieces:
        if piece.empty and not list(piece.columns):
            continue
        keys = _keys(piece)
        if len(keys) != 3:
            continue
        result = piece if result is None else result.merge(piece, on=keys, how="outer")
    return result if result is not None else pd.DataFrame(
        columns=["model", "noise_level", "model_seed"]
    )


def _add_seed_statistics(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    groups = result.groupby(["model", "noise_level"], dropna=False)
    for metric in metrics:
        if metric not in result:
            continue
        numeric = pd.to_numeric(result[metric], errors="coerce")
        result[metric] = numeric
        result[f"{metric}_seed_mean"] = groups[metric].transform("mean")
        result[f"{metric}_seed_std"] = groups[metric].transform("std")
        result[f"{metric}_seed_n"] = groups[metric].transform("count").astype("Int64")
    return result


def build_model_and_baseline_summaries(
    training: pd.DataFrame,
    baseline: pd.DataFrame,
    position: pd.DataFrame,
    repair: pd.DataFrame,
    stability: pd.DataFrame,
    strict: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_names = set(ALL_FAMILIES) | set(RANDOM_CONTROLS)
    metric_models = set()
    for frame in (position, repair, stability, strict):
        if "model" in frame:
            metric_models.update(frame["model"].map(_normalise_model).unique())

    neural_base = training.copy()
    model_summary = _outer_metric_summary(
        neural_base, position[~position.get("model", pd.Series(dtype=str)).isin(baseline_names)]
        if not position.empty else position,
        repair[~repair.get("model", pd.Series(dtype=str)).isin(baseline_names)]
        if not repair.empty else repair,
        stability[~stability.get("model", pd.Series(dtype=str)).isin(baseline_names)]
        if not stability.empty else stability,
        strict[~strict.get("model", pd.Series(dtype=str)).isin(baseline_names)]
        if not strict.empty else strict,
    )
    model_summary = _add_seed_statistics(model_summary, MAIN_METRICS)
    if not model_summary.empty and "standard_observed_auc_seed_mean" in model_summary:
        model_summary["paper_seed_mean_auc_absolute_deviation"] = model_summary.apply(
            lambda row: (
                abs(float(row["standard_observed_auc_seed_mean"])
                    - float(row["paper_expected_observed_auc"]))
                if pd.notna(row.get("standard_observed_auc_seed_mean"))
                and pd.notna(row.get("paper_expected_observed_auc"))
                else np.nan
            ),
            axis=1,
        )
        model_summary["paper_seed_mean_auc_deviation_gt_0p02"] = (
            model_summary["paper_seed_mean_auc_absolute_deviation"] > 0.02
        ).where(model_summary["paper_seed_mean_auc_absolute_deviation"].notna(), pd.NA)

    baseline_base = baseline.rename(columns={"baseline": "baseline_name"}).copy()
    baseline_summary = _outer_metric_summary(
        baseline_base,
        position[position["model"].isin(baseline_names)] if "model" in position else position,
        repair[repair["model"].isin(baseline_names)] if "model" in repair else repair,
        stability[stability["model"].isin(baseline_names)] if "model" in stability else stability,
        strict[strict["model"].isin(baseline_names)] if "model" in strict else strict,
    )
    # Evaluation-only random controls must remain visible even without a fake
    # standard-AUC row.
    for model in sorted(metric_models & baseline_names):
        if model not in set(baseline_summary.get("model", [])):
            pass
    baseline_summary = _add_seed_statistics(baseline_summary, MAIN_METRICS)
    return model_summary, baseline_summary


def adjacent_noise_differences(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if summary.empty:
        return pd.DataFrame(columns=["model", "model_seed", "from_noise", "to_noise"])
    indexed = summary.set_index(["model", "model_seed", "noise_level"], drop=False)
    for model in sorted(summary.model.dropna().unique()):
        for seed in sorted(summary.loc[summary.model.eq(model), "model_seed"].dropna().unique()):
            for low, high in zip(PI_LEVELS[:-1], PI_LEVELS[1:]):
                try:
                    before = indexed.loc[(model, seed, low)]
                    after = indexed.loc[(model, seed, high)]
                except KeyError:
                    continue
                if isinstance(before, pd.DataFrame):
                    before = before.iloc[-1]
                if isinstance(after, pd.DataFrame):
                    after = after.iloc[-1]
                row = {"model": model, "model_seed": seed,
                       "from_noise": low, "to_noise": high}
                for metric in MAIN_METRICS:
                    if metric in summary:
                        a, b = _float_or_nan(before.get(metric)), _float_or_nan(after.get(metric))
                        row[f"delta_{metric}"] = b - a if math.isfinite(a) and math.isfinite(b) else np.nan
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for metric in [name for name in result if name.startswith("delta_")]:
        groups = result.groupby(["model", "from_noise", "to_noise"])[metric]
        result[f"{metric}_seed_mean"] = groups.transform("mean")
        result[f"{metric}_seed_std"] = groups.transform("std")
        result[f"{metric}_seed_n"] = groups.transform("count").astype("Int64")
    return result


def merge_training_histories(run_dirs: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        done = _valid_done(run_dir / "done.json")
        if done is None:
            continue
        identity = _training_row(run_dir / "done.json", done)
        common = {
            "model": identity["model"], "noise_level": identity["noise_level"],
            "model_seed": identity["model_seed"], "run_directory": str(run_dir),
        }
        history_path = run_dir / "history.csv"
        if history_path.exists():
            try:
                frame = pd.read_csv(history_path)
                for key, value in reversed(list(common.items())):
                    if key not in frame:
                        frame.insert(0, key, value)
                histories.append(frame)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                pass
        diagnostic_path = run_dir / "epoch_diagnostics.csv"
        if diagnostic_path.exists():
            try:
                frame = pd.read_csv(diagnostic_path)
                for key, value in reversed(list(common.items())):
                    if key not in frame:
                        frame.insert(0, key, value)
                diagnostics.append(frame)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                pass
    history = pd.concat(histories, ignore_index=True, sort=False) if histories else pd.DataFrame(
        columns=["model", "noise_level", "model_seed", "epoch"]
    )
    epoch = pd.concat(diagnostics, ignore_index=True, sort=False) if diagnostics else pd.DataFrame(
        columns=["model", "noise_level", "model_seed", "epoch"]
    )
    return history, epoch


def _expected_configurations() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in SCRATCH_MODELS:
        for pi in PI_LEVELS:
            for seed in MODEL_SEEDS:
                rows.append({"stage": "training", "kind": "scratch", "model": model,
                             "noise_level": pi, "model_seed": seed, "required": True})
    for model in HF_ENDPOINT_MODELS:
        for pi in (0.0, 0.3):
            for seed in HF_MODEL_SEEDS:
                rows.append({"stage": "training", "kind": "hf", "model": model,
                             "noise_level": pi, "model_seed": seed, "required": True})
    for family in ALL_FAMILIES:
        for pi in PI_LEVELS:
            rows.append({"stage": "training", "kind": "baseline", "model": family,
                         "noise_level": pi, "model_seed": 9550, "required": True})
    evaluation = [dict(row, stage="evaluation") for row in rows]
    for model in RANDOM_CONTROLS:
        for pi in PI_LEVELS:
            for seed in MODEL_SEEDS:
                evaluation.append({
                    "stage": "evaluation", "kind": "random_control",
                    "model": model, "noise_level": pi,
                    "model_seed": seed, "required": True,
                })
    return rows + evaluation


def configuration_inventory(
    checkpoint_root: Path,
    results_root: Path,
    training: pd.DataFrame,
    baseline: pd.DataFrame,
    evaluation_runs: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    completed_training = set(zip(
        training.get("model", []), training.get("noise_level", []),
        training.get("model_seed", []),
    ))
    completed_baseline = set(zip(
        baseline.get("model", []), baseline.get("noise_level", []),
        baseline.get("model_seed", []),
    ))
    completed_evaluation = {
        (row["model"], row["noise_level"], row["model_seed"])
        for row in evaluation_runs
    }
    present_evaluation: set[tuple[str, float, int]] = set(completed_evaluation)
    evaluation_root = results_root / "runs" / "evaluation"
    if evaluation_root.exists():
        for done_path in evaluation_root.rglob("done.json"):
            payload = _read_json(done_path)
            if not isinstance(payload, dict) or _is_smoke(done_path, payload):
                continue
            model, pi, seed, _epoch = _eval_identity(done_path, payload)
            present_evaluation.add((model, pi, seed))
    inventory = []
    for expected in _expected_configurations():
        key = (expected["model"], expected["noise_level"], expected["model_seed"])
        if expected["stage"] == "evaluation":
            complete = key in completed_evaluation
            path = ""
            present = key in present_evaluation
        elif expected["kind"] == "baseline":
            complete = key in completed_baseline
            tag = f"{expected['noise_level']:.1f}".replace(".", "p")
            path_obj = checkpoint_root / "baselines" / f"pi_{tag}" / expected["model"]
            path, present = str(path_obj), path_obj.exists()
        else:
            complete = key in completed_training
            tag = f"{expected['noise_level']:.1f}".replace(".", "p")
            path_obj = (checkpoint_root / expected["model"].lower()
                        / f"pi_{tag}" / f"seed_{expected['model_seed']}")
            path, present = str(path_obj), path_obj.exists()
        status = "completed" if complete else "failed" if present else "missing"
        inventory.append({**expected, "status": status, "expected_path": path})

    # Preserve unplanned/incomplete directories as explicit failed inventory.
    expected_keys = {
        (row["stage"], row["model"], row["noise_level"], row["model_seed"])
        for row in inventory
    }
    if checkpoint_root.exists():
        for config_path in checkpoint_root.rglob("config.json"):
            if _is_smoke(config_path):
                continue
            done = _valid_done(config_path.parent / "done.json")
            if done is not None:
                continue
            config = _read_json(config_path) or {}
            model = _normalise_model(_first(config, "model_name", "model", "arm"))
            pi = _float_or_nan(_first(config, "noise_pi", "noise_level", "pi"))
            seed = _int_or_default(_first(config, "model_seed", "seed"), 0)
            key = ("training", model, pi, seed)
            if key not in expected_keys:
                inventory.append({"stage": "training", "kind": "unplanned",
                                  "model": model, "noise_level": pi,
                                  "model_seed": seed, "required": False,
                                  "status": "failed", "expected_path": str(config_path.parent)})
    return pd.DataFrame(inventory).sort_values(
        ["stage", "kind", "model", "noise_level", "model_seed"]
    ).reset_index(drop=True)


def _latex_escape(value: Any) -> str:
    text = str(value)
    return (text.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def _mean_sd_cell(group: pd.DataFrame, metric: str) -> str:
    if metric not in group:
        return "--"
    values = pd.to_numeric(group[metric], errors="coerce").dropna()
    if values.empty:
        return "--"
    mean = values.mean()
    if len(values) > 1:
        return f"{mean:.3f} $\\pm$ {values.std(ddof=1):.3f}"
    return f"{mean:.3f}"


def write_latex_tables(
    model_summary: pd.DataFrame, baseline_summary: pd.DataFrame, results_root: Path
) -> None:
    header = [
        "% Auto-generated by src.oc_completion.ordered_report; missing cells are not imputed.",
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\begin{tabular}{lcccccccc}", "\\toprule",
        "Model & $\\pi$ & Obs. AUC & Latent AUC & Pos. AUROC & Repair AUC & "
        "Top-1 valid & Fixed flip & Strict acc. \\\\", "\\midrule",
    ]
    lines = list(header)
    if not model_summary.empty:
        for (model, pi), group in model_summary.groupby(["model", "noise_level"], sort=True):
            cells = [_latex_escape(_model_label(model)), f"{float(pi):.1f}"]
            cells.extend(_mean_sd_cell(group, metric) for metric in MAIN_METRICS)
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Ordered Compliance standard and evaluation-only audit results. "
              "Mean $\\pm$ SD across available model seeds; -- denotes an unfinished result.}",
              "\\label{tab:oc-hole-main}", "\\end{table*}", ""]
    _atomic_text("\n".join(lines), results_root / "table_main.tex")

    lines = [
        "% Auto-generated by src.oc_completion.ordered_report; missing cells are not imputed.",
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        "Baseline & $\\pi$ & Standard AUC & Position AUROC & Repair AUC & "
        "Fixed flip & Strict acc. \\\\", "\\midrule",
    ]
    if not baseline_summary.empty:
        for (model, pi), group in baseline_summary.groupby(["model", "noise_level"], sort=True):
            cells = [_latex_escape(_model_label(model)), f"{float(pi):.1f}"]
            cells.extend(_mean_sd_cell(group, metric) for metric in (
                "standard_observed_auc", "decisive_position_auroc",
                "repair_macro_auc", "fixed_position_flip_rate", "strict_pair_accuracy",
            ))
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Ordered Compliance shortcut and random-control baselines.}",
              "\\label{tab:oc-hole-baselines}", "\\end{table*}", ""]
    _atomic_text("\n".join(lines), results_root / "table_baselines.tex")


def _svg_placeholder(path: Path, title: str, message: str = "No completed rows available") -> None:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="420" '
        'viewBox="0 0 820 420">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="410" y="45" text-anchor="middle" font-family="sans-serif" '
        f'font-size="20">{xml_escape(title)}</text>'
        f'<text x="410" y="215" text-anchor="middle" font-family="sans-serif" '
        f'font-size="15" fill="#6b7280">{xml_escape(message)}</text></svg>'
    )
    _atomic_text(svg, path)


def _svg_line_chart(
    path: Path,
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    series: str,
    title: str,
    x_label: str,
    y_label: str,
    y_domain: tuple[float, float] = (0.0, 1.0),
    error: str | None = None,
) -> None:
    required = {x, y, series}
    if frame.empty or not required.issubset(frame.columns):
        _svg_placeholder(path, title)
        return
    data = frame[list(required | ({error} if error and error in frame else set()))].copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data[y] = pd.to_numeric(data[y], errors="coerce")
    data = data.dropna(subset=[x, y, series])
    if data.empty:
        _svg_placeholder(path, title)
        return
    width, height = 900, 500
    left, right, top, bottom = 82, 220, 55, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    xmin, xmax = float(data[x].min()), float(data[x].max())
    if xmin == xmax:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    ymin, ymax = y_domain
    sx = lambda value: left + (float(value) - xmin) / (xmax - xmin) * plot_w
    sy = lambda value: top + (ymax - float(value)) / (ymax - ymin) * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left + plot_w/2:.1f}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="19">{xml_escape(title)}</text>',
    ]
    for tick in np.linspace(ymin, ymax, 6):
        yy = sy(tick)
        parts += [
            f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" '
            'stroke="#e5e7eb" stroke-width="1"/>',
            f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#4b5563">{tick:.2f}</text>',
        ]
    xticks = sorted(data[x].unique())
    for tick in xticks:
        xx = sx(tick)
        parts += [
            f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+plot_h}" '
            'stroke="#f3f4f6" stroke-width="1"/>',
            f'<text x="{xx:.1f}" y="{top+plot_h+22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{float(tick):g}</text>',
        ]
    parts += [
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#374151"/>',
        f'<text x="{left+plot_w/2:.1f}" y="{height-16}" text-anchor="middle" font-family="sans-serif" font-size="13">{xml_escape(x_label)}</text>',
        f'<text transform="translate(20 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{xml_escape(y_label)}</text>',
    ]
    for series_index, (name, group) in enumerate(data.groupby(series, sort=True)):
        group = group.sort_values(x).drop_duplicates(x, keep="last")
        color = PALETTE[series_index % len(PALETTE)]
        points = " ".join(f"{sx(row[x]):.1f},{sy(row[y]):.1f}" for _, row in group.iterrows())
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for _, row in group.iterrows():
            xx, yy = sx(row[x]), sy(row[y])
            if error and error in row and pd.notna(row[error]):
                spread = float(row[error])
                parts.append(f'<line x1="{xx:.1f}" y1="{sy(min(ymax,row[y]+spread)):.1f}" '
                             f'x2="{xx:.1f}" y2="{sy(max(ymin,row[y]-spread)):.1f}" stroke="{color}"/>')
            parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}"/>')
        legend_y = top + 18 + series_index * 19
        parts += [
            f'<line x1="{left+plot_w+20}" y1="{legend_y}" x2="{left+plot_w+44}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{left+plot_w+50}" y="{legend_y+4}" font-family="sans-serif" font-size="11">{xml_escape(_model_label(str(name)))}</text>',
        ]
    parts.append("</svg>")
    _atomic_text("".join(parts), path)


def _summary_for_plot(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary.empty or metric not in summary:
        return pd.DataFrame()
    rows = []
    for (model, pi), group in summary.groupby(["model", "noise_level"]):
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({"model": model, "noise_level": pi, "value": values.mean(),
                     "std": values.std(ddof=1) if len(values) > 1 else np.nan})
    return pd.DataFrame(rows)


def _sensitivity_samples(evaluation_runs: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for run in evaluation_runs:
        directory = Path(run["directory"]) / "hole_prediction_shards"
        shards = sorted(directory.glob("*.parquet"))[:1]
        for shard in shards:
            try:
                frame = pd.read_parquet(
                    shard,
                    columns=["base_sequence_id", "position", "candidate_Y_star",
                             "positive_logit", "model"],
                )
            except (OSError, ValueError):
                continue
            for (_base, _position), group in frame.groupby(
                ["base_sequence_id", "position"], sort=False
            ):
                labels = group.candidate_Y_star.to_numpy(dtype=np.int8)
                category = ("fixed-zero" if labels.max() == 0 else
                            "fixed-one" if labels.min() == 1 else "decisive")
                model = _normalise_model(group.model.iloc[0])
                key = (model, category)
                if counts[key] >= 5000:
                    continue
                records.append({"model": model, "category": category,
                                "delta": float(group.positive_logit.max()
                                               - group.positive_logit.min())})
                counts[key] += 1
    return pd.DataFrame(records)


def _svg_sensitivity(path: Path, samples: pd.DataFrame) -> None:
    title = "Sensitivity distributions by oracle hole category"
    if samples.empty:
        _svg_placeholder(path, title)
        return
    categories = [name for name in ("decisive", "fixed-zero", "fixed-one")
                  if name in set(samples.category)]
    values = pd.to_numeric(samples.delta, errors="coerce").dropna()
    if values.empty:
        _svg_placeholder(path, title)
        return
    width, height = 820, 460
    left, right, top, bottom = 80, 40, 60, 70
    plot_w, plot_h = width-left-right, height-top-bottom
    ymax = max(float(values.quantile(0.99)), 1e-6)
    sy = lambda value: top + (ymax-min(max(float(value), 0), ymax))/ymax*plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="19">{title}</text>']
    for tick in np.linspace(0, ymax, 6):
        yy=sy(tick); parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+plot_w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:.3g}</text>')
    slot=plot_w/max(len(categories),1)
    for idx, category in enumerate(categories):
        data=pd.to_numeric(samples.loc[samples.category.eq(category),"delta"],errors="coerce").dropna().to_numpy()
        q0,q1,q2,q3,q4=np.quantile(data,[0.05,.25,.5,.75,.95])
        x=left+(idx+.5)*slot; color=PALETTE[idx]
        parts += [f'<line x1="{x:.1f}" y1="{sy(q0):.1f}" x2="{x:.1f}" y2="{sy(q4):.1f}" stroke="{color}" stroke-width="2"/>',
                  f'<rect x="{x-45:.1f}" y="{sy(q3):.1f}" width="90" height="{sy(q1)-sy(q3):.1f}" fill="{color}" fill-opacity="0.25" stroke="{color}"/>',
                  f'<line x1="{x-45:.1f}" y1="{sy(q2):.1f}" x2="{x+45:.1f}" y2="{sy(q2):.1f}" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{x:.1f}" y="{top+plot_h+28}" text-anchor="middle" font-family="sans-serif" font-size="13">{category}</text>',
                  f'<text x="{x:.1f}" y="{top+plot_h+45}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#6b7280">n={len(data)}</text>']
    parts.append(f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">Logit range across 26 replacements</text></svg>')
    _atomic_text("".join(parts),path)


def _example_groups(evaluation_runs: Sequence[Mapping[str, Any]]) -> dict[str, pd.DataFrame]:
    wanted: dict[str, pd.DataFrame] = {}
    for run in evaluation_runs:
        for shard in sorted((Path(run["directory"]) / "hole_prediction_shards").glob("*.parquet"))[:3]:
            try:
                frame = pd.read_parquet(shard)
            except (OSError, ValueError):
                continue
            for (_base, _position), group in frame.groupby(["base_sequence_id", "position"], sort=False):
                labels=group.candidate_Y_star.to_numpy(dtype=np.int8)
                base_y=int(group.base_Y_star.iloc[0])
                if labels.min()!=labels.max():
                    category="create compliance" if base_y==0 else "destroy compliance"
                else:
                    category="fixed-zero" if labels.max()==0 else "fixed-one"
                wanted.setdefault(category,group.copy())
                if len(wanted)==4:
                    return wanted
    return wanted


def _svg_examples(path: Path, examples: Mapping[str, pd.DataFrame]) -> None:
    title="Example 26-letter replacement scores"
    if not examples:
        _svg_placeholder(path,title)
        return
    categories=("create compliance","destroy compliance","fixed-zero","fixed-one")
    width,height=1200,700; panel_w,panel_h=560,280
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">','<rect width="100%" height="100%" fill="white"/>',f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>']
    for index,category in enumerate(categories):
        x0=45+(index%2)*600; y0=55+(index//2)*320
        parts.append(f'<text x="{x0+panel_w/2}" y="{y0+16}" text-anchor="middle" font-family="sans-serif" font-size="15">{category}</text>')
        group=examples.get(category)
        if group is None:
            parts.append(f'<text x="{x0+panel_w/2}" y="{y0+140}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#6b7280">no completed example</text>')
            continue
        group=group.sort_values("candidate_letter")
        logits=group.positive_logit.to_numpy(dtype=float); lo,hi=float(logits.min()),float(logits.max())
        if hi==lo: hi=lo+1
        target=1 if category in ("create compliance","fixed-one") else 0
        valid=group.candidate_Y_star.to_numpy(dtype=int)==target
        baseline=y0+panel_h-35; chart_top=y0+35; chart_h=baseline-chart_top
        bar_w=panel_w/26*.7
        for j,(letter,value,is_valid) in enumerate(zip(group.candidate_letter,logits,valid)):
            xx=x0+j*panel_w/26; yy=baseline-(value-lo)/(hi-lo)*chart_h
            color="#059669" if is_valid else "#9ca3af"
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w:.1f}" height="{baseline-yy:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{xx+bar_w/2:.1f}" y="{baseline+12}" text-anchor="middle" font-family="sans-serif" font-size="8">{xml_escape(str(letter))}</text>')
        parts.append(f'<line x1="{x0}" y1="{baseline}" x2="{x0+panel_w}" y2="{baseline}" stroke="#374151"/>')
        parts.append(f'<text x="{x0+panel_w-5}" y="{y0+32}" text-anchor="end" font-family="sans-serif" font-size="9" fill="#059669">green = oracle-valid for target {target}</text>')
    parts.append('</svg>'); _atomic_text("".join(parts),path)


def write_figures(
    model_summary: pd.DataFrame,
    epoch_diagnostics: pd.DataFrame,
    evaluation_runs: Sequence[Mapping[str, Any]],
    results_root: Path,
) -> None:
    figures = results_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    specs = (
        ("standard_observed_auc", "figure_01_standard_observed_auc_vs_noise.svg", "Standard observed-label AUC versus noise", "Observed-label test AUC"),
        ("standard_latent_auc", "figure_02_latent_auc_vs_noise.svg", "Latent-label AUC versus noise", "Latent-label test AUC"),
        ("decisive_position_auroc", "figure_03_position_localization_auroc_vs_noise.svg", "Position-localization AUROC versus noise", "Decisive-position AUROC"),
        ("repair_macro_auc", "figure_04_valid_filling_auc_vs_noise.svg", "Valid-filling macro AUC versus noise", "Repair macro AUC"),
        ("fixed_position_flip_rate", "figure_05_fixed_position_flip_rate_vs_noise.svg", "Fixed-position flip rate versus noise", "Prediction flip rate"),
        ("strict_pair_accuracy", "figure_06_strict_pair_accuracy_vs_noise.svg", "Strict matched-pair accuracy versus noise", "Pair accuracy"),
    )
    for metric, filename, title, ylabel in specs:
        plot = _summary_for_plot(model_summary, metric)
        _svg_line_chart(figures / filename, plot, x="noise_level", y="value",
                        series="model", title=title, x_label="Noise probability pi",
                        y_label=ylabel, error="std")

    trajectory_metric = "strict_matched_pair_accuracy"
    if trajectory_metric not in epoch_diagnostics:
        trajectory_metric = "standard_validation_f1"
    combined = epoch_diagnostics.copy()
    if not combined.empty:
        combined["series"] = combined.apply(
            lambda row: f"{row.get('model','?')} pi={row.get('noise_level','?')} s={row.get('model_seed','?')}", axis=1
        )
    _svg_line_chart(
        figures / "figure_07_epoch_trajectories.svg", combined,
        x="epoch", y=trajectory_metric, series="series",
        title=f"Epoch trajectories: {trajectory_metric.replace('_',' ')}",
        x_label="Epoch", y_label=trajectory_metric.replace("_", " "),
    )
    trajectory_dir = figures / "epoch_trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    if not epoch_diagnostics.empty:
        metric_candidates = [name for name in (
            "standard_observed_validation_auc", "standard_latent_validation_auc",
            "hole_position_localization_auroc", "hole_repair_macro_auc",
            "hole_top1_valid_filling_accuracy", "fixed_position_prediction_flip_rate",
            "strict_matched_pair_accuracy",
        ) if name in epoch_diagnostics]
        for (model, pi, seed), group in epoch_diagnostics.groupby(
            ["model", "noise_level", "model_seed"]
        ):
            melted = group.melt(id_vars=["epoch"], value_vars=metric_candidates,
                                var_name="metric", value_name="value")
            name = f"{str(model).replace('/','_')}_pi_{str(pi).replace('.','p')}_seed_{seed}.svg"
            _svg_line_chart(trajectory_dir / name, melted, x="epoch", y="value",
                            series="metric", title=f"{_model_label(model)}, pi={pi}, seed={seed}",
                            x_label="Epoch", y_label="Metric")

    samples = _sensitivity_samples(evaluation_runs)
    _svg_sensitivity(figures / "figure_08_sensitivity_distributions.svg", samples)
    examples = _example_groups(evaluation_runs)
    _svg_examples(figures / "figure_09_example_26_letter_scores.svg", examples)


def _collect_commands(roots: Sequence[Path], reporter_command: str) -> list[str]:
    commands = [reporter_command]
    key_names = {"command", "exact_command", "invocation", "commands"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name == "report_manifest.json":
                continue
            payload = _read_json(path)
            if payload is None:
                continue
            stack = [payload]
            while stack:
                value = stack.pop()
                if isinstance(value, Mapping):
                    for key, child in value.items():
                        if key.lower() in key_names:
                            if isinstance(child, str) and child.strip():
                                commands.append(child.strip())
                            elif isinstance(child, list):
                                commands.extend(str(item).strip() for item in child if str(item).strip())
                        else:
                            stack.append(child)
                elif isinstance(value, list):
                    stack.extend(value)
    return list(dict.fromkeys(commands))


def _markdown_table(frame: pd.DataFrame, columns: Sequence[tuple[str, str]], limit: int = 100) -> str:
    if frame.empty:
        return "_No completed rows._"
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, rule]
    for _, row in frame.head(limit).iterrows():
        cells = []
        for name, _label in columns:
            value = row.get(name, "")
            if pd.isna(value):
                cells.append("—")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def write_report_markdown(
    *,
    results_root: Path,
    model_summary: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    inventory: pd.DataFrame,
    adjacent: pd.DataFrame,
    commands: Sequence[str],
    merge_stats: Mapping[str, Any],
) -> None:
    counts = inventory.status.value_counts().to_dict() if not inventory.empty else {}
    completed_models = model_summary[["model", "noise_level", "model_seed"]].drop_duplicates()
    deviations = model_summary[
        model_summary.get("paper_auc_deviation_gt_0p02", pd.Series(False, index=model_summary.index)).fillna(False).astype(bool)
    ] if not model_summary.empty else model_summary
    lines = [
        "# Ordered Compliance hole audit",
        "",
        "This report is generated only from valid completed-run markers. Missing cells and configurations are left missing; no result is inferred from a checkpoint filename or partial prediction shard.",
        "",
        "## Completion inventory",
        "",
        f"- Completed: {counts.get('completed', 0)}",
        f"- Failed or incomplete: {counts.get('failed', 0)}",
        f"- Missing: {counts.get('missing', 0)}",
        f"- Merged hole prediction rows: {merge_stats.get('hole_prediction_rows', 0)}",
        f"- Merged strict-pair prediction rows: {merge_stats.get('strict_prediction_rows', 0)}",
        "",
        "The row-level inventory is in `configuration_inventory.csv`.",
        "",
        "## Main per-seed results",
        "",
        _markdown_table(model_summary, (
            ("model", "Model"), ("noise_level", "pi"), ("model_seed", "Seed"),
            ("standard_observed_auc", "Observed AUC"),
            ("standard_latent_auc", "Latent AUC"),
            ("decisive_position_auroc", "Position AUROC"),
            ("repair_macro_auc", "Repair AUC"),
            ("top1_valid_filling_accuracy", "Top-1 valid"),
            ("fixed_position_flip_rate", "Fixed flip"),
            ("strict_pair_accuracy", "Strict accuracy"),
        )),
        "",
        "Seed-mean, seed-SD, and available-seed-count columns are retained alongside every main metric in `model_summary.csv`. Evaluation confidence intervals are retained from the clustered bootstrap whenever the evaluator emitted them.",
        f"The time-constrained pretrained-model sweep intentionally uses only pi=0.0 and pi=0.3 with model seed {HF_MODEL_SEEDS[0]}; seed-SD and intermediate-noise trajectories are therefore unavailable for BERT-LoRA and Llama-LoRA.",
        "",
        "## Shortcut and random controls",
        "",
        _markdown_table(baseline_summary, (
            ("model", "Baseline"), ("noise_level", "pi"),
            ("standard_observed_auc", "Standard AUC"),
            ("decisive_position_auroc", "Position AUROC"),
            ("repair_macro_auc", "Repair AUC"),
            ("fixed_position_flip_rate", "Fixed flip"),
            ("strict_pair_accuracy", "Strict accuracy"),
        )),
        "",
        "## Paper endpoint checks",
        "",
    ]
    if deviations.empty:
        lines.append("No completed endpoint run currently deviates from the paper's standard observed-label AUC by more than 0.02, or no comparable endpoint is complete.")
    else:
        lines.append(_markdown_table(deviations, (
            ("model", "Model"), ("noise_level", "pi"), ("model_seed", "Seed"),
            ("standard_observed_auc", "Observed AUC"),
            ("paper_expected_observed_auc", "Paper AUC"),
            ("paper_auc_absolute_deviation", "Absolute deviation"),
        )))
    lines += [
        "",
        "Runs outside tolerance are flagged, not discarded.",
        "",
        "## Paired adjacent-noise differences",
        "",
        _markdown_table(adjacent, (
            ("model", "Model"), ("model_seed", "Seed"),
            ("from_noise", "From pi"), ("to_noise", "To pi"),
            ("delta_standard_observed_auc", "Delta observed AUC"),
            ("delta_decisive_position_auroc", "Delta position AUROC"),
            ("delta_repair_macro_auc", "Delta repair AUC"),
            ("delta_strict_pair_accuracy", "Delta strict accuracy"),
        )),
        "",
        "Differences appear only when the same model seed is complete at both adjacent noise levels.",
        "",
        "## Interpretation",
        "",
        "Standard AUC measures ordinary held-out prediction only. Hole localization, valid-letter ranking, fixed-position stability, and strict matched-pair accuracy answer distinct evaluation-only questions. Strong strict-pair performance shows use of information beyond the controlled count, aggregated lag-pair, occupancy, run-length, unordered-chain, and global edge-count representations; it does not prove a transferable Ordered Compliance algorithm. The mechanism parameters remain fixed.",
        "",
        "## Exact recorded commands",
        "",
    ]
    if commands:
        for command in commands:
            lines.extend(["```bash", command, "```", ""])
    else:
        lines.append("_No exact command strings were recorded by completed artifacts._")
    lines += [
        "## Generated artifacts",
        "",
        "Aggregate CSVs, merged prediction parquets, LaTeX tables, and the nine SVG figures are colocated with this report. Placeholder SVGs explicitly say when no completed rows are available.",
        "",
    ]
    _atomic_text("\n".join(lines), results_root / "report.md")


def _copy_manifests(data_root: Path, results_root: Path) -> None:
    for filename in ("dataset_manifest.json", "noise_manifest.json"):
        source, target = data_root / filename, results_root / filename
        if source.exists() and source.resolve() != target.resolve():
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)


def build_report(
    *,
    results_root: Path = RESULT_ROOT,
    checkpoint_root: Path | None = None,
    data_root: Path = DATA_ROOT,
    include_smoke: bool = False,
    reporter_command: str | None = None,
) -> dict[str, Any]:
    results_root = Path(results_root).resolve()
    checkpoint_root = Path(
        checkpoint_root
        or os.environ.get("OC_CHECKPOINT_ROOT")
        or (results_root / "checkpoints")
    ).resolve()
    data_root = Path(data_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    _copy_manifests(data_root, results_root)

    training, training_dirs = discover_training_runs(
        checkpoint_root, include_smoke=include_smoke
    )
    baseline, _baseline_dirs = discover_baseline_runs(
        checkpoint_root, include_smoke=include_smoke
    )
    evaluation_runs = discover_evaluation_runs(
        results_root, include_smoke=include_smoke
    )
    position = _read_metric_files(
        evaluation_runs, "hole_position_metrics_*.csv",
        ("model", "noise_level", "model_seed", "split", "segment", "position_auroc"),
    )
    repair = _read_metric_files(
        evaluation_runs, "hole_repair_metrics_*.csv",
        ("model", "noise_level", "model_seed", "split", "segment", "macro_candidate_auc"),
    )
    stability = _read_metric_files(
        evaluation_runs, "hole_stability_metrics_*.csv",
        ("model", "noise_level", "model_seed", "split", "category", "threshold_name", "prediction_flip_rate"),
    )
    strict = _read_metric_files(
        evaluation_runs, "strict_pair_metrics_*.csv",
        ("model", "noise_level", "model_seed", "split", "background", "pair_accuracy"),
    )
    history, epoch = merge_training_histories(training_dirs)
    model_summary, baseline_summary = build_model_and_baseline_summaries(
        training, baseline, position, repair, stability, strict
    )
    difference_summary = pd.concat(
        [frame for frame in (model_summary, baseline_summary) if not frame.empty],
        ignore_index=True,
        sort=False,
    ) if (not model_summary.empty or not baseline_summary.empty) else pd.DataFrame()
    adjacent = adjacent_noise_differences(difference_summary)
    inventory = configuration_inventory(
        checkpoint_root, results_root, training, baseline, evaluation_runs
    )
    merge_stats = merge_prediction_artifacts(evaluation_runs, results_root)

    outputs = {
        "training_results.csv": training,
        "training_history.csv": history,
        "epoch_diagnostics.csv": epoch,
        "hole_position_metrics.csv": position,
        "hole_repair_metrics.csv": repair,
        "hole_stability_metrics.csv": stability,
        "strict_pair_metrics.csv": strict,
        "baseline_results.csv": baseline_summary,
        "model_summary.csv": model_summary,
        "adjacent_noise_differences.csv": adjacent,
        "configuration_inventory.csv": inventory,
    }
    for filename, frame in outputs.items():
        _atomic_csv(frame, results_root / filename)

    write_latex_tables(model_summary, baseline_summary, results_root)
    # Baselines are first-class completed audit configurations.  Include them
    # in the six noise-sweep figures so a baseline-only partial run produces
    # truthful plots instead of empty placeholders; neural rows are added as
    # soon as their valid selected-checkpoint evaluations exist.
    figure_summary = pd.concat(
        [frame for frame in (model_summary, baseline_summary) if not frame.empty],
        ignore_index=True,
        sort=False,
    ) if (not model_summary.empty or not baseline_summary.empty) else pd.DataFrame()
    write_figures(figure_summary, epoch, evaluation_runs, results_root)
    command = reporter_command or shlex.join(sys.argv)
    commands = _collect_commands((data_root, checkpoint_root, results_root), command)
    write_report_markdown(
        results_root=results_root, model_summary=model_summary,
        baseline_summary=baseline_summary, inventory=inventory,
        adjacent=adjacent, commands=commands, merge_stats=merge_stats,
    )
    manifest = {
        "status": "complete",
        "reporter_command": command,
        "completed_training_rows": int(len(training)),
        "completed_baseline_rows": int(len(baseline_summary)),
        "completed_evaluation_runs": int(len(evaluation_runs)),
        "model_summary_rows": int(len(model_summary)),
        "inventory_counts": inventory.status.value_counts().to_dict(),
        **merge_stats,
        "outputs": sorted([*outputs, "hole_predictions.parquet",
                           "strict_pair_predictions.parquet", "report.md",
                           "table_main.tex", "table_baselines.tex"]),
    }
    temporary = results_root / f".report_manifest.json.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, results_root / "report_manifest.json")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--include-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    command = shlex.join([sys.executable, "-m", "src.oc_completion.ordered_report", *(argv or sys.argv[1:])])
    manifest = build_report(
        results_root=args.results_root,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        include_smoke=args.include_smoke,
        reporter_command=command,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
