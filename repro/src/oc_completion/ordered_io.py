"""Shared paths, identifiers, and durable I/O for the OC hole audit."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

PI_LEVELS = (0.0, 0.1, 0.2, 0.3)
MODEL_SEEDS = (9550, 9551, 9552)
DATA_SEED = 9550
NOISE_SEED = 9650

REPO_ROOT = Path(os.environ.get("LLMSEQ_ROOT", Path(__file__).resolve().parents[3]))
DATA_ROOT = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")) / "simulation" / "oc_hole_audit"
RESULT_ROOT = REPO_ROOT / "results" / "oc_hole_audit"
CHECKPOINT_ROOT = Path(
    os.environ.get("OC_CHECKPOINT_ROOT", RESULT_ROOT / "checkpoints")
)


def pi_slug(pi: float) -> str:
    value = round(float(pi), 1)
    if value not in PI_LEVELS:
        raise ValueError(f"pi must be one of {PI_LEVELS}, got {pi}")
    return f"{value:.1f}".replace(".", "p")


def observed_label_column(pi: float) -> str:
    return f"Y_pi_{pi_slug(pi)}"


def load_split(data_root: Path, split: str, pi: float) -> pd.DataFrame:
    base = pd.read_parquet(data_root / "splits" / f"{split}.parquet")
    noise = pd.read_parquet(data_root / "noise" / f"{split}.parquet")
    label = observed_label_column(pi)
    required_base = {"sequence_id", "X", "Y_star"}
    if not required_base.issubset(base.columns):
        raise ValueError(f"{split} base split missing {required_base - set(base.columns)}")
    if label not in noise or "sequence_id" not in noise:
        raise ValueError(f"{split} noise split missing {label} or sequence_id")
    merged = base.merge(noise[["sequence_id", label]], on="sequence_id",
                        how="inner", validate="one_to_one")
    if len(merged) != len(base):
        raise AssertionError("base/noise sequence IDs do not match")
    return merged.rename(columns={label: "Y_observed"})


def run_dir(checkpoint_root: Path, model: str, pi: float, seed: int,
            smoke: bool = False) -> Path:
    suffix = "_smoke" if smoke else ""
    return checkpoint_root / model.lower() / f"pi_{pi_slug(pi)}" / f"seed_{seed}{suffix}"


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, default=str, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def valid_done(path: Path, expected: dict[str, Any] | None = None) -> bool:
    """A completion marker is valid JSON and matches identifying fields."""
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if payload.get("status") != "complete":
        return False
    return not expected or all(payload.get(k) == v for k, v in expected.items())
