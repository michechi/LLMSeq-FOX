"""Train the paper scratch models for the Ordered Compliance hole audit.

This runner is intentionally isolated from :mod:`train_dl`.  It consumes only
the ordinary complete-sequence splits produced by ``ordered_data`` and never
loads a hole manifest, replacement candidate, matched pair, or oracle
annotation.  Checkpoint selection is based exclusively on observed-label
validation F1.

The run layout beneath ``OC_CHECKPOINT_ROOT`` (or ``--output-root``) is::

    <model>/pi_<tag>/seed_<seed>/
        best_standard.pt
        last.pt
        config.json
        history.csv
        done.json
        epochs/epoch_001_eval.pt

Run from ``repro/`` (or set ``PYTHONPATH`` to it)::

    python -m src.oc_completion.ordered_train_dl \
        --model LSTM --noise-pi 0.1 --seed 9550 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

from src.experiments.DL_TR_baselines_experiment import create_model
from src.oc_completion.oracle import ALPHABET, N_EVENTS, SEP


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = (
    Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
    / "simulation"
    / "oc_hole_audit"
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "OC_CHECKPOINT_ROOT",
        REPO_ROOT / "results" / "oc_hole_audit" / "checkpoints",
    )
)

NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3)
MODEL_NAMES = ("LSTM", "Transformer", "RNNTransformer")
BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 30
DEFAULT_PATIENCE = 5
CHECKPOINT_FORMAT_VERSION = 1

PAD_ID = 0
TOKEN_TO_ID = {letter: index for index, letter in enumerate(ALPHABET, start=1)}
ENCODING = {
    "name": "ordered_compliance_a1_z26_pad0",
    "alphabet_ids": TOKEN_TO_ID,
    "padding_id": PAD_ID,
    "sequence_length": N_EVENTS,
    "vocab_size": len(ALPHABET) + 1,
}

# The paper anchor recipes requested for this audit.  The hybrid implementation
# in DL_TR_baselines_experiment is a one-layer bidirectional LSTM followed by a
# Transformer encoder; its recurrent depth is fixed at one in that canonical
# architecture.  The paper table does not vary the attention-head count, so the
# canonical four-head setting is retained and recorded explicitly.
PAPER_RECIPES: dict[str, dict[str, Any]] = {
    "LSTM": {
        "model_config": {
            "vocab_size": 27,
            "embedding_dim": 32,
            "hidden_dim": 128,
            "num_layers": 2,
            "num_classes": 2,
            "dropout": 0.2,
        },
        "learning_rate": 2e-3,
        "paper_spec": {
            "embedding_dimension": 32,
            "hidden_dimension": 128,
            "layers": 2,
            "dropout": 0.2,
        },
    },
    "Transformer": {
        "model_config": {
            "vocab_size": 27,
            "embedding_dim": 64,
            "num_heads": 4,
            "num_layers": 3,
            "dim_feedforward": 128,
            "num_classes": 2,
            "dropout": 0.3,
            "max_seq_length": N_EVENTS,
        },
        "learning_rate": 5e-4,
        "paper_spec": {
            "embedding_dimension": 64,
            "feed_forward_dimension": 128,
            "layers": 3,
            "dropout": 0.3,
        },
    },
    "RNNTransformer": {
        "model_config": {
            "vocab_size": 27,
            "embedding_dim": 128,
            "rnn_hidden_dim": 32,
            "num_heads": 4,
            "num_transformer_layers": 3,
            "dim_feedforward": 256,
            "num_classes": 2,
            "dropout": 0.3,
            "max_seq_length": N_EVENTS,
        },
        "learning_rate": 1e-4,
        "paper_spec": {
            "embedding_dimension": 128,
            "recurrent_hidden_dimension": 32,
            "recurrent_layers": 1,
            "transformer_feed_forward_dimension": 256,
            "transformer_layers": 3,
            "dropout": 0.3,
        },
    },
}

HISTORY_COLUMNS = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_observed_auc",
    "validation_latent_auc",
    "validation_observed_f1",
    "validation_observed_precision",
    "validation_observed_recall",
    "validation_selected_threshold",
    "best_validation_f1",
    "best_epoch",
    "patience_counter",
    "learning_rate",
    "elapsed_seconds",
)


def normalize_pi(pi: float) -> float:
    """Return one of the four registered noise levels or fail loudly."""

    value = float(pi)
    for allowed in NOISE_LEVELS:
        if abs(value - allowed) < 1e-12:
            return allowed
    raise ValueError(f"noise pi must be one of {NOISE_LEVELS}, got {pi!r}")


def pi_tag(pi: float) -> str:
    """Use the same stable noise tag as ``ordered_data``."""

    return f"{normalize_pi(pi):.1f}".replace(".", "p")


def run_directory(output_root: Path, model_name: str, pi: float, seed: int) -> Path:
    return (
        Path(output_root)
        / model_name.lower()
        / f"pi_{pi_tag(pi)}"
        / f"seed_{int(seed)}"
    )


def set_model_seed(seed: int) -> None:
    """Seed model initialization, dropout, batching, and optimization only."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _sequence_tokens(sequence: str) -> tuple[str, ...]:
    if not isinstance(sequence, str):
        raise TypeError(f"X must be a string, got {type(sequence).__name__}")
    if SEP in sequence:
        tokens = tuple(sequence.split(SEP))
    else:
        # This compatibility path accepts compact legacy storage, but still
        # produces the same twenty complete letter events.  No hole token is
        # recognized or introduced.
        tokens = tuple(sequence)
    if len(tokens) != N_EVENTS:
        raise ValueError(
            f"every training sequence must have {N_EVENTS} complete events; "
            f"found {len(tokens)} in {sequence!r}"
        )
    bad = [(index, token) for index, token in enumerate(tokens)
           if token not in TOKEN_TO_ID]
    if bad:
        raise ValueError(f"invalid complete-sequence events: {bad[:3]!r}")
    return tokens


def encode_complete_sequence(sequence: str) -> torch.Tensor:
    encoded = torch.tensor(
        [TOKEN_TO_ID[token] for token in _sequence_tokens(sequence)],
        dtype=torch.long,
    )
    if encoded.shape != (N_EVENTS,) or encoded.eq(PAD_ID).any():
        raise AssertionError("complete OC encoding violated its fixed-width contract")
    return encoded


class CompleteSequenceDataset(Dataset):
    """Only complete sequences and their ordinary binary labels."""

    def __init__(self, frame: pd.DataFrame):
        required = {"sequence_id", "X", "Y_star", "Y_observed"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"ordered split is missing columns: {sorted(missing)}")
        if frame["sequence_id"].duplicated().any():
            raise ValueError("sequence_id must be unique within each split")
        for label_column in ("Y_star", "Y_observed"):
            values = set(frame[label_column].astype(int).unique().tolist())
            if not values.issubset({0, 1}):
                raise ValueError(f"{label_column} is not binary: {sorted(values)}")

        self.sequence_ids = frame["sequence_id"].astype(str).tolist()
        self.inputs = torch.stack(
            [encode_complete_sequence(value) for value in frame["X"].tolist()]
        )
        self.observed = torch.as_tensor(
            frame["Y_observed"].to_numpy(dtype=np.int64), dtype=torch.long
        )
        self.latent = torch.as_tensor(
            frame["Y_star"].to_numpy(dtype=np.int64), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.observed)

    def __getitem__(self, index: int):
        return self.inputs[index], self.observed[index], self.latent[index]


def _limit_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or int(limit) == 0:
        return frame.reset_index(drop=True)
    if int(limit) < 0:
        raise ValueError("split row limits cannot be negative")
    return frame.iloc[: int(limit)].reset_index(drop=True)


def load_standard_split(
    data_root: Path,
    split: str,
    pi: float,
    *,
    row_limit: int | None = None,
) -> pd.DataFrame:
    """Load one merged complete-sequence/noise split from ``ordered_data``."""

    # Import lazily so architecture/checkpoint unit tests do not require the
    # data generator to have run.
    from src.oc_completion.ordered_data import load_split

    frame = load_split(Path(data_root), split, normalize_pi(pi))
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame = _limit_frame(frame, row_limit)
    required = {"sequence_id", "X", "Y_star", "Y_observed"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"ordered_data.load_split({split!r}) omitted {sorted(missing)}"
        )
    return frame


def split_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint the exact IDs, sequences, and labels used by this run."""

    columns = ["sequence_id", "X", "Y_star", "Y_observed"]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update(hashed.tobytes())
    digest.update(str(len(frame)).encode("ascii"))
    return digest.hexdigest()


def make_loader(
    dataset: CompleteSequenceDataset,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers),
    )


def build_scratch_model(model_name: str) -> nn.Module:
    if model_name not in PAPER_RECIPES:
        raise ValueError(f"unsupported scratch model {model_name!r}")
    return create_model(
        model_name,
        deepcopy(PAPER_RECIPES[model_name]["model_config"]),
    )


class ScratchCheckpointScorer:
    """Batch complete OC sequences into FP32 two-class checkpoint logits."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        batch_size: int = 1024,
    ) -> None:
        if batch_size < 1:
            raise ValueError("scorer batch_size must be positive")
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def __call__(self, sequences: Sequence[str]) -> np.ndarray:
        if len(sequences) == 0:
            return np.empty((0, 2), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(sequences), self.batch_size):
            inputs = torch.stack(
                [
                    encode_complete_sequence(sequence)
                    for sequence in sequences[start : start + self.batch_size]
                ]
            ).to(self.device)
            logits = self.model(inputs).float().cpu().numpy()
            batches.append(np.asarray(logits, dtype=np.float32))
        result = np.concatenate(batches, axis=0)
        if result.shape != (len(sequences), 2):
            raise AssertionError(f"unexpected scratch logit shape {result.shape}")
        return result


def load_scratch_scorer(
    checkpoint: Path | str,
    device: torch.device | str = "auto",
    batch_size: int = 1024,
) -> tuple[ScratchCheckpointScorer, dict[str, Any]]:
    """Load ``best_standard.pt`` or an epoch snapshot for evaluation.

    The returned callable accepts either compact strings (``ABCDEFGHIJKLMNOP``)
    or strings joined with the canonical unit separator and returns an
    ``(N, 2)`` NumPy array of FP32 logits.  It always disables dropout.
    """

    checkpoint_path = Path(checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = payload["model_name"]
    model_config = deepcopy(payload["model_config"])
    model = create_model(model_name, model_config)
    model.load_state_dict(payload["model_state_dict"])
    resolved_device = _resolve_device(str(device))
    scorer = ScratchCheckpointScorer(
        model, resolved_device, batch_size=batch_size
    )
    run_metadata = payload.get("run_metadata", {})
    snapshot_validation = payload.get("standard_validation", {})
    metadata = {
        "model": model_name,
        "model_name": model_name,
        "model_seed": run_metadata.get("model_seed", payload.get("model_seed")),
        "noise_pi": run_metadata.get("noise_pi", payload.get("noise_pi")),
        "checkpoint_epoch": payload.get("current_epoch", payload.get("epoch")),
        "best_epoch": payload.get("best_epoch"),
        "validation_selected_threshold": payload.get(
            "best_validation_threshold",
            snapshot_validation.get("selected_threshold"),
        ),
        "checkpoint_role": payload.get("checkpoint_role"),
        "checkpoint_path": str(checkpoint_path),
        "encoding": run_metadata.get("encoding", payload.get("encoding", ENCODING)),
        "device": str(resolved_device),
        "batch_size": int(batch_size),
        "logit_dtype": "float32",
        "positive_class_index": 1,
    }
    return scorer, metadata


def make_random_scorer(
    model_name: str,
    seed: int,
    device: torch.device | str = "auto",
    batch_size: int = 1024,
) -> tuple[ScratchCheckpointScorer, dict[str, Any]]:
    """Construct the paper architecture without training for a control arm."""

    if model_name not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    # Do not perturb an evaluator's surrounding Torch RNG stream merely by
    # constructing its control.  Architecture initialization itself is seeded
    # deterministically and the scorer always runs with dropout disabled.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model = build_scratch_model(model_name)
    finally:
        torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)

    resolved_device = _resolve_device(str(device))
    scorer = ScratchCheckpointScorer(
        model, resolved_device, batch_size=batch_size
    )
    metadata = {
        "model": f"random_untrained_{model_name}",
        "model_name": model_name,
        "model_seed": int(seed),
        "noise_pi": None,
        "checkpoint_epoch": 0,
        "best_epoch": None,
        "validation_selected_threshold": 0.5,
        "checkpoint_role": "random_untrained_control",
        "checkpoint_path": None,
        "encoding": deepcopy(ENCODING),
        "device": str(resolved_device),
        "batch_size": int(batch_size),
        "logit_dtype": "float32",
        "positive_class_index": 1,
    }
    return scorer, metadata


def _safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, probabilities))


def validation_f1_threshold(
    labels: Sequence[int], probabilities: Sequence[float]
) -> tuple[float, float]:
    """Choose an observed-validation threshold maximizing standard F1.

    The precision-recall construction is O(n log n), unlike an exhaustive
    threshold-by-example loop.  Deterministic ties prefer the threshold closest
    to 0.5, then the smaller threshold.
    """

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.size == 0:
        raise ValueError("cannot select a threshold on an empty validation set")
    if np.unique(y).size < 2:
        predictions = (p >= 0.5).astype(np.int64)
        return 0.5, float(f1_score(y, predictions, zero_division=0))

    precision, recall, thresholds = precision_recall_curve(y, p)
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_score = float(scores.max())
    candidates = np.flatnonzero(np.isclose(scores, best_score, rtol=0, atol=1e-12))
    selected_index = min(
        candidates.tolist(),
        key=lambda index: (abs(float(thresholds[index]) - 0.5),
                           float(thresholds[index])),
    )
    return float(thresholds[selected_index]), best_score


def threshold_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    predictions = (p >= float(threshold)).astype(np.int64)
    return {
        "auc": _safe_auc(y, p),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
    }


@torch.inference_mode()
def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    observed: list[np.ndarray] = []
    latent: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    positive_logits: list[np.ndarray] = []
    seen = 0
    for inputs, observed_labels, latent_labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        observed_device = observed_labels.to(device, non_blocking=True)
        logits = model(inputs)
        batch_size = int(inputs.shape[0])
        losses.append(float(criterion(logits, observed_device).item()) * batch_size)
        seen += batch_size
        logits_fp32 = logits.float()
        probabilities.append(
            torch.softmax(logits_fp32, dim=-1)[:, 1].cpu().numpy()
        )
        positive_logits.append(logits_fp32[:, 1].cpu().numpy())
        observed.append(observed_labels.numpy())
        latent.append(latent_labels.numpy())

    if seen == 0:
        raise ValueError("cannot evaluate an empty split")
    observed_array = np.concatenate(observed)
    latent_array = np.concatenate(latent)
    probability_array = np.concatenate(probabilities)
    return {
        "loss": float(sum(losses) / seen),
        "observed_labels": observed_array,
        "latent_labels": latent_array,
        "probabilities": probability_array,
        "positive_logits": np.concatenate(positive_logits),
        "observed_auc": _safe_auc(observed_array, probability_array),
        "latent_auc": _safe_auc(latent_array, probability_array),
    }


def hardware_metadata(device: torch.device) -> dict[str, Any]:
    gpu_names: list[str] = []
    if torch.cuda.is_available():
        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu_names": gpu_names,
        "cpu_count": os.cpu_count(),
    }


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "loader_generator": loader_generator.get_state(),
    }


def restore_rng_state(
    state: Mapping[str, Any], loader_generator: torch.Generator
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint CUDA RNG device count does not match this host"
            )
        torch.cuda.set_rng_state_all(cuda_states)
    loader_generator.set_state(state["loader_generator"])


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def atomic_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_history_dump(history: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame = pd.DataFrame(list(history), columns=HISTORY_COLUMNS)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _load_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def checkpoint_payload(
    *,
    role: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    run_metadata: Mapping[str, Any],
    current_epoch: int,
    best_validation_f1: float,
    best_epoch: int,
    best_validation_threshold: float,
    patience_counter: int,
    history: list[dict[str, Any]],
    training_time_seconds: float,
) -> dict[str, Any]:
    """Build a fully resumable epoch-boundary checkpoint."""

    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_role": role,
        "model_name": run_metadata["model_name"],
        "model_config": deepcopy(run_metadata["model_config"]),
        "model_state_dict": model.state_dict(),
        "optimizer_name": "AdamW",
        "optimizer_state_dict": optimizer.state_dict(),
        # The paper scratch recipe has no scheduler or mixed-precision scaler;
        # explicit null fields make the resume contract unambiguous.
        "scheduler_state_dict": None,
        "gradient_scaler_state_dict": None,
        "current_epoch": int(current_epoch),
        "best_validation_metric_name": "observed_validation_f1",
        "best_validation_f1": float(best_validation_f1),
        "best_epoch": int(best_epoch),
        "best_validation_threshold": float(best_validation_threshold),
        "patience_counter": int(patience_counter),
        "history": deepcopy(history),
        "training_time_seconds": float(training_time_seconds),
        "rng_state": capture_rng_state(loader_generator),
        "run_metadata": deepcopy(dict(run_metadata)),
    }


def epoch_snapshot_payload(
    *,
    model: nn.Module,
    run_metadata: Mapping[str, Any],
    epoch: int,
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the lightweight, evaluation-only state for one scratch epoch."""

    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_role": "epoch_evaluation_snapshot",
        "model_name": run_metadata["model_name"],
        "model_config": deepcopy(run_metadata["model_config"]),
        "model_state_dict": model.state_dict(),
        "encoding": deepcopy(ENCODING),
        "noise_pi": run_metadata["noise_pi"],
        "model_seed": run_metadata["model_seed"],
        "epoch": int(epoch),
        "standard_validation": deepcopy(dict(validation_metrics)),
    }


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def _validate_resume_metadata(
    checkpoint: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    if checkpoint.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported or unversioned Ordered Compliance checkpoint")
    stored = checkpoint.get("run_metadata", {})
    for key in (
        "model_name",
        "model_config",
        "noise_pi",
        "model_seed",
        "data_fingerprints",
        "split_rows",
        "batch_size",
        "optimizer",
        "learning_rate",
    ):
        if stored.get(key) != expected.get(key):
            raise ValueError(
                f"resume metadata mismatch for {key}: "
                f"checkpoint={stored.get(key)!r}, expected={expected.get(key)!r}"
            )


def restore_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    device: torch.device,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_resume_metadata(checkpoint, expected_metadata)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _optimizer_to_device(optimizer, device)
    restore_rng_state(checkpoint["rng_state"], loader_generator)
    return checkpoint


@torch.inference_mode()
def verify_checkpoint_logits(
    checkpoint_path: Path,
    encoded_inputs: torch.Tensor,
    reference_logits: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> float:
    """Reload a scratch checkpoint and assert that it reproduces logits."""

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_name = checkpoint["model_name"]
    model = create_model(model_name, deepcopy(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    resolved_device = torch.device(device)
    model = model.to(resolved_device).eval()
    reloaded_logits = model(encoded_inputs.to(resolved_device)).float().cpu()
    expected = reference_logits.detach().float().cpu()
    torch.testing.assert_close(
        reloaded_logits, expected, atol=float(atol), rtol=float(rtol)
    )
    if expected.numel() == 0:
        return 0.0
    return float((reloaded_logits - expected).abs().max().item())


def _valid_done_marker(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        payload = _load_optional_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        return False
    for key in ("model_name", "noise_pi", "model_seed"):
        if payload.get(key) != expected.get(key):
            return False
    run_dir = path.parent
    return all(
        (run_dir / filename).is_file()
        for filename in (
            "best_standard.pt",
            "last.pt",
            "config.json",
            "history.csv",
        )
    )


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "latent_prevalence": float(frame["Y_star"].mean()),
        "observed_prevalence": float(frame["Y_observed"].mean()),
    }
    if "flipped" in frame.columns:
        summary["flipped_labels"] = int(frame["flipped"].astype(bool).sum())
    return summary


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def train(args: argparse.Namespace) -> dict[str, Any]:
    pi = normalize_pi(args.noise_pi)
    if args.model not in MODEL_NAMES:
        raise ValueError(f"model must be one of {MODEL_NAMES}")
    if args.max_epochs < 1:
        raise ValueError("max_epochs must be at least one")
    if args.patience < 1:
        raise ValueError("patience must be at least one")

    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = run_directory(output_root, args.model, pi, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "epochs").mkdir(parents=True, exist_ok=True)

    identity = {
        "model_name": args.model,
        "noise_pi": pi,
        "model_seed": int(args.seed),
    }
    done_path = run_dir / "done.json"
    if _valid_done_marker(done_path, identity):
        print(f"[ordered_train_dl] complete run exists; skipping {run_dir}")
        return _load_optional_json(done_path)

    artifact_names = ("best_standard.pt", "last.pt", "history.csv")
    if not args.resume and any((run_dir / name).exists() for name in artifact_names):
        raise FileExistsError(
            f"incomplete artifacts already exist in {run_dir}; pass --resume "
            "to continue without overwriting them"
        )

    torch.set_num_threads(args.threads)
    device = _resolve_device(args.device)

    frames = {
        "train": load_standard_split(
            data_root, "train", pi, row_limit=args.max_train_rows
        ),
        "val": load_standard_split(
            data_root, "val", pi, row_limit=args.max_val_rows
        ),
        "test": load_standard_split(
            data_root, "test", pi, row_limit=args.max_test_rows
        ),
    }
    datasets = {name: CompleteSequenceDataset(frame)
                for name, frame in frames.items()}

    # Seeding occurs after deterministic data loading/subsetting, so the model
    # seed cannot alter X, Y_star, observed noise masks, or split membership.
    set_model_seed(args.seed)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(args.seed)
    pin_memory = device.type == "cuda"
    loaders = {
        "train": make_loader(
            datasets["train"], shuffle=True, generator=loader_generator,
            num_workers=args.num_workers, pin_memory=pin_memory,
        ),
        "val": make_loader(
            datasets["val"], shuffle=False, num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "test": make_loader(
            datasets["test"], shuffle=False, num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
    }

    recipe = deepcopy(PAPER_RECIPES[args.model])
    model = build_scratch_model(args.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["learning_rate"])
    )
    criterion = nn.CrossEntropyLoss()
    hardware = hardware_metadata(device)
    run_metadata = {
        **identity,
        "model_config": recipe["model_config"],
        "paper_spec": recipe["paper_spec"],
        "encoding": ENCODING,
        "optimizer": "AdamW",
        "learning_rate": float(recipe["learning_rate"]),
        "batch_size": BATCH_SIZE,
        "maximum_epochs": int(args.max_epochs),
        "early_stopping_patience": int(args.patience),
        "early_stopping_metric": "observed_validation_f1",
        "threshold_policy": "maximize observed validation F1",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "split_rows": {name: int(len(frame)) for name, frame in frames.items()},
        "data_fingerprints": {
            name: split_fingerprint(frame) for name, frame in frames.items()
        },
        "split_summary": {
            name: _split_summary(frame) for name, frame in frames.items()
        },
        "dataset_manifest": _load_optional_json(data_root / "dataset_manifest.json"),
        "noise_manifest": _load_optional_json(data_root / "noise_manifest.json"),
        "hardware": hardware,
        "training_inputs": (
            "complete length-20 sequences and observed binary labels only; "
            "no holes, replacements, pairs, oracle explanations, or structural annotations"
        ),
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
    }

    config_path = run_dir / "config.json"
    if not config_path.exists():
        atomic_json_dump(run_metadata, config_path)

    current_epoch = 0
    best_validation_f1 = -1.0
    best_epoch = -1
    best_validation_threshold = 0.5
    patience_counter = 0
    history: list[dict[str, Any]] = []
    prior_training_seconds = 0.0

    last_path = run_dir / "last.pt"
    best_path = run_dir / "best_standard.pt"
    if args.resume:
        resume_path = Path(args.resume_checkpoint).resolve() \
            if args.resume_checkpoint else last_path
        if resume_path.exists():
            checkpoint = restore_training_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                loader_generator=loader_generator,
                device=device,
                expected_metadata=run_metadata,
            )
            current_epoch = int(checkpoint["current_epoch"])
            best_validation_f1 = float(checkpoint["best_validation_f1"])
            best_epoch = int(checkpoint["best_epoch"])
            best_validation_threshold = float(
                checkpoint["best_validation_threshold"]
            )
            patience_counter = int(checkpoint["patience_counter"])
            history = list(checkpoint["history"])
            prior_training_seconds = float(
                checkpoint.get("training_time_seconds", 0.0)
            )
            if current_epoch > args.max_epochs:
                raise ValueError(
                    f"checkpoint completed epoch {current_epoch}, beyond "
                    f"requested maximum {args.max_epochs}"
                )
            print(
                f"[ordered_train_dl] resumed {args.model}/pi={pi}/s{args.seed} "
                f"after epoch {current_epoch}",
                flush=True,
            )

    training_started = time.perf_counter()
    if patience_counter < args.patience:
        for epoch in range(current_epoch + 1, args.max_epochs + 1):
            model.train()
            total_loss = 0.0
            examples_seen = 0
            for inputs, observed_labels, _latent_labels in loaders["train"]:
                inputs = inputs.to(device, non_blocking=True)
                observed_labels = observed_labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = criterion(logits, observed_labels)
                loss.backward()
                optimizer.step()
                batch_rows = int(inputs.shape[0])
                total_loss += float(loss.detach().item()) * batch_rows
                examples_seen += batch_rows

            validation = evaluate_predictions(
                model, loaders["val"], device, criterion
            )
            threshold, validation_f1 = validation_f1_threshold(
                validation["observed_labels"], validation["probabilities"]
            )
            observed_metrics = threshold_metrics(
                validation["observed_labels"],
                validation["probabilities"],
                threshold,
            )

            improved = validation_f1 > best_validation_f1
            if improved:
                best_validation_f1 = validation_f1
                best_epoch = epoch
                best_validation_threshold = threshold
                patience_counter = 0
            else:
                patience_counter += 1

            elapsed = (
                prior_training_seconds + time.perf_counter() - training_started
            )
            record = {
                "epoch": epoch,
                "train_loss": float(total_loss / examples_seen),
                "validation_loss": float(validation["loss"]),
                "validation_observed_auc": float(validation["observed_auc"]),
                "validation_latent_auc": float(validation["latent_auc"]),
                "validation_observed_f1": float(observed_metrics["f1"]),
                "validation_observed_precision": float(
                    observed_metrics["precision"]
                ),
                "validation_observed_recall": float(observed_metrics["recall"]),
                "validation_selected_threshold": float(threshold),
                "best_validation_f1": float(best_validation_f1),
                "best_epoch": int(best_epoch),
                "patience_counter": int(patience_counter),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": float(elapsed),
            }
            history.append(record)
            atomic_history_dump(history, run_dir / "history.csv")

            snapshot_validation = {
                "loss": validation["loss"],
                "observed_auc": validation["observed_auc"],
                "latent_auc": validation["latent_auc"],
                "observed_f1": observed_metrics["f1"],
                "observed_precision": observed_metrics["precision"],
                "observed_recall": observed_metrics["recall"],
                "selected_threshold": threshold,
            }
            atomic_torch_save(
                epoch_snapshot_payload(
                    model=model,
                    run_metadata=run_metadata,
                    epoch=epoch,
                    validation_metrics=snapshot_validation,
                ),
                run_dir / "epochs" / f"epoch_{epoch:03d}_eval.pt",
            )

            if improved:
                atomic_torch_save(
                    checkpoint_payload(
                        role="best_standard",
                        model=model,
                        optimizer=optimizer,
                        loader_generator=loader_generator,
                        run_metadata=run_metadata,
                        current_epoch=epoch,
                        best_validation_f1=best_validation_f1,
                        best_epoch=best_epoch,
                        best_validation_threshold=best_validation_threshold,
                        patience_counter=patience_counter,
                        history=history,
                        training_time_seconds=elapsed,
                    ),
                    best_path,
                )

            atomic_torch_save(
                checkpoint_payload(
                    role="last",
                    model=model,
                    optimizer=optimizer,
                    loader_generator=loader_generator,
                    run_metadata=run_metadata,
                    current_epoch=epoch,
                    best_validation_f1=best_validation_f1,
                    best_epoch=best_epoch,
                    best_validation_threshold=best_validation_threshold,
                    patience_counter=patience_counter,
                    history=history,
                    training_time_seconds=elapsed,
                ),
                last_path,
            )
            current_epoch = epoch

            print(
                f"[ordered_train_dl] {args.model}/pi={pi}/s{args.seed} "
                f"epoch={epoch} train_loss={record['train_loss']:.5f} "
                f"val_f1={validation_f1:.5f} "
                f"val_auc={validation['observed_auc']:.5f} "
                f"threshold={threshold:.6f} patience={patience_counter}",
                flush=True,
            )
            if patience_counter >= args.patience:
                break

    if not best_path.exists() or not last_path.exists():
        raise RuntimeError("training ended without both required checkpoints")

    training_seconds = (
        prior_training_seconds + time.perf_counter() - training_started
    )
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    selected_threshold = float(best_checkpoint["best_validation_threshold"])

    test_predictions = evaluate_predictions(
        model, loaders["test"], device, criterion
    )
    observed_test = threshold_metrics(
        test_predictions["observed_labels"],
        test_predictions["probabilities"],
        selected_threshold,
    )
    latent_test = threshold_metrics(
        test_predictions["latent_labels"],
        test_predictions["probabilities"],
        selected_threshold,
    )

    probe_inputs = datasets["test"].inputs[: min(32, len(datasets["test"]))]
    with torch.inference_mode():
        reference_logits = model(probe_inputs.to(device)).float().cpu()
    reload_max_abs_difference = verify_checkpoint_logits(
        best_path,
        probe_inputs,
        reference_logits,
        device=device,
        atol=args.reload_atol,
        rtol=args.reload_rtol,
    )

    result = {
        "status": "complete",
        **identity,
        "run_directory": str(run_dir),
        "data_root": str(data_root),
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "epochs_completed": int(current_epoch),
        "early_stopped": bool(patience_counter >= args.patience),
        "selection_metric": "observed_validation_f1",
        "best_validation_f1": float(best_checkpoint["best_validation_f1"]),
        "validation_selected_threshold": selected_threshold,
        "test_loss_observed": float(test_predictions["loss"]),
        "test_observed_auc": float(observed_test["auc"]),
        "test_observed_f1": float(observed_test["f1"]),
        "test_observed_precision": float(observed_test["precision"]),
        "test_observed_recall": float(observed_test["recall"]),
        "test_latent_auc": float(latent_test["auc"]),
        "test_latent_f1": float(latent_test["f1"]),
        "test_latent_precision": float(latent_test["precision"]),
        "test_latent_recall": float(latent_test["recall"]),
        "training_time_seconds": float(training_seconds),
        "hardware": hardware,
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_reload": {
            "examples": int(len(probe_inputs)),
            "maximum_absolute_logit_difference": reload_max_abs_difference,
            "absolute_tolerance": float(args.reload_atol),
            "relative_tolerance": float(args.reload_rtol),
            "passed": True,
        },
        "checkpoints": {
            "best_standard": str(best_path),
            "last": str(last_path),
            "epoch_snapshots": str(run_dir / "epochs" / "epoch_NNN_eval.pt"),
        },
        "config": str(config_path),
        "history": str(run_dir / "history.csv"),
        "split_summary": run_metadata["split_summary"],
        "data_fingerprints": run_metadata["data_fingerprints"],
    }
    atomic_json_dump(result, done_path)
    print(
        f"[ordered_train_dl] DONE {args.model}/pi={pi}/s{args.seed} "
        f"observed_auc={observed_test['auc']:.5f} "
        f"latent_auc={latent_test['auc']:.5f} best_epoch={best_epoch}",
        flush=True,
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--noise-pi", type=float, choices=NOISE_LEVELS, required=True)
    parser.add_argument("--seed", type=int, required=True, help="model seed")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-test-rows", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="fully resumable best_standard.pt or last.pt (default: run last.pt)",
    )
    parser.add_argument("--reload-atol", type=float, default=1e-6)
    parser.add_argument("--reload-rtol", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
