"""Shared data and one-hole oracle artifacts for the Ordered Compliance audit.

This module deliberately separates the immutable examples and latent labels
from the observed labels used at each noise level:

``splits/{train,val,test}.parquet``
    ``sequence_id, X, Y_star``

``noise/{train,val,test}.parquet``
    one common uniform variate per example and the four nested observed-label
    and flip-mask columns.

The hole manifest contains one row per held-out base sequence.  Candidate
labels and position metadata are stored as compact arrays in
``hole_oracle_labels.npz``; completed candidate strings are never persisted.
All candidate correctness comes from :func:`oc_label_tokens`, the repository's
canonical OC oracle.

Run from ``repro/`` (or set ``PYTHONPATH`` accordingly)::

    python -m src.oc_completion.ordered_data --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.oc_completion.gen_datasets import generate_pool
from src.oc_completion.oracle import (
    ALPHABET,
    KEY_SET,
    LAG,
    MECHANISM,
    N_EVENTS,
    NON_KEY_LETTERS,
    SEP,
    oc_label_tokens,
)


REPO_ROOT = Path(os.environ.get("LLMSEQ_ROOT", Path(__file__).resolve().parents[3]))
DEFAULT_DATA_ROOT = (
    Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
    / "simulation"
    / "oc_hole_audit"
)
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("RESULTS_DIR", REPO_ROOT / "results" / "oc_hole_audit")
)

NOISE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3)
FULL_SIZES = {"train": 400_000, "val": 50_000, "test": 50_000}
SMOKE_SIZES = {"train": 2_000, "val": 500, "test": 500}
FULL_HOLE_SIZES = {"validation": 2_000, "test": 10_000, "diagnostic": 500}
SMOKE_HOLE_SIZES = {"validation": 100, "test": 100, "diagnostic": 100}
SPLIT_ORDER = ("train", "val", "test")

CATEGORY_FIXED_ZERO = np.uint8(0)
CATEGORY_DECISIVE = np.uint8(1)
CATEGORY_FIXED_ONE = np.uint8(2)
CATEGORY_ENCODING = {
    "fixed_zero": int(CATEGORY_FIXED_ZERO),
    "decisive": int(CATEGORY_DECISIVE),
    "fixed_one": int(CATEGORY_FIXED_ONE),
}


def pi_tag(pi: float) -> str:
    """Stable filename/column tag for a supported noise probability."""
    value = float(pi)
    if value not in NOISE_LEVELS:
        raise ValueError(f"unsupported noise level {pi}; expected one of {NOISE_LEVELS}")
    return f"{value:.1f}".replace(".", "p")


def observed_column(pi: float) -> str:
    return f"Y_pi_{pi_tag(pi)}"


def flip_column(pi: float) -> str:
    return f"flip_pi_{pi_tag(pi)}"


def observed_label_oracle_auc(latent_prevalence: float, pi: float) -> float | None:
    """Paper oracle AUC for symmetric label noise and a binary Y* score.

    Let ``rho=P(Y*=1)`` and ``q=P(Y_pi=1)=pi+rho(1-2pi)``.  A binary oracle
    score has ``AUC = (1 + TPR - FPR) / 2`` with conditionals taken against
    the observed noisy label.  ``None`` is returned when observed labels have
    only one class, for which ROC AUC is undefined.
    """
    rho = float(latent_prevalence)
    noise = float(pi)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"latent_prevalence must lie in [0,1], got {rho}")
    if not 0.0 <= noise <= 1.0:
        raise ValueError(f"pi must lie in [0,1], got {noise}")
    q = noise + rho * (1.0 - 2.0 * noise)
    if q <= 0.0 or q >= 1.0:
        return None
    true_positive_rate = rho * (1.0 - noise) / q
    false_positive_rate = rho * noise / (1.0 - q)
    return float(0.5 * (1.0 + true_positive_rate - false_positive_rate))


def _validate_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    toks = tuple(tokens)
    if len(toks) != N_EVENTS:
        raise ValueError(f"OC sequences must have {N_EVENTS} events, got {len(toks)}")
    invalid = [token for token in toks if token not in ALPHABET]
    if invalid:
        raise ValueError(f"OC sequence contains non-alphabet tokens: {invalid[:3]}")
    return toks


def replacement_candidates(tokens: Sequence[str], position: int) -> Iterable[tuple[str, ...]]:
    """Yield the 26 complete replacements at one zero-based position."""
    toks = _validate_tokens(tokens)
    if not 0 <= position < N_EVENTS:
        raise IndexError(position)
    for letter in ALPHABET:
        candidate = list(toks)
        candidate[position] = letter
        yield tuple(candidate)


def enumerate_candidate_labels(tokens: Sequence[str]) -> np.ndarray:
    """Return canonical Y* for every position/letter replacement (20 x 26)."""
    toks = _validate_tokens(tokens)
    labels = np.empty((N_EVENTS, len(ALPHABET)), dtype=np.uint8)
    candidate = list(toks)
    for position in range(N_EVENTS):
        original = candidate[position]
        for letter_index, letter in enumerate(ALPHABET):
            candidate[position] = letter
            labels[position, letter_index] = oc_label_tokens(candidate)
        candidate[position] = original
    return labels


def _maximal_key_runs(tokens: Sequence[str]) -> list[tuple[int, ...]]:
    """Structural maximal key runs at lag LAG, including singleton runs."""
    toks = _validate_tokens(tokens)
    runs: list[tuple[int, ...]] = []
    for residue in range(LAG):
        current: list[int] = []
        for position in range(residue, N_EVENTS, LAG):
            if toks[position] in KEY_SET:
                current.append(position)
            elif current:
                runs.append(tuple(current))
                current = []
        if current:
            runs.append(tuple(current))
    return runs


def original_sequence_metadata(tokens: Sequence[str]) -> dict[str, np.ndarray | int]:
    """Return run/witness metadata while delegating witness labels to the oracle.

    The structural run enumeration does not re-implement compliance.  Each
    candidate witness run is isolated in an otherwise non-key sequence and
    classified by ``oc_label_tokens`` itself.
    """
    toks = _validate_tokens(tokens)
    run_length = np.zeros(N_EVENTS, dtype=np.uint8)
    in_witness = np.zeros(N_EVENTS, dtype=np.bool_)
    witness_count = 0
    non_key = NON_KEY_LETTERS[0]
    for run in _maximal_key_runs(toks):
        run_length[list(run)] = len(run)
        if len(run) < 2:
            continue
        isolated = [non_key] * N_EVENTS
        for position in run:
            isolated[position] = toks[position]
        if oc_label_tokens(isolated) == 1:
            witness_count += 1
            in_witness[list(run)] = True
    return {
        "original_is_key": np.asarray([token in KEY_SET for token in toks], dtype=np.bool_),
        "residue_chain_index": np.asarray(
            [(position % LAG) + 1 for position in range(N_EVENTS)], dtype=np.uint8
        ),
        "original_maximal_key_run_length": run_length,
        "position_in_ordered_witness": in_witness,
        "ordered_witness_count": witness_count,
    }


def enumerate_hole_oracle(tokens: Sequence[str]) -> dict[str, np.ndarray | int]:
    """Compute all candidate labels, exhaustive categories, and base metadata."""
    toks = _validate_tokens(tokens)
    base_y_star = int(oc_label_tokens(toks))
    labels = enumerate_candidate_labels(toks)
    num_one = labels.sum(axis=1, dtype=np.uint16).astype(np.uint8)
    num_zero = (len(ALPHABET) - num_one).astype(np.uint8)
    fixed_zero = num_one == 0
    fixed_one = num_one == len(ALPHABET)
    decisive = ~(fixed_zero | fixed_one)
    if not np.all(fixed_zero.astype(np.uint8) + fixed_one + decisive == 1):
        raise AssertionError("hole categories are not exhaustive and mutually exclusive")
    category = np.full(N_EVENTS, CATEGORY_DECISIVE, dtype=np.uint8)
    category[fixed_zero] = CATEGORY_FIXED_ZERO
    category[fixed_one] = CATEGORY_FIXED_ONE

    original_indices = np.asarray([ALPHABET.index(token) for token in toks])
    if not np.all(labels[np.arange(N_EVENTS), original_indices] == base_y_star):
        raise AssertionError("original candidate does not reproduce base Y_star")

    metadata = original_sequence_metadata(toks)
    return {
        "base_y_star": base_y_star,
        "candidate_y_star": labels,
        "category": category,
        "decisive": decisive,
        "fixed_zero": fixed_zero,
        "fixed_one": fixed_one,
        "can_create_compliance": (base_y_star == 0) & (num_one > 0),
        "can_destroy_compliance": (base_y_star == 1) & (num_zero > 0),
        "num_candidate_zero": num_zero,
        "num_candidate_one": num_one,
        **metadata,
    }


def _label_sequence(sequence: str) -> int:
    return int(oc_label_tokens(sequence.split(SEP)))


def _label_sequences(sequences: Sequence[str], n_workers: int) -> np.ndarray:
    if n_workers <= 1:
        return np.asarray([_label_sequence(sequence) for sequence in sequences], dtype=np.uint8)
    chunksize = max(1, len(sequences) // (n_workers * 16))
    with Pool(processes=n_workers) as pool:
        labels = pool.map(_label_sequence, sequences, chunksize=chunksize)
    return np.asarray(labels, dtype=np.uint8)


def construct_base_splits(
    sizes: Mapping[str, int],
    data_seed: int,
    n_workers: int = 1,
    pool_factor: float = 2.2,
) -> tuple[dict[str, pd.DataFrame], dict[str, float | int]]:
    """Generate one unique iid sequence pool and stratified immutable splits."""
    required = set(SPLIT_ORDER)
    if set(sizes) != required:
        raise ValueError(f"sizes must have exactly keys {sorted(required)}")
    if any(int(sizes[split]) <= 0 for split in SPLIT_ORDER):
        raise ValueError("all split sizes must be positive")
    n_total = sum(int(sizes[split]) for split in SPLIT_ORDER)
    n_pool = max(n_total, int(np.ceil(n_total * float(pool_factor))))
    sequences = generate_pool(n_pool, int(data_seed))
    y_pool = _label_sequences(sequences, n_workers)

    all_indices = np.arange(n_pool)
    if n_pool == n_total:
        kept = all_indices
    else:
        kept, _ = train_test_split(
            all_indices,
            train_size=n_total,
            stratify=y_pool,
            random_state=int(data_seed),
        )
    kept = np.asarray(kept)
    kept_y = y_pool[kept]

    train_idx, rest_idx = train_test_split(
        np.arange(n_total),
        train_size=int(sizes["train"]),
        stratify=kept_y,
        random_state=int(data_seed) + 1,
    )
    rest_y = kept_y[rest_idx]
    val_rel, test_rel = train_test_split(
        np.arange(len(rest_idx)),
        train_size=int(sizes["val"]),
        stratify=rest_y,
        random_state=int(data_seed) + 2,
    )
    split_kept_indices = {
        "train": train_idx,
        "val": rest_idx[val_rel],
        "test": rest_idx[test_rel],
    }

    frames: dict[str, pd.DataFrame] = {}
    for split in SPLIT_ORDER:
        selected = np.asarray(split_kept_indices[split])
        split_sequences = [sequences[int(kept[index])] for index in selected]
        split_y = kept_y[selected].astype(np.uint8)
        frames[split] = pd.DataFrame(
            {
                "sequence_id": [f"{split}_{i:09d}" for i in range(len(selected))],
                "X": split_sequences,
                "Y_star": split_y,
            }
        )
    if sum(frame["sequence_id"].nunique() for frame in frames.values()) != n_total:
        raise AssertionError("sequence IDs are not unique")
    if len(set().union(*(set(frame["X"]) for frame in frames.values()))) != n_total:
        raise AssertionError("generated splits overlap")
    return frames, {
        "pool_size": n_pool,
        "pool_latent_prevalence": float(y_pool.mean()),
    }


def make_nested_noise_tables(
    base_splits: Mapping[str, pd.DataFrame], noise_seed: int
) -> dict[str, pd.DataFrame]:
    """Create nested symmetric-noise masks from one global uniform vector."""
    for split in SPLIT_ORDER:
        required = {"sequence_id", "X", "Y_star"}
        missing = required - set(base_splits[split].columns)
        if missing:
            raise ValueError(f"{split} base split missing columns {sorted(missing)}")
    total = sum(len(base_splits[split]) for split in SPLIT_ORDER)
    uniforms = np.random.default_rng(int(noise_seed)).random(total)
    tables: dict[str, pd.DataFrame] = {}
    cursor = 0
    for split in SPLIT_ORDER:
        base = base_splits[split]
        n_rows = len(base)
        u = uniforms[cursor : cursor + n_rows]
        cursor += n_rows
        latent = base["Y_star"].to_numpy(dtype=np.uint8, copy=False)
        table = pd.DataFrame(
            {"sequence_id": base["sequence_id"].astype(str), "noise_uniform": u}
        )
        prior_mask = np.zeros(n_rows, dtype=np.bool_)
        for pi in NOISE_LEVELS:
            mask = u < pi
            if np.any(prior_mask & ~mask):
                raise AssertionError("noise masks are not nested")
            table[flip_column(pi)] = mask
            table[observed_column(pi)] = np.bitwise_xor(latent, mask.astype(np.uint8))
            prior_mask = mask
        tables[split] = table
    return tables


def load_split(data_root: Path | str, split: str, pi: float) -> pd.DataFrame:
    """Load the shared base split joined to one requested observed label."""
    if split not in SPLIT_ORDER:
        raise ValueError(f"unknown split {split}")
    root = Path(data_root)
    base = pd.read_parquet(root / "splits" / f"{split}.parquet")
    noise = pd.read_parquet(
        root / "noise" / f"{split}.parquet",
        columns=["sequence_id", "noise_uniform", flip_column(pi), observed_column(pi)],
    )
    merged = base.merge(noise, on="sequence_id", validate="one_to_one", sort=False)
    return merged.rename(
        columns={observed_column(pi): "Y_observed", flip_column(pi): "flipped"}
    )


def _stratified_sample(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    if n_rows <= 0:
        raise ValueError("sample size must be positive")
    if n_rows > len(frame):
        raise ValueError(f"cannot sample {n_rows} rows from {len(frame)}")
    if n_rows == len(frame):
        return frame.copy().reset_index(drop=True)
    selected, _ = train_test_split(
        np.arange(len(frame)),
        train_size=n_rows,
        stratify=frame["Y_star"].to_numpy(),
        random_state=int(seed),
    )
    return frame.iloc[np.sort(selected)].reset_index(drop=True)


def select_hole_manifest(
    base_splits: Mapping[str, pd.DataFrame],
    validation_size: int,
    test_size: int,
    diagnostic_size: int,
    hole_seed: int,
) -> pd.DataFrame:
    """Select fixed latent-stratified validation/test bases and diagnostics."""
    validation = _stratified_sample(base_splits["val"], validation_size, hole_seed)
    test = _stratified_sample(base_splits["test"], test_size, hole_seed + 1)
    diagnostic = _stratified_sample(validation, diagnostic_size, hole_seed + 2)
    diagnostic_ids = frozenset(diagnostic["sequence_id"].astype(str))

    validation = validation.copy()
    validation.insert(0, "hole_split", "validation")
    validation.insert(1, "source_split", "val")
    validation["is_epoch_diagnostic"] = validation["sequence_id"].isin(diagnostic_ids)
    test = test.copy()
    test.insert(0, "hole_split", "test")
    test.insert(1, "source_split", "test")
    test["is_epoch_diagnostic"] = False
    manifest = pd.concat([validation, test], ignore_index=True)
    manifest.insert(0, "oracle_row", np.arange(len(manifest), dtype=np.int64))
    # Stable evaluator-facing names.  Keep the shorter storage-layer names as
    # aliases so existing loaders can consume the same artifact.
    manifest.insert(1, "base_sequence_id", manifest["sequence_id"].astype(str))
    manifest.insert(2, "split", "")
    manifest.loc[manifest["hole_split"] == "validation", "split"] = "hole_val"
    manifest.loc[manifest["hole_split"] == "test", "split"] = "hole_test"
    manifest["base_Y_star"] = manifest["Y_star"].astype(np.uint8)
    manifest["Y_star"] = manifest["Y_star"].astype(np.uint8)
    if not manifest["sequence_id"].is_unique:
        raise AssertionError("hole base sequence IDs are not unique")
    if int(manifest["is_epoch_diagnostic"].sum()) != diagnostic_size:
        raise AssertionError("incorrect diagnostic subset size")
    return manifest


def _hole_oracle_worker(sequence: str) -> dict[str, np.ndarray | int]:
    return enumerate_hole_oracle(sequence.split(SEP))


def build_hole_oracle_arrays(
    hole_manifest: pd.DataFrame, n_workers: int = 1
) -> dict[str, np.ndarray]:
    """Build arrays aligned one-to-one with ``hole_manifest.oracle_row``."""
    sequences = hole_manifest["X"].astype(str).tolist()
    if n_workers <= 1:
        records = [_hole_oracle_worker(sequence) for sequence in sequences]
    else:
        chunksize = max(1, len(sequences) // (n_workers * 8))
        with Pool(processes=n_workers) as pool:
            records = pool.map(_hole_oracle_worker, sequences, chunksize=chunksize)
    if not records:
        raise ValueError("hole manifest is empty")

    position_keys = (
        "category",
        "decisive",
        "fixed_zero",
        "fixed_one",
        "can_create_compliance",
        "can_destroy_compliance",
        "num_candidate_zero",
        "num_candidate_one",
        "original_is_key",
        "residue_chain_index",
        "original_maximal_key_run_length",
        "position_in_ordered_witness",
    )
    arrays: dict[str, np.ndarray] = {
        "sequence_id": hole_manifest["sequence_id"].astype(str).to_numpy(dtype=str),
        "base_y_star": np.asarray([record["base_y_star"] for record in records], dtype=np.uint8),
        "candidate_y_star": np.stack(
            [record["candidate_y_star"] for record in records]
        ).astype(np.uint8, copy=False),
        "ordered_witness_count": np.asarray(
            [record["ordered_witness_count"] for record in records], dtype=np.uint8
        ),
        "alphabet": np.asarray(ALPHABET, dtype="<U1"),
        "positions": np.arange(1, N_EVENTS + 1, dtype=np.uint8),
    }
    # Preferred public name; candidate_y_star is retained for explicitness and
    # backward-compatible callers.  Both keys contain the same compact array.
    arrays["oracle_labels"] = arrays["candidate_y_star"]
    for key in position_keys:
        arrays[key] = np.stack([record[key] for record in records])

    expected_shape = (len(hole_manifest), N_EVENTS, len(ALPHABET))
    if arrays["candidate_y_star"].shape != expected_shape:
        raise AssertionError(
            f"candidate oracle shape {arrays['candidate_y_star'].shape} != {expected_shape}"
        )
    if not np.array_equal(
        arrays["base_y_star"], hole_manifest["Y_star"].to_numpy(dtype=np.uint8)
    ):
        raise AssertionError("stored hole Y_star disagrees with canonical oracle")
    return arrays


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _empirical_auc(observed: np.ndarray, latent: np.ndarray) -> float | None:
    if np.unique(observed).size < 2:
        return None
    return float(roc_auc_score(observed, latent))


def write_ordered_artifacts(
    base_splits: Mapping[str, pd.DataFrame],
    noise_tables: Mapping[str, pd.DataFrame],
    data_root: Path,
    results_root: Path,
    data_seed: int,
    noise_seed: int,
    hole_seed: int,
    hole_sizes: Mapping[str, int],
    n_workers: int,
    pool_metadata: Mapping[str, float | int] | None = None,
    command: str | None = None,
) -> tuple[dict, dict]:
    """Persist shared splits/noise tables, fixed holes, arrays, and manifests."""
    data_root = Path(data_root)
    results_root = Path(results_root)
    split_entries: dict[str, dict] = {}
    noise_entries: dict[str, dict] = {}

    for split in SPLIT_ORDER:
        base_path = data_root / "splits" / f"{split}.parquet"
        noise_path = data_root / "noise" / f"{split}.parquet"
        _write_parquet(base_path, base_splits[split])
        _write_parquet(noise_path, noise_tables[split])
        latent = base_splits[split]["Y_star"].to_numpy(dtype=np.uint8)
        split_entries[split] = {
            "n": len(base_splits[split]),
            "path": str(base_path),
            "sha256": sha256_file(base_path),
            "latent_prevalence": float(latent.mean()),
        }
        condition_entries: dict[str, dict] = {}
        for pi in NOISE_LEVELS:
            observed = noise_tables[split][observed_column(pi)].to_numpy(dtype=np.uint8)
            flips = noise_tables[split][flip_column(pi)].to_numpy(dtype=np.bool_)
            condition_entries[pi_tag(pi)] = {
                "pi": pi,
                "observed_column": observed_column(pi),
                "flip_column": flip_column(pi),
                "observed_prevalence": float(observed.mean()),
                "number_flipped": int(flips.sum()),
                "empirical_flip_fraction": float(flips.mean()),
                "observed_label_oracle_auc_formula": observed_label_oracle_auc(
                    float(latent.mean()), pi
                ),
                "observed_label_oracle_auc_empirical": _empirical_auc(observed, latent),
            }
        noise_entries[split] = {
            "n": len(noise_tables[split]),
            "path": str(noise_path),
            "sha256": sha256_file(noise_path),
            "conditions": condition_entries,
        }

    hole_manifest = select_hole_manifest(
        base_splits,
        validation_size=int(hole_sizes["validation"]),
        test_size=int(hole_sizes["test"]),
        diagnostic_size=int(hole_sizes["diagnostic"]),
        hole_seed=hole_seed,
    )
    hole_arrays = build_hole_oracle_arrays(hole_manifest, n_workers=n_workers)
    hole_manifest_path = results_root / "hole_manifest.parquet"
    hole_arrays_path = results_root / "hole_oracle_labels.npz"
    _write_parquet(hole_manifest_path, hole_manifest)
    _write_npz(hole_arrays_path, hole_arrays)

    dataset_manifest = {
        "schema_version": 1,
        "mechanism": MECHANISM,
        "data_seed": int(data_seed),
        "noise_seed": int(noise_seed),
        "hole_seed": int(hole_seed),
        "proposal": "iid Uniform(A..Z), length 20, with replacement, deduplicated pool",
        "split_procedure": (
            "latent-stratified fixed split; random states data_seed, data_seed+1, "
            "data_seed+2"
        ),
        "base_schema": {"sequence_id": "string", "X": "string", "Y_star": "uint8"},
        "sizes": {split: len(base_splits[split]) for split in SPLIT_ORDER},
        "splits": split_entries,
        "hole_artifacts": {
            "manifest_path": str(hole_manifest_path),
            "manifest_sha256": sha256_file(hole_manifest_path),
            "oracle_arrays_path": str(hole_arrays_path),
            "oracle_arrays_sha256": sha256_file(hole_arrays_path),
            "oracle_label_shape": list(hole_arrays["candidate_y_star"].shape),
            "category_encoding": CATEGORY_ENCODING,
            "validation_bases": int((hole_manifest["hole_split"] == "validation").sum()),
            "test_bases": int((hole_manifest["hole_split"] == "test").sum()),
            "diagnostic_bases": int(hole_manifest["is_epoch_diagnostic"].sum()),
            "diagnostic_policy": "latent-stratified subset of hole validation bases",
        },
        "command": command,
        **dict(pool_metadata or {}),
    }
    noise_manifest = {
        "schema_version": 1,
        "mechanism": MECHANISM,
        "data_seed": int(data_seed),
        "noise_seed": int(noise_seed),
        "noise_levels": list(NOISE_LEVELS),
        "construction": "Y_pi = Y_star XOR (shared_uniform < pi)",
        "one_uniform_per_example": True,
        "nested_flip_masks": True,
        "global_uniform_order": list(SPLIT_ORDER),
        "splits": noise_entries,
        "command": command,
    }
    for root in (data_root, results_root):
        _write_json(root / "dataset_manifest.json", dataset_manifest)
        _write_json(root / "noise_manifest.json", noise_manifest)
    return dataset_manifest, noise_manifest


def build_ordered_data(
    data_root: Path = DEFAULT_DATA_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    sizes: Mapping[str, int] = FULL_SIZES,
    hole_sizes: Mapping[str, int] = FULL_HOLE_SIZES,
    data_seed: int = 9550,
    noise_seed: int = 9650,
    hole_seed: int = 9750,
    n_workers: int = 1,
    pool_factor: float = 2.2,
    command: str | None = None,
) -> tuple[dict, dict]:
    """Generate and persist the complete shared/nested-noise data layer."""
    started = time.time()
    base_splits, pool_metadata = construct_base_splits(
        sizes=sizes,
        data_seed=data_seed,
        n_workers=n_workers,
        pool_factor=pool_factor,
    )
    noise_tables = make_nested_noise_tables(base_splits, noise_seed=noise_seed)
    dataset_manifest, noise_manifest = write_ordered_artifacts(
        base_splits=base_splits,
        noise_tables=noise_tables,
        data_root=Path(data_root),
        results_root=Path(results_root),
        data_seed=data_seed,
        noise_seed=noise_seed,
        hole_seed=hole_seed,
        hole_sizes=hole_sizes,
        n_workers=n_workers,
        pool_metadata=pool_metadata,
        command=command,
    )
    dataset_manifest["generation_time_s"] = round(time.time() - started, 3)
    # Refresh both copies with final timing included.
    for root in (Path(data_root), Path(results_root)):
        _write_json(root / "dataset_manifest.json", dataset_manifest)
    return dataset_manifest, noise_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-seed", type=int, default=9550)
    parser.add_argument("--noise-seed", type=int, default=9650)
    parser.add_argument("--hole-seed", type=int, default=9750)
    parser.add_argument("--n-workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--pool-factor", type=float, default=2.2)
    parser.add_argument("--hole-validation-size", type=int, default=None)
    parser.add_argument("--hole-test-size", type=int, default=None)
    parser.add_argument("--diagnostic-size", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sizes = SMOKE_SIZES if args.smoke else FULL_SIZES
    defaults = SMOKE_HOLE_SIZES if args.smoke else FULL_HOLE_SIZES
    hole_sizes = {
        "validation": args.hole_validation_size or defaults["validation"],
        "test": args.hole_test_size or defaults["test"],
        "diagnostic": args.diagnostic_size or defaults["diagnostic"],
    }
    command = " ".join(sys.argv if argv is None else ["ordered_data", *argv])
    dataset_manifest, noise_manifest = build_ordered_data(
        data_root=args.data_root,
        results_root=args.results_root,
        sizes=sizes,
        hole_sizes=hole_sizes,
        data_seed=args.data_seed,
        noise_seed=args.noise_seed,
        hole_seed=args.hole_seed,
        n_workers=args.n_workers,
        pool_factor=args.pool_factor,
        command=command,
    )
    print(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "results_root": str(args.results_root),
                "sizes": dataset_manifest["sizes"],
                "hole_label_shape": dataset_manifest["hole_artifacts"]["oracle_label_shape"],
                "noise_levels": noise_manifest["noise_levels"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
