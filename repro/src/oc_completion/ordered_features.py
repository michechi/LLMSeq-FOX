"""Canonical feature views used by Ordered Compliance shortcut controls."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from src.analysis.mechanism_id.scripts.common import (
    feat_count26,
    feat_lag_pair,
    feat_lag_pair_position_sparse_row,
)
from src.oc_completion.oracle import KEY_SET, LAG, N_EVENTS, SEP


def as_tokens(sequence: str | Sequence[str]) -> list[str]:
    if isinstance(sequence, str):
        return sequence.split(SEP) if SEP in sequence else list(sequence)
    return list(sequence)


def residue_chains(tokens: Sequence[str]) -> list[list[str]]:
    """All seven complete modulo-lag chains (six length 3, one length 2)."""
    if len(tokens) != N_EVENTS:
        raise ValueError(f"expected {N_EVENTS} tokens, got {len(tokens)}")
    return [[tokens[p] for p in range(r, N_EVENTS, LAG)] for r in range(LAG)]


def chain_key_counts(tokens: Sequence[str]) -> np.ndarray:
    return np.asarray([
        sum(letter in KEY_SET for letter in chain)
        for chain in residue_chains(tokens)
    ], dtype=np.int16)


def maximal_key_runs(tokens: Sequence[str]) -> tuple[int, ...]:
    """Lengths of all maximal consecutive key blocks within residue chains."""
    lengths: list[int] = []
    for chain in residue_chains(tokens):
        run = 0
        for letter in chain:
            if letter in KEY_SET:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return tuple(lengths)


def maximal_run_histogram(tokens: Sequence[str]) -> np.ndarray:
    hist = np.zeros(3, dtype=np.int16)
    for length in maximal_key_runs(tokens):
        hist[length - 1] += 1
    return hist


def occupancy_features(tokens: Sequence[str]) -> np.ndarray:
    """Requested chain-count/profile/histogram/max/run-hist feature vector."""
    counts = chain_key_counts(tokens)
    sorted_counts = np.sort(counts)
    occupancy_hist = np.bincount(counts, minlength=4)[:4]
    return np.concatenate([
        counts,
        sorted_counts,
        occupancy_hist.astype(np.int16),
        np.asarray([counts.max(initial=0)], dtype=np.int16),
        maximal_run_histogram(tokens),
    ]).astype(np.float32)


def per_chain_unordered_bags(tokens: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(sorted(chain)) for chain in residue_chains(tokens))


def depth_letter_multisets(tokens: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    chains = residue_chains(tokens)
    return tuple(
        tuple(sorted(chain[depth] for chain in chains if len(chain) > depth))
        for depth in range(3)
    )


def directional_key_edge_counts(tokens: Sequence[str]) -> tuple[int, int, int]:
    """Increasing, decreasing, and equal directed key-key lag edges."""
    from src.oc_completion.oracle import KAPPA

    increasing = decreasing = equal = 0
    for i in range(N_EVENTS - LAG):
        a, b = tokens[i], tokens[i + LAG]
        if a not in KEY_SET or b not in KEY_SET:
            continue
        if KAPPA[a] < KAPPA[b]:
            increasing += 1
        elif KAPPA[a] > KAPPA[b]:
            decreasing += 1
        else:
            equal += 1
    return increasing, decreasing, equal


def lag_trigram_sparse_row(tokens: Sequence[str]):
    counts: Counter[int] = Counter()
    for t in range(len(tokens) - 2 * LAG):
        a, b, c = (ord(tokens[t]) - 65, ord(tokens[t + LAG]) - 65,
                   ord(tokens[t + 2 * LAG]) - 65)
        counts[a * 26 * 26 + b * 26 + c] += 1
    cols = sorted(counts)
    return cols, [float(counts[c]) for c in cols], 26 ** 3


def _sparse_matrix(sequences, row_fn) -> csr_matrix:
    indptr, indices, values = [0], [], []
    ncols = 0
    for sequence in sequences:
        cols, data, width = row_fn(as_tokens(sequence))
        ncols = width
        indices.extend(cols)
        values.extend(data)
        indptr.append(len(indices))
    return csr_matrix((np.asarray(values, dtype=np.float32),
                       np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int64)),
                      shape=(len(sequences), ncols))


def build_features(sequences, family: str):
    """Build one of the required shortcut representations."""
    if family == "letter_count":
        return np.stack([feat_count26(as_tokens(s)) for s in sequences])
    if family == "lag_pair":
        return np.stack([feat_lag_pair(as_tokens(s), LAG) for s in sequences])
    if family == "chain_occupancy":
        return np.stack([occupancy_features(as_tokens(s)) for s in sequences])
    if family == "chain_counts_xgb":
        return np.stack([chain_key_counts(as_tokens(s)) for s in sequences])
    if family == "position_lag_pair":
        return _sparse_matrix(
            sequences,
            lambda t: feat_lag_pair_position_sparse_row(t, LAG, N_EVENTS),
        )
    if family == "lag_trigram":
        return _sparse_matrix(sequences, lag_trigram_sparse_row)
    raise KeyError(f"unknown feature family: {family}")


DETERMINISTIC_OCCUPANCY_FAMILIES = (
    "occupancy_max_count",
    "occupancy_n_chains_ge2",
    "occupancy_max_run",
)


def deterministic_occupancy_score(sequences, family: str) -> np.ndarray:
    scores = []
    for sequence in sequences:
        toks = as_tokens(sequence)
        counts = chain_key_counts(toks)
        if family == "occupancy_max_count":
            value = counts.max(initial=0)
        elif family == "occupancy_n_chains_ge2":
            value = np.sum(counts >= 2)
        elif family == "occupancy_max_run":
            runs = maximal_key_runs(toks)
            value = max(runs, default=0)
        else:
            raise KeyError(family)
        scores.append(float(value))
    return np.asarray(scores, dtype=np.float64)
