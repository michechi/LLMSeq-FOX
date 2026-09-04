"""Strict four-chain matched pairs for the Ordered-Compliance audit.

This module adds the primary matched-pair family requested by the exhaustive
hole audit.  It deliberately leaves :mod:`src.oc_completion.gen_pairs`
unchanged: that module's historical ``two_hole`` family is retained as a
useful control, but is exposed here under the descriptive name
``occupancy_changing_legacy`` rather than as evidence of order learning.

For four key letters ``kappa(a) < kappa(b) < kappa(c) < kappa(d)``, four
complete lag-7 residue chains receive these templates::

                 negative       positive
        row 1    a c b          a b c
        row 2    a d c          d a c
        row 3    d a b          a d b
        row 4    d b c          d c b

The construction changes the canonical latent label while exactly matching
letter counts, aggregated directional lag-7 pair counts, the key mask, chain
occupancy, maximal-run structure, per-chain unordered bags, depth-wise letter
multisets, and global counts of increasing/decreasing key edges.

The command-line interface writes one Parquet manifest containing clean and
held-out validation/test pairs plus a JSON generation report.  Held-out pairs
are explicitly conditioned on the retained background not independently
making the negative candidate compliant.

Run from ``repro/``::

    DATA_DIR=/root/LLMSeq/data python -m src.oc_completion.strict_pairs
    DATA_DIR=/root/LLMSeq/data python -m src.oc_completion.strict_pairs --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.mechanism_id.scripts.common import feat_count26, feat_lag_pair
from src.oc_completion.oracle import (
    ALPHABET,
    KAPPA,
    KEY_LETTERS,
    KEY_SET,
    LAG,
    MECHANISM,
    N_EVENTS,
    NON_KEY_LETTERS,
    SEP,
    oc_label_tokens,
)


REPO_ROOT = Path(os.environ.get("LLMSEQ_ROOT", "/root/LLMSeq"))
DATA_ROOT = (
    Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
    / "simulation"
    / "oc_hole_audit"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "oc_hole_audit"

STRICT_FAMILY = "strict_four_chain"
LEGACY_SOURCE_FAMILY = "two_hole"
LEGACY_FAMILY = "occupancy_changing_legacy"
PAIR_SEED = 9700
FULL_SIZES = {"val": 2_000, "test": 10_000}
SMOKE_SIZES = {"val": 200, "test": 200}

# The first six residue chains are the complete length-three chains in a
# length-20 sequence at lag seven.  The seventh residue chain has length two
# and is never selected by the strict template.
COMPLETE_CHAIN_STARTS = tuple(range(N_EVENTS - 2 * LAG))
ALL_CHAIN_STARTS = tuple(range(LAG))

LEGACY_CONTROL_METADATA = {
    "source_module": "src.oc_completion.gen_pairs",
    "source_family": LEGACY_SOURCE_FAMILY,
    "audit_name": LEGACY_FAMILY,
    "role": "legacy_control",
    "primary_matched_test": False,
    "warning": (
        "The historical a,b,x two-hole family changes per-chain key "
        "occupancy and is not direct evidence of order learning."
    ),
}


def legacy_family_name(source_family: str) -> str:
    """Return the audit-facing name for an old pair-family identifier."""
    if source_family == LEGACY_SOURCE_FAMILY:
        return LEGACY_FAMILY
    return source_family


def chain_positions(start: int) -> tuple[int, ...]:
    """Positions in one lag-7 residue chain (zero-indexed)."""
    if start not in ALL_CHAIN_STARTS:
        raise ValueError(f"chain start must be in {ALL_CHAIN_STARTS}, got {start}")
    return tuple(range(start, N_EVENTS, LAG))


def _tokens(sequence: str | Sequence[str]) -> list[str]:
    if isinstance(sequence, str):
        toks = sequence.split(SEP) if SEP in sequence else list(sequence)
    else:
        toks = list(sequence)
    if len(toks) != N_EVENTS:
        raise ValueError(f"strict pair sequences must have length {N_EVENTS}")
    if any(t not in ALPHABET for t in toks):
        raise ValueError("strict pair sequences must contain only A,...,Z")
    return toks


def key_membership_mask(toks: Sequence[str]) -> np.ndarray:
    return np.fromiter((t in KEY_SET for t in toks), dtype=np.uint8, count=len(toks))


def per_chain_key_counts(toks: Sequence[str]) -> np.ndarray:
    """Key occupancy for all seven residue chains, in residue order."""
    return np.asarray(
        [sum(toks[p] in KEY_SET for p in chain_positions(s))
         for s in ALL_CHAIN_STARTS],
        dtype=np.int8,
    )


def sorted_chain_occupancy_profile(toks: Sequence[str]) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in per_chain_key_counts(toks)))


def maximal_run_length_histogram(toks: Sequence[str]) -> np.ndarray:
    """Histogram of maximal consecutive key-run lengths along residue chains.

    Index ``q`` is the number of maximal key runs of length ``q``.  Index zero
    is retained (and always zero) so the representation has a fixed shape.
    """
    hist = np.zeros(N_EVENTS + 1, dtype=np.int16)
    for start in ALL_CHAIN_STARTS:
        run = 0
        for p in chain_positions(start):
            if toks[p] in KEY_SET:
                run += 1
            elif run:
                hist[run] += 1
                run = 0
        if run:
            hist[run] += 1
    return hist


def per_chain_unordered_letter_bags(
    toks: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    """Sorted full-letter multiset within each of the seven residue chains."""
    return tuple(tuple(sorted(toks[p] for p in chain_positions(s)))
                 for s in ALL_CHAIN_STARTS)


def depth_letter_multisets(toks: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Letter multiset across all residue chains at each chain depth."""
    max_depth = max(len(chain_positions(s)) for s in ALL_CHAIN_STARTS)
    depths = []
    for depth in range(max_depth):
        letters = [toks[s + depth * LAG] for s in ALL_CHAIN_STARTS
                   if s + depth * LAG < N_EVENTS]
        depths.append(tuple(sorted(letters)))
    return tuple(depths)


def selected_depth_letter_multisets(
    toks: Sequence[str], selected_starts: Sequence[int]
) -> tuple[tuple[str, ...], ...]:
    """Letter multisets at depths 0,1,2 across the four selected chains."""
    return tuple(
        tuple(sorted(toks[s + depth * LAG] for s in selected_starts))
        for depth in range(3)
    )


def key_edge_direction_counts(toks: Sequence[str]) -> tuple[int, int, int]:
    """Counts of strictly increasing, strictly decreasing, and tied key edges."""
    increasing = decreasing = equal = 0
    for p in range(N_EVENTS - LAG):
        left, right = toks[p], toks[p + LAG]
        if left not in KEY_SET or right not in KEY_SET:
            continue
        if KAPPA[left] < KAPPA[right]:
            increasing += 1
        elif KAPPA[left] > KAPPA[right]:
            decreasing += 1
        else:
            equal += 1
    return increasing, decreasing, equal


def ordered_selected_chains(
    toks: Sequence[str], selected_starts: Sequence[int]
) -> tuple[int, ...]:
    """Selected complete chains whose three key ranks are non-decreasing."""
    ordered = []
    for start in selected_starts:
        chain = [toks[p] for p in chain_positions(int(start))]
        if (len(chain) >= 2 and all(t in KEY_SET for t in chain)
                and all(KAPPA[x] <= KAPPA[y]
                        for x, y in zip(chain, chain[1:]))):
            ordered.append(int(start))
    return tuple(ordered)


def _ordered_maximal_run_count(toks: Sequence[str]) -> int:
    """Descriptive count used only for background-generation statistics.

    Latent acceptance is *always* decided by :func:`oc_label_tokens`; this
    helper does not provide or replace the canonical label rule.
    """
    count = 0
    for start in ALL_CHAIN_STARTS:
        chain = [toks[p] for p in chain_positions(start)]
        run: list[str] = []
        for letter in chain + [NON_KEY_LETTERS[0]]:
            if letter in KEY_SET:
                run.append(letter)
                continue
            if (len(run) >= 2
                    and all(KAPPA[x] <= KAPPA[y]
                            for x, y in zip(run, run[1:]))):
                count += 1
            run = []
    return count


def strict_representations(toks: Sequence[str]) -> dict[str, Any]:
    """All representations required to tie within a strict pair."""
    return {
        "letter_counts": feat_count26(toks),
        "lag_pair_tensor": feat_lag_pair(toks, LAG).reshape(26, 26),
        "key_mask": key_membership_mask(toks),
        "per_chain_key_counts": per_chain_key_counts(toks),
        "sorted_chain_occupancy": sorted_chain_occupancy_profile(toks),
        "maximal_run_histogram": maximal_run_length_histogram(toks),
        "per_chain_unordered_bags": per_chain_unordered_letter_bags(toks),
        "depth_letter_multisets": depth_letter_multisets(toks),
        "key_edge_directions": key_edge_direction_counts(toks),
    }


def strict_templates(
    key_letters: Sequence[str],
) -> tuple[tuple[tuple[str, str, str], ...],
           tuple[tuple[str, str, str], ...]]:
    """Return the four negative and four positive template rows."""
    if len(key_letters) != 4:
        raise ValueError("strict templates require exactly four key letters")
    a, b, c, d = key_letters
    if (len(set(key_letters)) != 4 or any(k not in KEY_SET for k in key_letters)
            or not (KAPPA[a] < KAPPA[b] < KAPPA[c] < KAPPA[d])):
        raise ValueError("keys must be distinct and ordered by kappa")
    negative = ((a, c, b), (a, d, c), (d, a, b), (d, b, c))
    positive = ((a, b, c), (d, a, c), (a, d, b), (d, c, b))
    return negative, positive


def _fill_chain(toks: list[str], start: int, letters: Sequence[str]) -> None:
    positions = chain_positions(start)
    if len(positions) != 3 or len(letters) != 3:
        raise ValueError("strict templates require complete length-three chains")
    for p, letter in zip(positions, letters):
        toks[p] = letter


def construct_strict_pair(
    base: str | Sequence[str],
    key_letters: Sequence[str],
    template_to_chain: Sequence[int],
    *,
    background: str,
    base_sequence_id: str | int | None = None,
    base_index: int = -1,
) -> dict[str, Any]:
    """Construct one pair from explicit keys/chains (useful for audit tests)."""
    base_toks = _tokens(base)
    starts = tuple(int(s) for s in template_to_chain)
    if len(starts) != 4 or len(set(starts)) != 4:
        raise ValueError("template_to_chain must contain four distinct starts")
    if any(s not in COMPLETE_CHAIN_STARTS for s in starts):
        raise ValueError("strict templates may use only complete residue chains")

    negative_rows, positive_rows = strict_templates(tuple(key_letters))
    negative, positive = list(base_toks), list(base_toks)
    for row_index, start in enumerate(starts):
        _fill_chain(negative, start, negative_rows[row_index])
        _fill_chain(positive, start, positive_rows[row_index])

    selected_positions = {
        p for start in starts for p in chain_positions(start)
    }
    retained_background = list(base_toks)
    for p in selected_positions:
        retained_background[p] = NON_KEY_LETTERS[0]

    return {
        "family": STRICT_FAMILY,
        "background": background,
        "base_sequence_id": (None if base_sequence_id is None
                             else str(base_sequence_id)),
        "base_index": int(base_index),
        "base": base_toks,
        "negative": negative,
        "positive": positive,
        "key_letters": tuple(key_letters),
        "selected_chain_starts": tuple(sorted(starts)),
        "template_to_chain": starts,
        "background_key_count": int(sum(t in KEY_SET
                                         for t in retained_background)),
        "background_ordered_run_count": _ordered_maximal_run_count(
            retained_background
        ),
    }


def assert_strict_pair_invariants(rec: Mapping[str, Any]) -> None:
    """Assert the complete strict-pair invariant checklist."""
    negative = _tokens(rec["negative"])
    positive = _tokens(rec["positive"])
    keys = tuple(rec["key_letters"])
    starts = tuple(int(s) for s in rec["selected_chain_starts"])
    template_to_chain = tuple(int(s) for s in rec["template_to_chain"])

    assert len(negative) == len(positive) == N_EVENTS, "sequence length differs"
    assert len(keys) == len(set(keys)) == 4, "selected keys are not distinct"
    assert all(k in KEY_SET for k in keys), "selected letter outside S"
    assert tuple(sorted(keys, key=KAPPA.__getitem__)) == keys, \
        "selected letters are not ordered by kappa"
    assert all(KAPPA[x] < KAPPA[y] for x, y in zip(keys, keys[1:])), \
        "selected key ranks are not strictly increasing"
    assert len(starts) == len(set(starts)) == 4, "selected chains not distinct"
    assert all(s in COMPLETE_CHAIN_STARTS for s in starts), "invalid chain start"
    assert set(template_to_chain) == set(starts), "template assignment mismatch"

    assert oc_label_tokens(negative) == 0, "negative Y_star != 0"
    assert oc_label_tokens(positive) == 1, "positive Y_star != 1"

    negative_rows, positive_rows = strict_templates(keys)
    for row_index, start in enumerate(template_to_chain):
        positions = chain_positions(start)
        assert tuple(negative[p] for p in positions) == negative_rows[row_index], \
            "negative selected-chain template changed"
        assert tuple(positive[p] for p in positions) == positive_rows[row_index], \
            "positive selected-chain template changed"

    neg_ordered = ordered_selected_chains(negative, starts)
    pos_ordered = ordered_selected_chains(positive, starts)
    assert not neg_ordered, "negative has an ordered selected chain"
    assert len(pos_ordered) == 1, "positive does not have exactly one ordered selected chain"
    assert pos_ordered[0] == template_to_chain[0], \
        "the unique ordered chain is not the a,b,c template"

    # Each chosen letter occurs exactly three times inside the selected block.
    selected_positions = [p for s in starts for p in chain_positions(s)]
    assert Counter(negative[p] for p in selected_positions) == Counter({k: 3 for k in keys})
    assert Counter(positive[p] for p in selected_positions) == Counter({k: 3 for k in keys})

    neg_repr = strict_representations(negative)
    pos_repr = strict_representations(positive)
    for name in (
        "letter_counts",
        "lag_pair_tensor",
        "key_mask",
        "per_chain_key_counts",
        "maximal_run_histogram",
    ):
        assert np.array_equal(neg_repr[name], pos_repr[name]), f"{name} differs"
    for name in (
        "sorted_chain_occupancy",
        "per_chain_unordered_bags",
        "depth_letter_multisets",
        "key_edge_directions",
    ):
        assert neg_repr[name] == pos_repr[name], f"{name} differs"

    # The selected four chains also match their depth-wise multisets exactly.
    assert selected_depth_letter_multisets(negative, starts) == \
        selected_depth_letter_multisets(positive, starts), \
        "selected-chain depth multisets differ"
    neg_inc, neg_dec, _ = key_edge_direction_counts(negative)
    pos_inc, pos_dec, _ = key_edge_direction_counts(positive)
    assert (neg_inc, neg_dec) == (pos_inc, pos_dec), \
        "increasing/decreasing key-edge counts differ"


def _sample_attempt(
    rng: np.random.Generator,
    background: str,
    base: str | Sequence[str] | None,
    base_sequence_id: str | int | None,
    base_index: int,
) -> tuple[dict[str, Any] | None, str | None, dict[str, int]]:
    if background == "clean":
        base_toks = [NON_KEY_LETTERS[int(i)] for i in
                     rng.integers(len(NON_KEY_LETTERS), size=N_EVENTS)]
        base_sequence_id = None
        base_index = -1
    elif background == "heldout":
        if base is None:
            raise ValueError("heldout strict pairs require a base sequence")
        base_toks = _tokens(base)
    else:
        raise ValueError("background must be 'clean' or 'heldout'")

    sampled_keys = rng.choice(np.asarray(KEY_LETTERS), size=4, replace=False)
    keys = tuple(sorted((str(k) for k in sampled_keys), key=KAPPA.__getitem__))
    selected = np.sort(
        rng.choice(np.asarray(COMPLETE_CHAIN_STARTS), size=4, replace=False)
    )
    # Template row -> residue chain is a separate random permutation.
    template_to_chain = tuple(int(selected[i]) for i in rng.permutation(4))
    rec = construct_strict_pair(
        base_toks,
        keys,
        template_to_chain,
        background=background,
        base_sequence_id=base_sequence_id,
        base_index=base_index,
    )
    diagnostics = {
        "background_key_count": rec["background_key_count"],
        "background_ordered_run_count": rec["background_ordered_run_count"],
    }

    y_negative = oc_label_tokens(rec["negative"])
    y_positive = oc_label_tokens(rec["positive"])
    if y_negative != 0:
        return None, "negative_compliant_background", diagnostics
    if y_positive != 1:
        return None, "positive_noncompliant", diagnostics
    assert_strict_pair_invariants(rec)
    return rec, None, diagnostics


def make_strict_pair(
    rng: np.random.Generator,
    background: str,
    base: str | Sequence[str] | None = None,
    *,
    base_sequence_id: str | int | None = None,
    base_index: int = -1,
) -> tuple[dict[str, Any] | None, str | None]:
    """Make one randomized strict-pair attempt.

    A held-out attempt can be rejected when the retained background makes the
    negative sequence compliant.  The positive candidate is always checked by
    the canonical oracle as well.
    """
    rec, reason, _ = _sample_attempt(
        rng, background, base, base_sequence_id, base_index
    )
    return rec, reason


def _counter_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(int(key)): int(counter[key]) for key in sorted(counter)}


def _row_from_record(
    rec: Mapping[str, Any],
    rng: np.random.Generator,
    split: str,
    generation_order: int,
) -> dict[str, Any]:
    positive_index = int(rng.integers(2))
    negative = SEP.join(rec["negative"])
    positive = SEP.join(rec["positive"])
    candidates = [negative, negative]
    candidates[positive_index] = positive
    candidates[1 - positive_index] = negative
    keys = tuple(rec["key_letters"])
    template_to_chain = tuple(int(s) for s in rec["template_to_chain"])
    inc, dec, tied = key_edge_direction_counts(rec["negative"])
    return {
        "pair_id": "",  # assigned in randomized order after generation
        "pair_id_order": -1,
        "generation_order": int(generation_order),
        "family": STRICT_FAMILY,
        "background": rec["background"],
        "split": split,
        "base_sequence_id": rec["base_sequence_id"],
        # Compatibility alias used by the matched-pair evaluator.  Keep both
        # names so clustered inference can use either schema without guessing.
        "base_id": rec["base_sequence_id"],
        "base_index": int(rec["base_index"]),
        "key_a": keys[0],
        "key_b": keys[1],
        "key_c": keys[2],
        "key_d": keys[3],
        "kappa_a": int(KAPPA[keys[0]]),
        "kappa_b": int(KAPPA[keys[1]]),
        "kappa_c": int(KAPPA[keys[2]]),
        "kappa_d": int(KAPPA[keys[3]]),
        "template1_chain": template_to_chain[0] + 1,
        "template2_chain": template_to_chain[1] + 1,
        "template3_chain": template_to_chain[2] + 1,
        "template4_chain": template_to_chain[3] + 1,
        "selected_chains": ",".join(
            str(s + 1) for s in rec["selected_chain_starts"]
        ),
        "background_key_count": int(rec["background_key_count"]),
        "background_ordered_run_count": int(
            rec["background_ordered_run_count"]
        ),
        "increasing_key_edges": int(inc),
        "decreasing_key_edges": int(dec),
        "tied_key_edges": int(tied),
        "positive_index": positive_index,
        "cand0_Y_star": int(positive_index == 0),
        "cand1_Y_star": int(positive_index == 1),
        "cand0": candidates[0],
        "cand1": candidates[1],
    }


def record_from_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the invariant-record form from one manifest row."""
    positive_index = int(row["positive_index"])
    positive = _tokens(row[f"cand{positive_index}"])
    negative = _tokens(row[f"cand{1 - positive_index}"])
    keys = tuple(str(row[f"key_{letter}"]) for letter in "abcd")
    template_to_chain = tuple(int(row[f"template{i}_chain"]) - 1
                              for i in range(1, 5))
    return {
        "family": STRICT_FAMILY,
        "background": row["background"],
        "base_sequence_id": row.get("base_sequence_id"),
        "base_index": int(row.get("base_index", -1)),
        "negative": negative,
        "positive": positive,
        "key_letters": keys,
        "selected_chain_starts": tuple(sorted(template_to_chain)),
        "template_to_chain": template_to_chain,
    }


def assert_manifest_row_invariants(row: Mapping[str, Any]) -> None:
    """Validate labels, storage orientation, and every pair invariant."""
    positive_index = int(row["positive_index"])
    assert positive_index in (0, 1), "positive_index must be zero or one"
    assert int(row[f"cand{positive_index}_Y_star"]) == 1
    assert int(row[f"cand{1 - positive_index}_Y_star"]) == 0
    rec = record_from_manifest_row(row)
    assert_strict_pair_invariants(rec)


def generate_strict_family(
    background: str,
    split: str,
    n_pairs: int,
    rng: np.random.Generator,
    heldout_bases: Sequence[str] | None = None,
    base_sequence_ids: Sequence[str | int] | None = None,
    *,
    id_prefix: str | None = None,
    max_factor: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one strict family/background/split manifest.

    For held-out backgrounds, unaccepted bases are retried with new templates
    before any already-accepted base is reused.  Therefore at most one pair is
    accepted per base whenever the requested size permits it.
    """
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    if background == "heldout":
        if heldout_bases is None or not len(heldout_bases):
            raise ValueError("heldout generation requires base sequences")
        if base_sequence_ids is None:
            base_sequence_ids = [f"{split}:{i}" for i in range(len(heldout_bases))]
        if len(base_sequence_ids) != len(heldout_bases):
            raise ValueError("base_sequence_ids and heldout_bases differ in length")
        accepted_per_base = np.zeros(len(heldout_bases), dtype=np.int32)
        eligible_order = np.empty(0, dtype=np.int64)
        eligible_cursor = 0
    elif background != "clean":
        raise ValueError("background must be 'clean' or 'heldout'")

    rows: list[dict[str, Any]] = []
    attempts = 0
    reasons: Counter[str] = Counter()
    attempted_key_counts: Counter[int] = Counter()
    accepted_key_counts: Counter[int] = Counter()
    attempted_ordered_runs: Counter[int] = Counter()
    accepted_ordered_runs: Counter[int] = Counter()

    while len(rows) < n_pairs:
        attempts += 1
        if attempts > max_factor * n_pairs:
            raise RuntimeError(
                f"{STRICT_FAMILY}/{background}/{split}: rejection rate too "
                f"high ({len(rows)}/{attempts})"
            )

        if background == "clean":
            base = None
            base_index = -1
            base_id = None
        else:
            # Prefer bases which do not yet have an accepted pair.  Rebuild a
            # randomized pass when the current pass is exhausted.  Reuse is
            # allowed only after every base has one accepted pair.
            if eligible_cursor >= len(eligible_order):
                unused = np.flatnonzero(accepted_per_base == 0)
                pool = unused if len(unused) else np.arange(len(heldout_bases))
                eligible_order = rng.permutation(pool)
                eligible_cursor = 0
            base_index = int(eligible_order[eligible_cursor])
            eligible_cursor += 1
            base = heldout_bases[base_index]
            base_id = base_sequence_ids[base_index]

        rec, reason, diagnostics = _sample_attempt(
            rng, background, base, base_id, base_index
        )
        attempted_key_counts[diagnostics["background_key_count"]] += 1
        attempted_ordered_runs[diagnostics["background_ordered_run_count"]] += 1
        if rec is None:
            reasons[str(reason)] += 1
            continue

        if background == "heldout":
            accepted_per_base[base_index] += 1
        accepted_key_counts[diagnostics["background_key_count"]] += 1
        accepted_ordered_runs[diagnostics["background_ordered_run_count"]] += 1
        rows.append(_row_from_record(rec, rng, split, len(rows)))

    # Randomize row order and pair-ID order independently.  IDs remain stable,
    # unique, deterministic under the pair seed, and non-informative.
    row_order = rng.permutation(n_pairs)
    id_order = rng.permutation(n_pairs)
    prefix = id_prefix or f"strict_{background}_{split}"
    shuffled: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(row_order):
        row = rows[int(source_index)]
        row["pair_id_order"] = int(id_order[output_index])
        row["pair_id"] = f"{prefix}_{int(id_order[output_index]):06d}"
        shuffled.append(row)

    frame = pd.DataFrame(shuffled)
    assert frame["pair_id"].is_unique, "pair IDs are not unique"

    if background == "heldout":
        used = accepted_per_base[accepted_per_base > 0]
        unique_bases = int(len(used))
        repeated_pairs = int(np.maximum(used - 1, 0).sum())
    else:
        unique_bases = n_pairs
        repeated_pairs = 0

    stats: dict[str, Any] = {
        "attempted": int(attempts),
        "accepted": int(n_pairs),
        "rejected": int(attempts - n_pairs),
        "rejection_rate": float(1.0 - n_pairs / attempts),
        "rejection_reasons": dict(sorted(reasons.items())),
        "background_key_count_distribution": {
            "attempted": _counter_dict(attempted_key_counts),
            "accepted": _counter_dict(accepted_key_counts),
        },
        "background_ordered_run_distribution": {
            "attempted": _counter_dict(attempted_ordered_runs),
            "accepted": _counter_dict(accepted_ordered_runs),
        },
        "unique_base_sequences": unique_bases,
        "repeated_base_pairs": repeated_pairs,
        "at_most_one_pair_per_base_when_possible": True,
        "candidate_storage_counts": {
            str(int(key)): int(value)
            for key, value in sorted(frame["positive_index"].value_counts().items())
        },
        "oracle_pair_accuracy": 1.0,
        "invariant_checks": int(n_pairs),
        "invariant_violations": 0,
        "exact_representation_ties": {
            "letter_counts": int(n_pairs),
            "aggregated_lag_pair_counts": int(n_pairs),
            "key_membership_mask": int(n_pairs),
            "chain_occupancy": int(n_pairs),
            "maximal_run_histogram": int(n_pairs),
            "per_chain_unordered_bags": int(n_pairs),
            "within_chain_depth_multisets": int(n_pairs),
            "increasing_decreasing_edge_counts": int(n_pairs),
        },
        "conditioning": (
            "Held-out backgrounds are conditioned on not independently "
            "making the negative candidate compliant."
            if background == "heldout"
            else "Clean backgrounds contain only non-key distractors."
        ),
    }
    return frame, stats


def _load_sequence_frame(path: Path, split: str) -> tuple[list[str], list[str]]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    sequence_col = next(
        (name for name in ("Sequences", "sequence", "X") if name in frame),
        None,
    )
    if sequence_col is None:
        raise ValueError(f"no sequence column found in {path}")
    id_col = next(
        (name for name in ("sequence_id", "base_sequence_id") if name in frame),
        None,
    )
    sequences = frame[sequence_col].astype(str).tolist()
    ids = (frame[id_col].astype(str).tolist() if id_col is not None
           else [f"{split}:{i}" for i in range(len(frame))])
    return sequences, ids


def build_strict_manifests(
    *,
    sizes: Mapping[str, int],
    pair_seed: int,
    val_sequences: Sequence[str],
    test_sequences: Sequence[str],
    val_sequence_ids: Sequence[str | int] | None = None,
    test_sequence_ids: Sequence[str | int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build all four strict manifests with independent deterministic streams."""
    t0 = time.time()
    frames = []
    datasets: dict[str, Any] = {}
    stream = 0
    for background in ("clean", "heldout"):
        for split in ("val", "test"):
            stream += 1
            rng = np.random.default_rng([pair_seed, stream])
            bases = val_sequences if split == "val" else test_sequences
            ids = val_sequence_ids if split == "val" else test_sequence_ids
            name = f"strict_{background}_{split}"
            frame, stats = generate_strict_family(
                background,
                split,
                int(sizes[split]),
                rng,
                heldout_bases=bases if background == "heldout" else None,
                base_sequence_ids=ids if background == "heldout" else None,
                id_prefix=name,
            )
            frames.append(frame)
            datasets[name] = stats
    manifest = pd.concat(frames, ignore_index=True)
    if not manifest["pair_id"].is_unique:
        raise AssertionError("pair IDs are not globally unique")
    report = {
        "family": STRICT_FAMILY,
        "mechanism": MECHANISM,
        "pair_seed": int(pair_seed),
        "sizes": {key: int(value) for key, value in sizes.items()},
        "datasets": datasets,
        "legacy_control": LEGACY_CONTROL_METADATA,
        "invariant_representations": [
            "full 26-dimensional letter counts",
            "full 26x26 aggregated directional lag-7 pair tensor",
            "binary key-membership mask",
            "per-chain key-count vector and sorted occupancy profile",
            "maximal-run-length histogram",
            "per-chain unordered letter multiset",
            "letter multiset at each within-chain depth",
            "counts of increasing and decreasing key-key edges",
        ],
        "generation_time_s": round(time.time() - t0, 3),
    }
    return manifest, report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pair_seed", type=int, default=PAIR_SEED)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--data_dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--data_tag", default=None)
    parser.add_argument("--val_file", type=Path, default=None)
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)

    defaults = SMOKE_SIZES if args.smoke else FULL_SIZES
    sizes = {
        "val": args.n_val if args.n_val is not None else defaults["val"],
        "test": args.n_test if args.n_test is not None else defaults["test"],
    }
    suffix = "_smoke" if args.smoke else ""
    data_tag = args.data_tag or f"ocdet{suffix}"
    if args.val_file is not None:
        val_path = args.val_file
    elif (args.data_dir / "splits" / "val.parquet").exists():
        val_path = args.data_dir / "splits" / "val.parquet"
    else:
        # Compatibility with the earlier two-condition matched-completion data.
        val_path = args.data_dir / f"X_val_{data_tag}.csv"
    if args.test_file is not None:
        test_path = args.test_file
    elif (args.data_dir / "splits" / "test.parquet").exists():
        test_path = args.data_dir / "splits" / "test.parquet"
    else:
        test_path = args.data_dir / f"X_test_{data_tag}.csv"
    val_sequences, val_ids = _load_sequence_frame(val_path, "val")
    test_sequences, test_ids = _load_sequence_frame(test_path, "test")

    manifest, report = build_strict_manifests(
        sizes=sizes,
        pair_seed=args.pair_seed,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        val_sequence_ids=val_ids,
        test_sequence_ids=test_ids,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem_suffix = "_smoke" if args.smoke else ""
    parquet_path = args.out_dir / f"strict_pair_manifest{stem_suffix}.parquet"
    report_path = args.out_dir / f"strict_pair_generation_report{stem_suffix}.json"
    manifest.to_parquet(parquet_path, index=False)
    report.update({
        "classification_data_tag": data_tag,
        "source_files": {"val": str(val_path), "test": str(test_path)},
        "manifest_path": str(parquet_path),
        "rows": int(len(manifest)),
        "command": " ".join(sys.argv),
    })
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"[strict_pairs] wrote {len(manifest)} pairs to {parquet_path} "
        f"in {report['generation_time_s']:.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
