"""Unit and invariant tests for the OC matched-completion pipeline.

Run from the repo root:
    PYTHONPATH=/root/LLMSeq/repro /root/LLMSeq/.venv/bin/python -m pytest \
        tests/test_oc_completion.py -q
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.analysis.mechanism_id.scripts.common import feat_count26, feat_lag_pair
from src.oc_completion.gen_pairs import (
    chain_positions,
    check_pair_invariants,
    generate_family,
    make_one_hole_pair,
    make_two_hole_pair,
)
from src.oc_completion.oracle import (
    ALPHABET,
    KAPPA,
    KEY_LETTERS,
    KEY_SET,
    LAG,
    N_EVENTS,
    NON_KEY_LETTERS,
    SEP,
    oc_label,
    oc_label_tokens,
)


# ---------------------------------------------------------------------------
# Oracle semantics
# ---------------------------------------------------------------------------
def brute_force_oc(toks) -> int:
    """Independent reference: enumerate maximal lag-7 runs directly.

    Test-only reference implementation (the production oracle lives solely in
    src.oc_completion.oracle).
    """
    n = len(toks)
    label = 0
    for t in range(n):
        if toks[t] not in KEY_SET:
            continue
        if t - LAG >= 0 and toks[t - LAG] in KEY_SET:
            continue  # not the start of a maximal run
        run = [t]
        while run[-1] + LAG < n and toks[run[-1] + LAG] in KEY_SET:
            run.append(run[-1] + LAG)
        if len(run) < 2:
            continue
        ks = [KAPPA[toks[p]] for p in run]
        if all(ks[i] <= ks[i + 1] for i in range(len(ks) - 1)):
            label = 1
    return label


def test_oracle_matches_brute_force_on_random_sequences():
    rng = np.random.default_rng(1234)
    letters = np.asarray(ALPHABET)
    # Uniform sequences are key-sparse; also generate key-rich sequences so
    # multi-chain and long-run cases are exercised.
    for key_boost in (0.0, 0.35, 0.7):
        probs = np.full(26, (1 - key_boost) / 26)
        for k in KEY_LETTERS:
            probs[ALPHABET.index(k)] += key_boost / 6
        probs /= probs.sum()
        idx = rng.choice(26, size=(4000, N_EVENTS), p=probs)
        for row in letters[idx]:
            toks = list(row)
            assert oc_label_tokens(toks) == brute_force_oc(toks)


def test_oracle_known_cases():
    base = ["A"] * N_EVENTS
    # ordered pair W (kappa 0) -> D (kappa 1) at lag 7
    s = base.copy()
    s[2], s[9] = "W", "D"
    assert oc_label_tokens(s) == 1
    # reversed pair D -> W is non-compliant
    s = base.copy()
    s[2], s[9] = "D", "W"
    assert oc_label_tokens(s) == 0
    # equal letters at lag 7 are compliant (kappa equal)
    s = base.copy()
    s[2], s[9] = "Q", "Q"
    assert oc_label_tokens(s) == 1
    # ordered pair at the wrong lag is non-compliant
    s = base.copy()
    s[2], s[8] = "W", "D"
    assert oc_label_tokens(s) == 0
    # ordered sub-pair inside a non-ordered MAXIMAL run does not count:
    # run W(kappa0) -> D(kappa1) -> W(kappa0) is maximal and not ordered.
    s = base.copy()
    s[1], s[8], s[15] = "W", "D", "W"
    assert oc_label_tokens(s) == 0
    # single key letter: no run
    s = base.copy()
    s[4] = "U"
    assert oc_label_tokens(s) == 0
    # string API
    assert oc_label(SEP.join(["A"] * 6 + ["W"] + ["A"] * 6 + ["U"] + ["A"] * 6)) == 1


def test_oracle_matches_stored_ocnoisy_labels_statistically():
    """Tag `_9` was generated with this rule + pi=0.3 flips: agreement with
    stored observed labels must be ~0.7."""
    import os

    import pandas as pd
    data_dir = os.environ.get("DATA_DIR",
                              "/root/LLMSeq/data") + "/simulation/tested"
    if not os.path.exists(f"{data_dir}/X_val_9.csv"):
        pytest.skip("legacy tag-9 dataset not present on this machine")
    X = pd.read_csv(f"{data_dir}/X_val_9.csv")["Sequences"]
    y = pd.read_csv(f"{data_dir}/y_val_9.csv")["Outcome"]
    n = 20_000
    pred = np.array([oc_label(s) for s in X.iloc[:n]])
    agree = float((pred == y.iloc[:n].values).mean())
    assert 0.68 < agree < 0.72, f"agreement {agree} incompatible with pi=0.3"


# ---------------------------------------------------------------------------
# Two-hole pattern algebra: both orientations, all parameter combinations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("orientation", [1, 2])
def test_two_hole_patterns_exhaustive_clean(orientation):
    from src.oc_completion.gen_pairs import _fill, _two_hole_patterns

    x = "A"
    background = ["B"] * N_EVENTS
    for r, s in itertools.permutations(range(6), 2):
        for i1, i2 in itertools.permutations(range(6), 2):
            if i1 >= i2:
                continue
            a, b = KEY_LETTERS[i1], KEY_LETTERS[i2]
            neg_r, neg_s, pos_r, pos_s = _two_hole_patterns(a, b, x, orientation)
            neg, pos = background.copy(), background.copy()
            _fill(neg, r, neg_r)
            _fill(neg, s, neg_s)
            _fill(pos, r, pos_r)
            _fill(pos, s, pos_s)
            assert oc_label_tokens(neg) == 0
            assert oc_label_tokens(pos) == 1
            assert np.array_equal(feat_count26(neg), feat_count26(pos))
            assert np.array_equal(feat_lag_pair(neg, LAG), feat_lag_pair(pos, LAG))


# ---------------------------------------------------------------------------
# Generator-level invariants (both families, both backgrounds)
# ---------------------------------------------------------------------------
def _fake_heldout_bases(n=500, seed=7):
    rng = np.random.default_rng(seed)
    letters = np.asarray(ALPHABET)
    return [SEP.join(letters[rng.integers(0, 26, size=N_EVENTS)]) for _ in range(n)]


@pytest.mark.parametrize("background", ["clean", "heldout"])
def test_two_hole_generator_invariants(background):
    rng = np.random.default_rng(42)
    bases = _fake_heldout_bases()
    df, stats = generate_family("two_hole", background, 300, rng, bases, "t")
    assert stats["accepted"] == 300
    assert df["pair_id"].is_unique
    # storage order actually randomized
    assert 0.3 < df["positive_index"].mean() < 0.7
    # both orientations present
    assert set(df["orientation"]) == {1, 2}
    for _, row in df.iterrows():
        pos = row[f"cand{row['positive_index']}"].split(SEP)
        neg = row[f"cand{1 - row['positive_index']}"].split(SEP)
        assert oc_label_tokens(pos) == 1
        assert oc_label_tokens(neg) == 0
        assert np.array_equal(feat_count26(neg), feat_count26(pos))
        assert np.array_equal(feat_lag_pair(neg, LAG), feat_lag_pair(pos, LAG))


@pytest.mark.parametrize("background", ["clean", "heldout"])
def test_one_hole_generator_invariants(background):
    rng = np.random.default_rng(43)
    bases = _fake_heldout_bases()
    df, stats = generate_family("one_hole", background, 200, rng, bases, "o")
    for _, row in df.iterrows():
        pos = row[f"cand{row['positive_index']}"].split(SEP)
        neg = row[f"cand{1 - row['positive_index']}"].split(SEP)
        assert oc_label_tokens(pos) == 1
        assert oc_label_tokens(neg) == 0


def test_clean_background_non_key_only():
    rng = np.random.default_rng(44)
    rec, reason = make_two_hole_pair(rng, "clean", None)
    assert rec is not None
    chain_pos = set(chain_positions(rec["chain_r"] - 1)) | set(
        chain_positions(rec["chain_s"] - 1))
    for p in range(N_EVENTS):
        if p not in chain_pos:
            assert rec["neg"][p] not in KEY_SET
            assert rec["neg"][p] == rec["pos"][p]


def test_heldout_background_retained():
    rng = np.random.default_rng(45)
    bases = _fake_heldout_bases()
    for _ in range(50):
        rec, reason = make_two_hole_pair(rng, "heldout", bases)
        if rec is None:
            continue
        base = bases[rec["base_index"]].split(SEP)
        chain_pos = set(chain_positions(rec["chain_r"] - 1)) | set(
            chain_positions(rec["chain_s"] - 1))
        for p in range(N_EVENTS):
            if p not in chain_pos:
                assert rec["neg"][p] == base[p]
                assert rec["pos"][p] == base[p]


def test_oracle_pair_accuracy_is_one_by_construction():
    """Acceptance criterion: the oracle itself has pair accuracy 1.0."""
    rng = np.random.default_rng(46)
    df, _ = generate_family("two_hole", "clean", 500, rng, None, "acc")
    correct = 0
    for _, row in df.iterrows():
        s0 = oc_label(row["cand0"])
        s1 = oc_label(row["cand1"])
        pred_pos = int(s1 > s0)
        correct += int(pred_pos == row["positive_index"])
    assert correct == 500
