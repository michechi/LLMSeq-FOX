"""Generate matched completion pairs for the Ordered-Compliance analysis.

Two pair families, each under two background conditions:

* one-hole control (easy): a single position differs between candidates;
  one completion creates an ordered lag-7 key run, the other leaves the
  sequence non-compliant. Shortcut features (letter counts, lag-pair counts)
  are allowed to differ.

* matched two-hole test (main): two length-three residue chains
  (r, r+7, r+14) and (s, s+7, s+14), r != s in {1..6} (1-indexed), carry the
  patterns below with key letters a, b (kappa(a) < kappa(b)) and a non-key
  distractor x. The two candidates have exactly equal 26-dim letter-count
  vectors and exactly equal aggregated lag-7 pair-count tensors, but opposite
  latent labels.

      orientation 1:  negative  chain r: a b a   chain s: x b x
                      positive  chain r: a b x   chain s: x b a
      orientation 2:  negative  chain r: x a x   chain s: b a b
                      positive  chain r: x a b   chain s: b a x

Backgrounds:
* clean:    every non-chain position filled with uniform non-key letters
            (same background for both candidates of a pair);
* heldout:  base sequence drawn from the original OC test split; only the
            chain positions are overwritten.

Every accepted pair is verified with the canonical oracle
(Y*(negative)=0, Y*(positive)=1) plus the full invariant checklist.

Usage (from repro/ root):
    DATA_DIR=/root/LLMSeq/data python -m src.oc_completion.gen_pairs \
        [--smoke] [--pair_seed 9600] [--out_dir ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

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
DATA_ROOT = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data")) / "simulation" / "oc_completion"

# 0-indexed starts of the six length-three lag-7 residue chains in a
# 20-position sequence: starts 0..5 -> positions (s, s+7, s+14).
CHAIN_STARTS = tuple(range(6))


def chain_positions(start: int) -> tuple[int, int, int]:
    return (start, start + LAG, start + 2 * LAG)


def _two_hole_patterns(a: str, b: str, x: str, orientation: int):
    """Return (neg_r, neg_s, pos_r, pos_s) letter triples for the two chains."""
    if orientation == 1:
        return (a, b, a), (x, b, x), (a, b, x), (x, b, a)
    if orientation == 2:
        return (x, a, x), (b, a, b), (x, a, b), (b, a, x)
    raise ValueError(f"orientation must be 1 or 2, got {orientation}")


def _fill(base: list[str], start: int, letters: tuple[str, str, str]) -> None:
    for p, letter in zip(chain_positions(start), letters):
        base[p] = letter


def make_two_hole_pair(rng: np.random.Generator, background: str,
                       heldout_bases: list[str] | None):
    """One attempt at a matched two-hole pair. Returns (record, reason).

    record is None when the attempt is rejected; reason explains why.
    """
    r, s = rng.choice(CHAIN_STARTS, size=2, replace=False)
    r, s = int(r), int(s)
    k1, k2 = rng.choice(len(KEY_LETTERS), size=2, replace=False)
    a, b = KEY_LETTERS[min(k1, k2)], KEY_LETTERS[max(k1, k2)]  # kappa(a)<kappa(b)
    x = NON_KEY_LETTERS[int(rng.integers(len(NON_KEY_LETTERS)))]
    orientation = int(rng.integers(1, 3))

    if background == "clean":
        base = [NON_KEY_LETTERS[int(i)]
                for i in rng.integers(len(NON_KEY_LETTERS), size=N_EVENTS)]
        base_index = -1
    elif background == "heldout":
        base_index = int(rng.integers(len(heldout_bases)))
        base = heldout_bases[base_index].split(SEP)
    else:
        raise ValueError(background)

    neg_r, neg_s, pos_r, pos_s = _two_hole_patterns(a, b, x, orientation)
    neg = list(base)
    _fill(neg, r, neg_r)
    _fill(neg, s, neg_s)
    pos = list(base)
    _fill(pos, r, pos_r)
    _fill(pos, s, pos_s)

    y_neg, y_pos = oc_label_tokens(neg), oc_label_tokens(pos)
    if y_neg != 0:
        return None, "negative_compliant_background"
    if y_pos != 1:
        return None, "positive_noncompliant"

    return {
        "chain_r": r + 1,          # report 1-indexed as in the spec
        "chain_s": s + 1,
        "key_a": a,
        "key_b": b,
        "distractor_x": x,
        "kappa_gap": KAPPA[b] - KAPPA[a],
        "orientation": orientation,
        "base_index": base_index,
        "neg": neg,
        "pos": pos,
    }, None


def make_one_hole_pair(rng: np.random.Generator, background: str,
                       heldout_bases: list[str] | None):
    """One attempt at a one-hole control pair."""
    c = int(rng.choice(CHAIN_STARTS))
    k1, k2 = rng.choice(len(KEY_LETTERS), size=2, replace=False)
    a, b = KEY_LETTERS[min(k1, k2)], KEY_LETTERS[max(k1, k2)]
    x = NON_KEY_LETTERS[int(rng.integers(len(NON_KEY_LETTERS)))]
    z = NON_KEY_LETTERS[int(rng.integers(len(NON_KEY_LETTERS)))]  # negative filler

    if background == "clean":
        base = [NON_KEY_LETTERS[int(i)]
                for i in rng.integers(len(NON_KEY_LETTERS), size=N_EVENTS)]
        base_index = -1
    else:
        base_index = int(rng.integers(len(heldout_bases)))
        base = heldout_bases[base_index].split(SEP)

    # chain c: (a, HOLE, x). Positive completion: HOLE=b (ordered run a->b).
    # Negative completion: HOLE=z (non-key) leaves a single key -> no run.
    pos = list(base)
    _fill(pos, c, (a, b, x))
    neg = list(base)
    _fill(neg, c, (a, z, x))

    y_neg, y_pos = oc_label_tokens(neg), oc_label_tokens(pos)
    if y_neg != 0:
        return None, "negative_compliant_background"
    if y_pos != 1:
        return None, "positive_noncompliant"

    return {
        "chain_r": c + 1,
        "chain_s": -1,
        "key_a": a,
        "key_b": b,
        "distractor_x": x,
        "kappa_gap": KAPPA[b] - KAPPA[a],
        "orientation": 1,
        "base_index": base_index,
        "neg": neg,
        "pos": pos,
    }, None


def check_pair_invariants(rec: dict, family: str) -> None:
    neg, pos = rec["neg"], rec["pos"]
    assert len(neg) == N_EVENTS and len(pos) == N_EVENTS, "length != 20"
    assert oc_label_tokens(pos) == 1, "positive Y* != 1"
    assert oc_label_tokens(neg) == 0, "negative Y* != 0"
    assert rec["key_a"] in KEY_SET and rec["key_b"] in KEY_SET, "a or b not in S"
    assert KAPPA[rec["key_a"]] < KAPPA[rec["key_b"]], "kappa(a) >= kappa(b)"
    assert rec["distractor_x"] not in KEY_SET, "x in S"
    if family == "two_hole":
        assert rec["chain_r"] != rec["chain_s"], "chains identical"
        assert 1 <= rec["chain_r"] <= 6 and 1 <= rec["chain_s"] <= 6, "invalid chain"
        assert np.array_equal(feat_count26(neg), feat_count26(pos)), \
            "letter counts differ"
        assert np.array_equal(feat_lag_pair(neg, LAG), feat_lag_pair(pos, LAG)), \
            "lag-7 pair tensors differ"
    else:
        assert 1 <= rec["chain_r"] <= 6, "invalid chain"


def generate_family(family: str, background: str, n_pairs: int,
                    rng: np.random.Generator, heldout_bases: list[str] | None,
                    id_prefix: str, max_factor: int = 30):
    make = make_two_hole_pair if family == "two_hole" else make_one_hole_pair
    rows, attempts, reasons = [], 0, Counter()
    while len(rows) < n_pairs:
        attempts += 1
        if attempts > max_factor * n_pairs:
            raise RuntimeError(
                f"{family}/{background}: rejection rate too high "
                f"({len(rows)}/{attempts})")
        rec, reason = make(rng, background, heldout_bases)
        if rec is None:
            reasons[reason] += 1
            continue
        check_pair_invariants(rec, family)
        # randomized storage order
        pos_idx = int(rng.integers(2))
        cands = [None, None]
        cands[pos_idx] = SEP.join(rec["pos"])
        cands[1 - pos_idx] = SEP.join(rec["neg"])
        rows.append({
            "pair_id": f"{id_prefix}_{len(rows):06d}",
            "family": family,
            "background": background,
            "orientation": rec["orientation"],
            "chain_r": rec["chain_r"],
            "chain_s": rec["chain_s"],
            "key_a": rec["key_a"],
            "key_b": rec["key_b"],
            "distractor_x": rec["distractor_x"],
            "kappa_gap": rec["kappa_gap"],
            "base_index": rec["base_index"],
            "positive_index": pos_idx,
            "cand0": cands[0],
            "cand1": cands[1],
        })
    df = pd.DataFrame(rows)
    assert df["pair_id"].is_unique, "pair ids not unique"
    stats = {
        "attempted": attempts,
        "accepted": len(rows),
        "rejection_rate": round(1 - len(rows) / attempts, 4),
        "rejection_reasons": dict(reasons),
    }
    return df, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pair_seed", type=int, default=9600)
    ap.add_argument("--n_val", type=int, default=2_000)
    ap.add_argument("--n_test", type=int, default=10_000)
    ap.add_argument("--data_dir", type=Path, default=DATA_ROOT,
                    help="dir with the regenerated OC splits")
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--data_tag", type=str, default=None,
                    help="tag of the classification datasets (default: ocdet"
                         " or ocdet_smoke)")
    args = ap.parse_args()

    if args.smoke:
        args.n_val, args.n_test = 200, 500
    suffix = "_smoke" if args.smoke else ""
    data_tag = args.data_tag or f"ocdet{suffix}"
    out_dir = args.out_dir or (args.data_dir / f"pairs{suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bases for the held-out background: the original OC test split (the X
    # sequences are shared between ocdet and ocnoisy by construction).
    test_X = pd.read_csv(args.data_dir / f"X_test_{data_tag}.csv")["Sequences"].tolist()
    train_X = set(
        pd.read_csv(args.data_dir / f"X_train_{data_tag}.csv")["Sequences"].tolist()
    )

    t0 = time.time()
    report = {
        "mechanism": MECHANISM,
        "pair_seed": args.pair_seed,
        "sizes": {"val": args.n_val, "test": args.n_test},
        "classification_data_tag": data_tag,
        "datasets": {},
        "command": " ".join(sys.argv),
    }
    stream = 0
    for family in ("two_hole", "one_hole"):
        for background in ("clean", "heldout"):
            for split, n_pairs in (("val", args.n_val), ("test", args.n_test)):
                stream += 1
                rng = np.random.default_rng([args.pair_seed, stream])
                name = f"{family}_{background}_{split}"
                df, stats = generate_family(
                    family, background, n_pairs, rng,
                    test_X, id_prefix=name)
                # exact-sequence overlap with the original training split
                cands = pd.concat([df["cand0"], df["cand1"]])
                overlap = int(cands.isin(train_X).sum())
                stats["train_overlap_sequences"] = overlap
                assert overlap == 0, f"{name}: completed sequence found in train"
                path = out_dir / f"pairs_{name}{suffix}.csv"
                df.to_csv(path, index=False)
                report["datasets"][name] = {
                    **stats, "path": str(path),
                }
                print(f"[pairs] {name}: accepted={stats['accepted']} "
                      f"attempted={stats['attempted']} "
                      f"rej={stats['rejection_rate']}", flush=True)

    report["generation_time_s"] = round(time.time() - t0, 1)
    with open(out_dir / f"pair_generation_report{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[pairs] done in {report['generation_time_s']}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
