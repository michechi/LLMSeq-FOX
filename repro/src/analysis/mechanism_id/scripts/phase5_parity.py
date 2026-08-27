"""Phase 5 — Parity decomposition.

For the current Parity regime (ℓ=26, m=6, n=20):
  (A) Hidden-K — raw letter input, label = parity-of-count rule
  (B) Known-K membership sequence — input = per-position 0/1 b_t = 1[X_t in K]
  (C) Known-K count vector — input = per-key integer counts (6-dim)
  (D) Total-count parity — input = scalar total key count

All trained with LogisticRegression and MLP (hidden=64), reporting AUC.

Also sweep hidden-K parity across alphabet sizes ℓ and subset sizes m to
show how difficulty scales in the balanced (|K|/|A|=1/2) vs. paper
(m=6, ℓ=26, p≈0.23) regimes.
"""
from __future__ import annotations

import argparse
import string
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    KEY_LETTERS_IMPL, RESULTS_DIR, load_split, tokens,
)

SEED = 20260421


def _labels(df) -> np.ndarray:
    y = df["Outcome"].values.astype(float)
    return (y >= 0.5).astype(int)


def load_real_parity(rows_train: int = 400_000, rows_eval: int = 50_000):
    splits = {}
    for split, rows in [("train", rows_train), ("val", rows_eval), ("test", rows_eval)]:
        X, y = load_split("test_just_pair", split, rows=rows)
        splits[split] = (X, y)
    return splits


def _feat_letter_onehot_flat(toks, n: int, alphabet_size: int = 26):
    """Hidden-K regime A: one-hot position * letter, flattened.
    dim = n * alphabet_size."""
    arr = np.zeros((n, alphabet_size), dtype=np.float32)
    for i, t in enumerate(toks):
        if i >= n or len(t) != 1:
            continue
        a = ord(t) - ord("A")
        if 0 <= a < alphabet_size:
            arr[i, a] = 1.0
    return arr.reshape(-1)


def _feat_count26(toks, alphabet_size: int = 26):
    c = np.zeros(alphabet_size, dtype=np.float32)
    for t in toks:
        if len(t) == 1:
            a = ord(t) - ord("A")
            if 0 <= a < alphabet_size:
                c[a] += 1
    return c


def _feat_key_count(toks, keys) -> np.ndarray:
    idx = {k: i for i, k in enumerate(keys)}
    c = np.zeros(len(keys), dtype=np.float32)
    for t in toks:
        if t in idx:
            c[idx[t]] += 1
    return c


def _feat_key_count_parity(toks, keys) -> np.ndarray:
    """Per-key parity bit (0/1)."""
    c = _feat_key_count(toks, keys)
    return (c % 2).astype(np.float32)


def _feat_membership_bits(toks, keys, n: int) -> np.ndarray:
    key_set = frozenset(keys)
    arr = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(toks):
        if i >= n:
            break
        if t in key_set:
            arr[i] = 1.0
    return arr


def _feat_total_key_count(toks, keys) -> np.ndarray:
    key_set = frozenset(keys)
    return np.array([sum(1 for t in toks if t in key_set)], dtype=np.float32)


def _eval(model, Xtr, ytr, Xv, yv, Xte, yte):
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        pv = model.predict_proba(Xv)[:, 1]
        pt = model.predict_proba(Xte)[:, 1]
    else:
        pv = model.decision_function(Xv)
        pt = model.decision_function(Xte)
    if len(np.unique(yte)) >= 2:
        auc = float(roc_auc_score(yte, pt))
    else:
        auc = float("nan")
    thrs = np.linspace(pt.min() - 1e-6, pt.max() + 1e-6, 201)
    if len(np.unique(yv)) >= 2:
        scores = [f1_score(yv, pv >= t) for t in thrs]
        bt = thrs[int(np.argmax(scores))]
        f1 = float(f1_score(yte, pt >= bt))
    else:
        f1 = float("nan")
    return auc, f1


def run_paper_regime():
    print("[phase5] loading real parity dataset (test_just_pair)")
    data = load_real_parity()
    ytr = _labels(data["train"][1])
    yv = _labels(data["val"][1])
    yte = _labels(data["test"][1])

    tok_tr = [tokens(s) for s in data["train"][0]["Sequences"]]
    tok_v = [tokens(s) for s in data["val"][0]["Sequences"]]
    tok_te = [tokens(s) for s in data["test"][0]["Sequences"]]

    variants = {
        "A_hidden_letters_onehot": (
            lambda t: _feat_letter_onehot_flat(t, 20, 26)),
        "A_hidden_count26":         (lambda t: _feat_count26(t, 26)),
        "B_known_membership_bits":  (lambda t: _feat_membership_bits(t, KEY_LETTERS_IMPL, 20)),
        "C_known_key_count":        (lambda t: _feat_key_count(t, KEY_LETTERS_IMPL)),
        "C_known_key_count_parity": (lambda t: _feat_key_count_parity(t, KEY_LETTERS_IMPL)),
        "D_total_key_count":        (lambda t: _feat_total_key_count(t, KEY_LETTERS_IMPL)),
    }

    rows = []
    for vname, ex in variants.items():
        t0 = time.time()
        Xtr = np.stack([ex(t) for t in tok_tr], 0)
        Xv = np.stack([ex(t) for t in tok_v], 0)
        Xte = np.stack([ex(t) for t in tok_te], 0)
        dt = time.time() - t0
        print(f"  [{vname}] dim={Xtr.shape[1]} build_t={dt:.1f}s", flush=True)

        for mname, model in [
            ("logreg", LogisticRegression(max_iter=2000, solver="lbfgs",
                                          random_state=SEED, n_jobs=-1)),
            ("mlp_64", MLPClassifier(hidden_layer_sizes=(64,), random_state=SEED,
                                     max_iter=200, early_stopping=True)),
        ]:
            t0 = time.time()
            auc, f1 = _eval(model, Xtr, ytr, Xv, yv, Xte, yte)
            dt = time.time() - t0
            rows.append({
                "regime": "paper_m6_l26_n20",
                "variant": vname, "model": mname,
                "feat_dim": int(Xtr.shape[1]),
                "AUC": auc, "F1": f1, "train_sec": dt,
                "rho_test": float(np.mean(yte)),
            })
            print(f"    {mname}: AUC={auc} F1={f1} dt={dt:.1f}s", flush=True)
    return rows


def run_sweep(ell_list=(4, 8, 12, 16, 26), m_ratios=(0.5,), n_list=(20,),
              n_rows=50_000):
    """Synthesize balanced (p=|K|/|A|=1/2) and skewed sweeps."""
    alph_all = list(string.ascii_uppercase)
    rng_master = np.random.default_rng(SEED)
    rows = []
    for ell in ell_list:
        alph = alph_all[:ell]
        for r in m_ratios:
            m = max(1, int(r * ell))
            K = tuple(alph[:m])
            key_set = frozenset(K)
            for n in n_list:
                rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
                idx = rng.integers(0, ell, size=(n_rows, n))
                arr = np.array(alph)[idx]
                seqs = [list(row) for row in arr]
                # Paper-rule parity: even #keys with even count  <=>  total count parity iff m even.
                tot = np.array([sum(1 for t in s if t in key_set) for s in seqs])
                y = (tot % 2 == 0).astype(int)

                # Train on hidden-letter count26 (A), membership (B), total-count (D).
                def _onehot(s):
                    a = np.zeros((n, ell), dtype=np.float32)
                    for i, t in enumerate(s):
                        a[i, alph.index(t)] = 1.0
                    return a.reshape(-1)

                def _count(s):
                    a = np.zeros(ell, dtype=np.float32)
                    for t in s:
                        a[alph.index(t)] += 1
                    return a

                def _bits(s):
                    return np.array([1.0 if t in key_set else 0.0 for t in s],
                                    dtype=np.float32)

                variants = {
                    "A_hidden_onehot": _onehot,
                    "A_hidden_count":  _count,
                    "B_known_membership_bits": _bits,
                }

                split = n_rows // 2
                split2 = (3 * n_rows) // 4
                for vname, ex in variants.items():
                    X_all = np.stack([ex(s) for s in seqs], 0)
                    Xtr, ytr = X_all[:split], y[:split]
                    Xv, yv = X_all[split:split2], y[split:split2]
                    Xte, yte = X_all[split2:], y[split2:]
                    model = LogisticRegression(max_iter=2000, solver="lbfgs",
                                               random_state=SEED, n_jobs=-1)
                    try:
                        auc, f1 = _eval(model, Xtr, ytr, Xv, yv, Xte, yte)
                    except Exception as e:
                        auc, f1 = float("nan"), float("nan")
                        print(f"  [sweep ell={ell} m={m} n={n} {vname}] ERROR: {e}")
                    rows.append({
                        "regime": f"sweep_ell{ell}_m{m}_n{n}",
                        "ell": ell, "m": m, "n": n, "p": m / ell,
                        "variant": vname, "model": "logreg",
                        "feat_dim": int(X_all.shape[1]),
                        "AUC": auc, "F1": f1, "rho_test": float(np.mean(yte)),
                    })
                    print(f"  sweep ell={ell} m={m} n={n} {vname}: AUC={auc}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "phase5_parity.csv")
    p.add_argument("--skip_paper", action="store_true")
    p.add_argument("--skip_sweep", action="store_true")
    p.add_argument("--sweep_rows", type=int, default=50_000)
    args = p.parse_args()

    rows = []
    if not args.skip_paper:
        rows.extend(run_paper_regime())
        pd.DataFrame(rows).to_csv(args.out, index=False)
    if not args.skip_sweep:
        rows.extend(run_sweep(n_rows=args.sweep_rows))
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[phase5] wrote {args.out}")


if __name__ == "__main__":
    main()
