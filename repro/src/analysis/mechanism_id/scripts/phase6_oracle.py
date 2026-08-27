"""Phase 6 — Oracle AUC / F1 audit.

Under symmetric label noise π and base rate ρ, the paper's Theorem (Eq. 3.3
style) gives AUC* = 0.5 + 0.5*(A - C) and F1* depending on ρ. These assume
the *proposal* law P0 (uniform i.i.d.). But the actual benchmark is a
*label-stratified subsample* of P0 (see Section 3.1 paragraph after Def.1).

For each dataset:
  - Recompute ρ on train/val/test.
  - Compute the empirical oracle: apply the recovered deterministic rule
    (best rule from Phase 1) on the retained distribution, giving oracle
    AUC and F1 directly.
  - Compare to the paper's reported ceilings.

This also flags cases where the paper-claimed F1 ceiling is inconsistent
with the retained distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    KAPPA_IMPL, KAPPA_PAPER, KEY_LETTERS_PAPER, KEY_SET_IMPL, RESULTS_DIR,
    load_split, rule_greedy_monotone_impl, rule_pair_any_lag,
    rule_parity_total_count, tokens,
)

KEY_SET_PAPER = frozenset(KEY_LETTERS_PAPER)


def eval_rule_on_dataset(tag: str, rule_fn, key_set=KEY_SET_IMPL, kappa=KAPPA_IMPL,
                         lag: int = 7, pi: float = 0.0,
                         rows_train: int = 0, rows_eval: int = 50_000):
    """Apply rule_fn (yields 0/1) to each split, report oracle AUC/F1."""
    out = {}
    for split in ("train", "val", "test"):
        rows = rows_train if split == "train" else rows_eval
        if rows <= 0 and split == "train":
            continue
        X, y = load_split(tag, split, rows=rows)
        y = y["Outcome"].values.astype(float)
        y_int = (y >= 0.5).astype(int)
        toks = [tokens(s) for s in X["Sequences"]]
        if lag is not None:
            preds = np.fromiter((rule_fn(t, lag, key_set, kappa)
                                 for t in toks), dtype=int, count=len(toks))
        else:
            preds = np.fromiter((rule_fn(t, key_set)
                                 for t in toks), dtype=int, count=len(toks))

        auc = float(roc_auc_score(y_int, preds)) if len(np.unique(y_int)) > 1 else float("nan")
        f1 = float(f1_score(y_int, preds)) if len(np.unique(y_int)) > 1 else float("nan")
        out[split] = {
            "rho_y": float(np.mean(y_int)),
            "rho_pred": float(np.mean(preds)),
            "AUC_oracle": auc, "F1_oracle": f1,
            "n": int(len(y_int)),
            "noise_acc": float(np.mean(preds == y_int)),  # 1 - empirical π
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows_eval", type=int, default=50_000)
    p.add_argument("--rows_train", type=int, default=0)
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "phase6_oracle.csv")
    args = p.parse_args()

    rows = []

    # Tricky Det. = "6" (claimed π=0),  Tricky Rnd. = "9" (claimed π=0.3)
    key_variants = [
        ("S_impl_WDQJXN", KEY_SET_IMPL, KAPPA_IMPL),
        ("S_paper_WDQJXU", KEY_SET_PAPER, KAPPA_PAPER),
    ]
    for tag, expected_pi in [("6", 0.0), ("9", 0.3)]:
        for rule_name, rule_fn, lag in [
            ("paper_pair_any_lag7", rule_pair_any_lag, 7),
            ("impl_greedy_tol0_min2",
             lambda t, L, K, kp: rule_greedy_monotone_impl(t, L, K, kp, tolerance=False,
                                                           min_chain_length=2), 7),
            ("impl_greedy_tol1_min2",
             lambda t, L, K, kp: rule_greedy_monotone_impl(t, L, K, kp, tolerance=True,
                                                           min_chain_length=2), 7),
        ]:
            for kname, kset, kkap in key_variants:
                res = eval_rule_on_dataset(tag, rule_fn, key_set=kset, kappa=kkap,
                                           lag=lag, rows_train=args.rows_train,
                                           rows_eval=args.rows_eval)
                for split, r in res.items():
                    rows.append({
                        "tag": tag, "rule": f"{rule_name}__{kname}", "split": split,
                        "expected_pi": expected_pi,
                        **r,
                    })

    # Parity = "test_just_pair"
    def parity_wrap(t, K):
        return rule_parity_total_count(t, K)

    res = eval_rule_on_dataset("test_just_pair",
                               lambda t, L, K, kp: parity_wrap(t, K),
                               lag=7, rows_train=args.rows_train,
                               rows_eval=args.rows_eval)
    for split, r in res.items():
        rows.append({
            "tag": "test_just_pair", "rule": "parity_total_count",
            "split": split, "expected_pi": 0.0, **r,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[phase6] wrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
