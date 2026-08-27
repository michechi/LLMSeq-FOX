"""Phase 1b — infer the Tricky label rule from LogReg weights on 26×26
aggregated lag-pair features. LogReg AUC is 0.997 on `_6` so the label is
essentially a linear function of these counts; the weights reveal which
(a, b) pairs at spacing λ are "evidence for Y=1".

Outputs:
  - phase1b_logreg_weights.csv  — all 676 (a, b) coefficients per tag
  - phase1b_top_pairs.csv       — top-30 positive coefficients per tag
  - phase1b_rule_check.txt      — human-readable summary of recovered S, κ
"""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import ALPHABET, RESULTS_DIR, feat_lag_pair, load_split, tokens


def fit_logreg(tag, rows=200_000, lag=7, C=1.0):
    Xtr, ytr = load_split(tag, "train", rows=rows)
    Xte, yte = load_split(tag, "test", rows=rows // 4)

    Ftr = np.stack([feat_lag_pair(tokens(s), lag) for s in Xtr.Sequences], 0)
    Fte = np.stack([feat_lag_pair(tokens(s), lag) for s in Xte.Sequences], 0)

    ytr = (ytr["Outcome"].values >= 0.5).astype(int)
    yte = (yte["Outcome"].values >= 0.5).astype(int)

    clf = LogisticRegression(C=C, max_iter=3000, solver="lbfgs", n_jobs=-1,
                             random_state=42)
    clf.fit(Ftr, ytr)
    p_te = clf.predict_proba(Fte)[:, 1]
    auc = roc_auc_score(yte, p_te)
    return clf.coef_.reshape(26, 26), clf.intercept_[0], auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["6", "9"])
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--top_k", type=int, default=30)
    args = ap.parse_args()

    alpha = list(string.ascii_uppercase)
    weight_rows = []
    top_rows = []
    txt_lines = []

    for tag in args.tags:
        W, b, auc = fit_logreg(tag, rows=args.rows)
        print(f"[phase1b] tag={tag}  test AUC={auc:.4f}  intercept={b:.3f}", flush=True)
        txt_lines.append(f"tag={tag}  test AUC={auc:.4f}  intercept={b:.3f}")

        for i, a in enumerate(alpha):
            for j, bl in enumerate(alpha):
                weight_rows.append({"tag": tag, "a": a, "b": bl, "w": float(W[i, j])})

        flat = [(float(W[i, j]), alpha[i], alpha[j])
                for i in range(26) for j in range(26)]
        flat.sort(reverse=True)
        txt_lines.append(f"\n=== tag={tag}: top-{args.top_k} positive (a,b)->Y=1 pairs ===")
        for rank, (w, a, bl) in enumerate(flat[:args.top_k]):
            top_rows.append({"tag": tag, "rank": rank, "a": a, "b": bl, "w": w,
                             "sign": "pos"})
            txt_lines.append(f"{rank:3d}. {a}->{bl}  w={w:+.3f}")
        txt_lines.append(f"\n=== tag={tag}: top-{args.top_k} negative (a,b)->Y=0 pairs ===")
        for rank, (w, a, bl) in enumerate(flat[-args.top_k:][::-1]):
            top_rows.append({"tag": tag, "rank": rank, "a": a, "b": bl, "w": w,
                             "sign": "neg"})
            txt_lines.append(f"{rank:3d}. {a}->{bl}  w={w:+.3f}")

        # Recover S as the union of letters in top-k positive pairs and their
        # relative frequency; rank letters by their marginal weight.
        marg_a = W.sum(axis=1)  # effect of letter as "first" of pair
        marg_b = W.sum(axis=0)  # effect of letter as "second" of pair
        letters_by_marginal = sorted(
            [(float(marg_a[i] + marg_b[i]), alpha[i]) for i in range(26)], reverse=True,
        )
        txt_lines.append(f"\n=== tag={tag}: letters ranked by (marginal_a + marginal_b) ===")
        for w, a in letters_by_marginal[:10]:
            txt_lines.append(f"  {a}: {w:+.3f}")

    pd.DataFrame(weight_rows).to_csv(RESULTS_DIR / "phase1b_logreg_weights.csv", index=False)
    pd.DataFrame(top_rows).to_csv(RESULTS_DIR / "phase1b_top_pairs.csv", index=False)
    (RESULTS_DIR / "phase1b_rule_check.txt").write_text("\n".join(txt_lines))
    print(f"[phase1b] wrote {RESULTS_DIR / 'phase1b_logreg_weights.csv'}")
    print(f"[phase1b] wrote {RESULTS_DIR / 'phase1b_top_pairs.csv'}")
    print(f"[phase1b] wrote {RESULTS_DIR / 'phase1b_rule_check.txt'}")


if __name__ == "__main__":
    main()
