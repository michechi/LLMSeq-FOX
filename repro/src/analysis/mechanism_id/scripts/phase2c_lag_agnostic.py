"""Phase 2c — lag-agnostic baseline ladder (fair comparison).

The Phase-2 C1 baseline feeds the model the true λ=7 by construction.
This script instead gives the model a feature representation that
*could* discover λ from data:

  stacked_all_lag: G_{λ,a,b}(X) = Σ_t 1[X_t=a, X_{t+λ}=b] for
                   every λ ∈ {1,…,n-1}, a,b ∈ Σ.  Dim = (n-1)×26×26.
  max_over_lag:    M_{a,b}(X)   = max_λ G_{λ,a,b}(X).  Dim = 676.
  sum_over_lag:    S_{a,b}(X)   = Σ_λ G_{λ,a,b}(X).   Dim = 676.

Models: L1 LogReg (to drive irrelevant λ-blocks to zero), L2 LogReg,
XGBoost. Primary metric: AUC. Also records the per-λ feature weight
norm for L1 — so we can see directly whether the linear model found
λ=7 on its own.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import ALPHABET, RESULTS_DIR, load_split, tokens

SEED = 20260421
N_DEFAULT = 20


def feat_stacked_all_lag(toks, n: int) -> np.ndarray:
    """Dim = (n-1) * 26 * 26."""
    feats = np.zeros((n - 1, 26, 26), dtype=np.float32)
    L = len(toks)
    for lag in range(1, n):
        for t in range(L - lag):
            a, b = toks[t], toks[t + lag]
            if len(a) == 1 and len(b) == 1:
                ai = ord(a) - ord("A")
                bi = ord(b) - ord("A")
                if 0 <= ai < 26 and 0 <= bi < 26:
                    feats[lag - 1, ai, bi] += 1
    return feats.reshape(-1)


def feat_max_over_lag(toks, n: int) -> np.ndarray:
    """Dim = 26 * 26."""
    feats = np.zeros((n - 1, 26, 26), dtype=np.float32)
    L = len(toks)
    for lag in range(1, n):
        for t in range(L - lag):
            a, b = toks[t], toks[t + lag]
            if len(a) == 1 and len(b) == 1:
                ai = ord(a) - ord("A")
                bi = ord(b) - ord("A")
                if 0 <= ai < 26 and 0 <= bi < 26:
                    feats[lag - 1, ai, bi] += 1
    return feats.max(axis=0).reshape(-1)


def feat_sum_over_lag(toks, n: int) -> np.ndarray:
    """Dim = 26 * 26.  Pooling across all lags (contiguous-bigram-like
    but here any lag, not just λ=1)."""
    feats = np.zeros((26, 26), dtype=np.float32)
    L = len(toks)
    for lag in range(1, n):
        for t in range(L - lag):
            a, b = toks[t], toks[t + lag]
            if len(a) == 1 and len(b) == 1:
                ai = ord(a) - ord("A")
                bi = ord(b) - ord("A")
                if 0 <= ai < 26 and 0 <= bi < 26:
                    feats[ai, bi] += 1
    return feats.reshape(-1)


def _label(df):
    return (df["Outcome"].values.astype(float) >= 0.5).astype(int)


def _stack(data, split, ex):
    return np.stack([ex(tokens(s)) for s in data[split][0]["Sequences"]], axis=0)


def _scores(yv, pv, yte, pte):
    if len(np.unique(yte)) < 2:
        return float("nan"), float("nan"), float("nan")
    auc = float(roc_auc_score(yte, pte))
    thrs = np.linspace(0, 1, 201)
    bt = thrs[np.argmax([f1_score(yv, pv >= t) for t in thrs])]
    f1 = float(f1_score(yte, pte >= bt))
    return auc, f1, float(bt)


def fit_logreg(Xtr, ytr, Xv, yv, Xte, yte, penalty: str, C: float):
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        penalty=penalty, C=C, max_iter=3000, solver=solver,
        random_state=SEED, n_jobs=-1 if solver == "lbfgs" else None,
    )
    clf.fit(Xtr, ytr)
    pv = clf.predict_proba(Xv)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    return (*_scores(yv, pv, yte, pte), clf)


def fit_xgb(Xtr, ytr, Xv, yv, Xte, yte):
    from xgboost import XGBClassifier
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, tree_method="hist", n_jobs=-1,
        random_state=SEED, eval_metric="logloss",
    )
    clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    pv = clf.predict_proba(Xv)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    return (*_scores(yv, pv, yte, pte), clf)


def per_lag_weight_norm(coef: np.ndarray, n: int) -> np.ndarray:
    """Given a 12844-dim coefficient vector, sum |w| within each
    (n-1) block of 676 so we can see which lag the model used."""
    mat = coef.reshape(n - 1, 26 * 26)
    return np.abs(mat).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["6", "9"])
    ap.add_argument("--rows_train", type=int, default=100_000)
    ap.add_argument("--rows_eval", type=int, default=50_000)
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    ap.add_argument("--families", nargs="+",
                    default=["stacked_all_lag", "max_over_lag", "sum_over_lag"])
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "phase2c_lag_agnostic.csv")
    ap.add_argument("--weights_out", type=Path,
                    default=RESULTS_DIR / "phase2c_per_lag_weights.csv")
    args = ap.parse_args()

    all_rows = []
    weight_rows = []

    for tag in args.tags:
        print(f"[phase2c] loading tag={tag}", flush=True)
        data = {s: load_split(tag, s, rows=args.rows_train if s == "train"
                               else args.rows_eval)
                for s in ("train", "val", "test")}
        ytr, yv, yte = (_label(data[s][1]) for s in ("train", "val", "test"))

        for fam in args.families:
            t0 = time.time()
            if fam == "stacked_all_lag":
                ex = lambda t: feat_stacked_all_lag(t, args.n)  # noqa: E731
            elif fam == "max_over_lag":
                ex = lambda t: feat_max_over_lag(t, args.n)  # noqa: E731
            elif fam == "sum_over_lag":
                ex = lambda t: feat_sum_over_lag(t, args.n)  # noqa: E731
            else:
                raise ValueError(fam)

            Xtr = _stack(data, "train", ex)
            Xv = _stack(data, "val", ex)
            Xte = _stack(data, "test", ex)
            dt = time.time() - t0
            print(f"  [{tag}/{fam}] dim={Xtr.shape[1]} build_t={dt:.1f}s",
                  flush=True)

            for model_name, fit_fn, kwargs in [
                ("logreg_L2", fit_logreg, dict(penalty="l2", C=1.0)),
                ("logreg_L1", fit_logreg, dict(penalty="l1", C=0.5)),
                ("xgboost",   fit_xgb,    dict()),
            ]:
                t0 = time.time()
                try:
                    auc, f1, thr, clf = fit_fn(Xtr, ytr, Xv, yv, Xte, yte, **kwargs)
                except Exception as e:
                    print(f"    [{model_name}] ERROR: {e}", flush=True)
                    continue
                dt = time.time() - t0

                row = {"tag": tag, "family": fam, "model": model_name,
                       "feat_dim": int(Xtr.shape[1]),
                       "n_train": int(Xtr.shape[0]),
                       "AUC": auc, "F1": f1, "threshold": thr,
                       "train_sec": dt}
                all_rows.append(row)
                print(f"    [{model_name}] AUC={auc:.4f} F1={f1:.4f} dt={dt:.1f}s",
                      flush=True)

                # Record per-lag weight norm for stacked family + linear models
                if fam == "stacked_all_lag" and model_name in ("logreg_L1", "logreg_L2"):
                    coef = clf.coef_.reshape(-1)
                    per_lag = per_lag_weight_norm(coef, args.n)
                    top_lag = int(np.argmax(per_lag)) + 1
                    print(f"      per-λ |w| : {[f'{v:.2f}' for v in per_lag]}",
                          flush=True)
                    print(f"      top-weight λ = {top_lag}", flush=True)
                    for lag, w in enumerate(per_lag, 1):
                        weight_rows.append({
                            "tag": tag, "model": model_name,
                            "lag": lag, "weight_l1_norm": float(w),
                            "is_top_lag": (lag == top_lag),
                        })

            pd.DataFrame(all_rows).to_csv(args.out, index=False)
            if weight_rows:
                pd.DataFrame(weight_rows).to_csv(args.weights_out, index=False)

    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    if weight_rows:
        pd.DataFrame(weight_rows).to_csv(args.weights_out, index=False)
    print(f"[phase2c] wrote {args.out}")
    print(f"[phase2c] wrote {args.weights_out}")


if __name__ == "__main__":
    main()
