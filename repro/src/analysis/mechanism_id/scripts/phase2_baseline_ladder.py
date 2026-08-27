"""Phase 2 — Baseline ladder A..D on Tricky tasks (standard eval).

Feature families:
  A1 — full 26-dim letter counts
  A2 — key-only 6-dim letter counts
  B  — residue-class counts mod lag (26 * lag = 182 dims for lag=7)
  C1 — aggregated 26x26 lag-pair counts
  C1k — aggregated 6x6 lag-pair counts restricted to key letters
  C2 — position-aware lag-pair indicators (sparse, (n-lag) * 26 * 26 = ~8788 dim)
  D1k — aggregated lag-trigram counts restricted to key letters (6^3 = 216)

Models per family: LogisticRegression (L2), XGBoost (where feasible).

Metrics: AUC (primary), F1 (threshold tuned on val).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    KEY_LETTERS_IMPL, RESULTS_DIR, bundle_features, feat_count26,
    feat_count_key, feat_lag_pair, feat_lag_pair_key_only, feat_lag_pair_position,
    feat_lag_pair_position_sparse_row, feat_lag_trigram_key_only, feat_residue,
    load_split, stack_sparse_bundle, tokens,
)

SEED = 42


def _load(tag: str, rows_train: int | None, rows_eval: int | None):
    out = {}
    for split, rows in [("train", rows_train), ("val", rows_eval), ("test", rows_eval)]:
        X, y = load_split(tag, split, rows=rows)
        out[split] = (X, y)
    return out


def _labels(data, split):
    y = data[split][1]["Outcome"].values.astype(float)
    return (y >= 0.5).astype(int)


def _fit_logreg(Xtr, ytr, Xv, yv, Xte, yte, C=1.0, max_iter=2000):
    clf = LogisticRegression(
        C=C, max_iter=max_iter, solver="lbfgs", random_state=SEED, n_jobs=-1,
    )
    clf.fit(Xtr, ytr)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, pt))
    thrs = np.linspace(0, 1, 201)
    bt = thrs[np.argmax([f1_score(yv, pv >= t) for t in thrs])]
    f1 = float(f1_score(yte, pt >= bt))
    return auc, f1, bt


def _fit_xgb(Xtr, ytr, Xv, yv, Xte, yte):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, None, None
    import scipy.sparse as sp
    # XGBoost's hist method accepts CSR; but keeping memory in check for big C2.
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, tree_method="hist", n_jobs=-1,
        random_state=SEED, eval_metric="logloss",
    )
    clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, pt))
    thrs = np.linspace(0, 1, 201)
    bt = thrs[np.argmax([f1_score(yv, pv >= t) for t in thrs])]
    f1 = float(f1_score(yte, pt >= bt))
    return auc, f1, bt


def run_family(tag: str, family: str, data, lag: int, n: int):
    ytr = _labels(data, "train")
    yv = _labels(data, "val")
    yte = _labels(data, "test")

    t0 = time.time()
    # C2 uses sparse CSR to avoid OOM on 8788-dim dense @ 400K rows.
    if family == "C2_lagpair_pos":
        mats = stack_sparse_bundle(data,
                                   lambda t: feat_lag_pair_position_sparse_row(t, lag, n))
        Xtr, Xv, Xte = mats["train"], mats["val"], mats["test"]
        build_t = time.time() - t0
        print(f"  [{tag}/{family}] SPARSE feat dim={Xtr.shape[1]} nnz={Xtr.nnz} "
              f"build_t={build_t:.1f}s", flush=True)
    else:
        if family == "A1_count26":
            extractor = feat_count26
        elif family == "A2_count_key":
            def extractor(toks): return feat_count_key(toks, KEY_LETTERS_IMPL)
        elif family == "B_residue":
            def extractor(toks): return feat_residue(toks, lag)
        elif family == "C1_lagpair":
            def extractor(toks): return feat_lag_pair(toks, lag)
        elif family == "C1k_lagpair_key":
            def extractor(toks): return feat_lag_pair_key_only(toks, lag, KEY_LETTERS_IMPL)
        elif family == "D1k_lagtrigram_key":
            def extractor(toks): return feat_lag_trigram_key_only(toks, lag, KEY_LETTERS_IMPL)
        else:
            raise ValueError(f"unknown family {family}")

        fb = bundle_features(data, extractor, name=family)
        build_t = time.time() - t0
        Xtr, Xv, Xte = fb.Xtr, fb.Xv, fb.Xte
        print(f"  [{tag}/{family}] feat dim={Xtr.shape[1]}, build_t={build_t:.1f}s", flush=True)

    rows = []
    for model_name, fit_fn in [
        ("logreg_L2", _fit_logreg),
        ("xgboost", _fit_xgb),
    ]:
        t0 = time.time()
        auc, f1, thr = fit_fn(Xtr, ytr, Xv, yv, Xte, yte)
        dt = time.time() - t0
        rows.append({
            "tag": tag, "family": family, "model": model_name,
            "feat_dim": int(Xtr.shape[1]),
            "n_train": Xtr.shape[0], "n_val": Xv.shape[0], "n_test": Xte.shape[0],
            "AUC": auc, "F1": f1, "threshold": thr,
            "train_sec": dt,
        })
        print(f"    [{model_name}] AUC={auc} F1={f1} dt={dt:.1f}s", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["6", "9"])
    p.add_argument("--families", nargs="+",
                   default=["A1_count26", "A2_count_key", "B_residue",
                            "C1_lagpair", "C1k_lagpair_key", "C2_lagpair_pos",
                            "D1k_lagtrigram_key"])
    p.add_argument("--rows_train", type=int, default=400_000)
    p.add_argument("--rows_eval", type=int, default=50_000)
    p.add_argument("--lag", type=int, default=7)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=Path,
                   default=RESULTS_DIR / "phase2_ladder_standard.csv")
    args = p.parse_args()

    all_rows = []
    for tag in args.tags:
        print(f"[phase2] loading tag={tag}", flush=True)
        data = _load(tag, args.rows_train, args.rows_eval)
        for fam in args.families:
            rows = run_family(tag, fam, data, args.lag, args.n)
            all_rows.extend(rows)
            # Incremental dump so partial results survive a crash.
            pd.DataFrame(all_rows).to_csv(args.out, index=False)

    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"[phase2] wrote {args.out}")


if __name__ == "__main__":
    main()
