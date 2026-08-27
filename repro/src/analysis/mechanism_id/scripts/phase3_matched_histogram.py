"""Phase 3 — Full 26-letter histogram-matched evaluation.

Group the test set by the *full 26-dim letter-count vector*. Evaluate models
only on sequences belonging to groups that contain both positive and negative
labels (ambiguous groups), since in a non-ambiguous group every classifier
is trivial. Optionally also return a balanced-within-group subsample.

Compares the best baselines from Phase 2 (lag-pair features) with the
content-only ceiling (count26 XGB) under three evaluation regimes:
  (a) full test set (standard);
  (b) ambiguous groups only;
  (c) balanced-within-ambiguous-groups.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    ALPHABET, KEY_LETTERS_IMPL, RESULTS_DIR, feat_count26, feat_count_key,
    feat_lag_pair, feat_lag_pair_key_only, feat_lag_pair_position,
    feat_residue, load_split, tokens,
)

SEED = 42


def _load(tag, rows_train, rows_eval):
    out = {}
    for split, rows in [("train", rows_train), ("val", rows_eval), ("test", rows_eval)]:
        X, y = load_split(tag, split, rows=rows)
        out[split] = (X, y)
    return out


def _labels(data, split):
    y = data[split][1]["Outcome"].values.astype(float)
    return (y >= 0.5).astype(int)


def _stack(data, split, extractor):
    return np.stack([extractor(tokens(s)) for s in data[split][0]["Sequences"]],
                    axis=0)


def _matched_indices(data, lag: int):
    """Group test sequences by (a) the full 26-dim letter-count vector and
    (b) the 6-dim key-letter count vector. Return index arrays for:
        full26_ambig    — groups w/ both labels, keyed by 26-vec
        keycount_ambig  — groups w/ both labels, keyed by key-6-vec
        full26_balanced — balanced subsample within full26 ambiguous groups
        keycount_balanced — ditto for key-count grouping
    """
    X = data["test"][0]["Sequences"].tolist()
    y = _labels(data, "test")
    rng = np.random.default_rng(SEED)

    def key_full(seq):
        c = Counter(tokens(seq))
        return tuple(c.get(a, 0) for a in ALPHABET)

    def key_key(seq):
        c = Counter(tokens(seq))
        return tuple(c.get(k, 0) for k in KEY_LETTERS_IMPL)

    def collect(key_fn):
        groups = defaultdict(list)
        for i, s in enumerate(X):
            groups[key_fn(s)].append(i)
        ambig_idx = []
        balanced_idx = []
        n_groups = len(groups)
        n_ambig_groups = 0
        for g, idxs in groups.items():
            labs = y[idxs]
            if labs.min() != labs.max():
                n_ambig_groups += 1
                ambig_idx.extend(idxs)
                pos = [i for i in idxs if y[i] == 1]
                neg = [i for i in idxs if y[i] == 0]
                k = min(len(pos), len(neg))
                if k > 0:
                    ppos = rng.choice(pos, size=k, replace=False).tolist()
                    nneg = rng.choice(neg, size=k, replace=False).tolist()
                    balanced_idx.extend(ppos)
                    balanced_idx.extend(nneg)
        return {
            "ambig_idx": np.array(ambig_idx, dtype=int),
            "balanced_idx": np.array(balanced_idx, dtype=int),
            "n_groups": n_groups,
            "n_ambig_groups": n_ambig_groups,
        }

    return {"full26": collect(key_full), "key6": collect(key_key)}


def _fit_logreg(Xtr, ytr, Xv, yv, Xte, yte, eval_idx=None):
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xte)[:, 1]

    def _scores(idx):
        if idx is None:
            yy, pp, pv_use, yv_use = yte, pt, pv, yv
        else:
            if len(idx) == 0:
                return {"AUC": float("nan"), "F1": float("nan"), "n": 0}
            yy, pp = yte[idx], pt[idx]
            pv_use, yv_use = pv, yv
        if len(np.unique(yy)) < 2:
            return {"AUC": float("nan"), "F1": float("nan"), "n": int(len(yy))}
        auc = float(roc_auc_score(yy, pp))
        thrs = np.linspace(0, 1, 201)
        bt = thrs[np.argmax([f1_score(yv_use, pv_use >= t) for t in thrs])]
        f1 = float(f1_score(yy, pp >= bt))
        return {"AUC": auc, "F1": f1, "n": int(len(yy))}

    return _scores


def _fit_xgb(Xtr, ytr, Xv, yv, Xte, yte):
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
        tree_method="hist", n_jobs=-1, random_state=SEED, eval_metric="logloss",
    )
    clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    pv = clf.predict_proba(Xv)[:, 1]
    pt = clf.predict_proba(Xte)[:, 1]

    def _scores(idx=None):
        if idx is None:
            yy, pp, pv_use, yv_use = yte, pt, pv, yv
        else:
            if len(idx) == 0:
                return {"AUC": float("nan"), "F1": float("nan"), "n": 0}
            yy, pp = yte[idx], pt[idx]
            pv_use, yv_use = pv, yv
        if len(np.unique(yy)) < 2:
            return {"AUC": float("nan"), "F1": float("nan"), "n": int(len(yy))}
        auc = float(roc_auc_score(yy, pp))
        thrs = np.linspace(0, 1, 201)
        bt = thrs[np.argmax([f1_score(yv_use, pv_use >= t) for t in thrs])]
        f1 = float(f1_score(yy, pp >= bt))
        return {"AUC": auc, "F1": f1, "n": int(len(yy))}

    return _scores


def run_tag(tag: str, data, lag: int, n: int):
    ytr = _labels(data, "train")
    yv = _labels(data, "val")
    yte = _labels(data, "test")
    matched = _matched_indices(data, lag)

    families = {
        "A1_count26":       lambda t: feat_count26(t),
        "A2_count_key":     lambda t: feat_count_key(t, KEY_LETTERS_IMPL),
        "B_residue":        lambda t: feat_residue(t, lag),
        "C1_lagpair":       lambda t: feat_lag_pair(t, lag),
        "C1k_lagpair_key":  lambda t: feat_lag_pair_key_only(t, lag, KEY_LETTERS_IMPL),
        "C2_lagpair_pos":   lambda t: feat_lag_pair_position(t, lag, n),
    }

    rows = []
    for fname, ex in families.items():
        t0 = time.time()
        Xtr = np.stack([ex(tokens(s)) for s in data["train"][0]["Sequences"]], 0)
        Xv = np.stack([ex(tokens(s)) for s in data["val"][0]["Sequences"]], 0)
        Xte = np.stack([ex(tokens(s)) for s in data["test"][0]["Sequences"]], 0)
        dt = time.time() - t0
        print(f"  [{tag}/{fname}] feat dim={Xtr.shape[1]} build_t={dt:.1f}s", flush=True)

        for model_name, fit_fn in [
            ("logreg_L2", _fit_logreg), ("xgboost", _fit_xgb),
        ]:
            t0 = time.time()
            scorer = fit_fn(Xtr, ytr, Xv, yv, Xte, yte)
            dt = time.time() - t0
            eval_regimes = {
                "standard": None,
                "full26_ambig": matched["full26"]["ambig_idx"],
                "full26_balanced": matched["full26"]["balanced_idx"],
                "key6_ambig": matched["key6"]["ambig_idx"],
                "key6_balanced": matched["key6"]["balanced_idx"],
            }
            for regime, idx in eval_regimes.items():
                sc = scorer(idx)
                rows.append({
                    "tag": tag, "family": fname, "model": model_name,
                    "eval_regime": regime,
                    "feat_dim": int(Xtr.shape[1]),
                    "n_eval": sc["n"],
                    "AUC": sc["AUC"], "F1": sc["F1"],
                    "train_sec": dt,
                    "n_groups_full26": matched["full26"]["n_groups"],
                    "n_ambig_groups_full26": matched["full26"]["n_ambig_groups"],
                    "n_groups_key6": matched["key6"]["n_groups"],
                    "n_ambig_groups_key6": matched["key6"]["n_ambig_groups"],
                })
                print(f"    [{model_name}/{regime}] AUC={sc['AUC']} F1={sc['F1']} n={sc['n']}",
                      flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", default=["6", "9"])
    p.add_argument("--rows_train", type=int, default=400_000)
    p.add_argument("--rows_eval", type=int, default=50_000)
    p.add_argument("--lag", type=int, default=7)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=Path,
                   default=RESULTS_DIR / "phase3_matched_histogram.csv")
    args = p.parse_args()

    all_rows = []
    for tag in args.tags:
        print(f"[phase3] tag={tag}", flush=True)
        data = _load(tag, args.rows_train, args.rows_eval)
        rows = run_tag(tag, data, args.lag, args.n)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(args.out, index=False)

    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"[phase3] wrote {args.out}")


if __name__ == "__main__":
    main()
