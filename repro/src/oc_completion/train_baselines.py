"""Fit the traditional feature baselines on the original OC training split and
save them for pair evaluation.

Feature families (extractors imported from the mechanism-id audit - single
source): letter counts (26), aggregated lag-7 pair counts (676), all-offset
stacked pair counts (19*676 = 12844), position-aware lag-7 pair indicators
(13*676 = 8788, sparse), aggregated lag-7 trigram counts (26^3, sparse),
key-only lag-7 trigram counts (216), contiguous k-gram probability baselines
(k = 1, 2, 3).

Models: L2 logistic regression (lbfgs, C=1.0) as in
`mechanism_id/scripts/phase2_baseline_ladder.py`; XGBoost (hist, 200 trees,
depth 6) for the all-offset family.

By construction, `count26` and `lag_pair` MUST assign exactly equal scores to
the two candidates of every matched two-hole pair; `eval_pairs` verifies this.

Usage (from repro/ root):
    DATA_DIR=/root/LLMSeq/data python -m src.oc_completion.train_baselines \
        --task ocdet [--smoke] [--families all]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.analysis.mechanism_id.scripts.common import (
    feat_count26,
    feat_lag_pair,
    feat_lag_pair_position_sparse_row,
    feat_lag_trigram_key_only,
    tokens,
)
from src.oc_completion.oracle import KEY_LETTERS, LAG, N_EVENTS
from src.oc_completion.train_dl import DATA_ROOT, RESULTS_DIR, append_result

CKPT_ROOT = Path(os.environ.get("LLMSEQ_ROOT", "/root/LLMSeq")) / "checkpoints" / "oc_completion"
TASK_DIRNAME = {"ocdet": "oc_deterministic", "ocnoisy": "oc_noisy"}

SEED = 9550


# --------------------------------------------------------------------------
# feature builders (dense / sparse)
# --------------------------------------------------------------------------
def all_offset_sparse_row(toks):
    """lambda-agnostic stacked pair counts, dim (n-1)*676, as sparse row.

    Mirrors mechanism_id/scripts/phase2c_lag_agnostic.feat_stacked_all_lag.
    """
    counts: Counter = Counter()
    L = len(toks)
    for lag in range(1, N_EVENTS):
        for t in range(L - lag):
            a, b = toks[t], toks[t + lag]
            ai, bi = ord(a) - 65, ord(b) - 65
            if 0 <= ai < 26 and 0 <= bi < 26:
                counts[(lag - 1) * 676 + ai * 26 + bi] += 1
    cols = sorted(counts)
    return cols, [float(counts[c]) for c in cols], (N_EVENTS - 1) * 676


def lag_trigram_sparse_row(toks):
    """aggregated lag-7 trigram counts over the full alphabet, 26^3 sparse."""
    counts: Counter = Counter()
    L = len(toks)
    for t in range(L - 2 * LAG):
        a, b, c = toks[t], toks[t + LAG], toks[t + 2 * LAG]
        ai, bi, ci = ord(a) - 65, ord(b) - 65, ord(c) - 65
        if 0 <= ai < 26 and 0 <= bi < 26 and 0 <= ci < 26:
            counts[ai * 676 + bi * 26 + ci] += 1
    cols = sorted(counts)
    return cols, [float(counts[c]) for c in cols], 26 ** 3


def sparse_matrix(seqs, row_fn):
    indptr, indices, values, ncols = [0], [], [], 0
    for s in seqs:
        cols, data, D = row_fn(tokens(s))
        ncols = max(ncols, D)
        indices.extend(cols)
        values.extend(data)
        indptr.append(len(indices))
    return csr_matrix((np.asarray(values, dtype=np.float32),
                       np.asarray(indices, dtype=np.int64),
                       np.asarray(indptr, dtype=np.int64)),
                      shape=(len(seqs), ncols))


def dense_matrix(seqs, extractor):
    return np.stack([extractor(tokens(s)) for s in seqs], axis=0)


FAMILIES = {
    # name -> (kind, builder, model)
    "count26": ("dense", lambda seqs: dense_matrix(seqs, feat_count26), "logreg"),
    "lag_pair": ("dense", lambda seqs: dense_matrix(
        seqs, lambda t: feat_lag_pair(t, LAG)), "logreg"),
    "all_offset_pair": ("sparse", lambda seqs: sparse_matrix(
        seqs, all_offset_sparse_row), "logreg"),
    "all_offset_pair_xgb": ("sparse", lambda seqs: sparse_matrix(
        seqs, all_offset_sparse_row), "xgb"),
    "pos_lag_pair": ("sparse", lambda seqs: sparse_matrix(
        seqs, lambda t: feat_lag_pair_position_sparse_row(t, LAG, N_EVENTS)),
        "logreg"),
    "lag_trigram": ("sparse", lambda seqs: sparse_matrix(
        seqs, lag_trigram_sparse_row), "logreg"),
    "lag_trigram_key": ("dense", lambda seqs: dense_matrix(
        seqs, lambda t: feat_lag_trigram_key_only(t, LAG, KEY_LETTERS)),
        "logreg"),
}

KGRAM_KS = (1, 2, 3)


def make_model(kind: str):
    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                  random_state=42, n_jobs=-1)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                         subsample=0.8, tree_method="hist", n_jobs=-1,
                         random_state=42, eval_metric="logloss")


# --------------------------------------------------------------------------
# contiguous k-gram probability baseline (signal_decomposition.kgram_baseline)
# --------------------------------------------------------------------------
class KgramBaseline:
    def __init__(self, k: int):
        self.k = k
        self.probs: dict = {}

    @staticmethod
    def _ngrams(seq: str, k: int):
        toks = tokens(seq)
        return ["-".join(toks[i:i + k]) for i in range(len(toks) - k + 1)]

    def fit(self, seqs, labels):
        stats: dict = {}
        for s, y in zip(seqs, labels):
            for ng in self._ngrams(s, self.k):
                c = stats.setdefault(ng, [0, 0])
                c[int(y)] += 1
        self.probs = {ng: c[1] / (c[0] + c[1]) for ng, c in stats.items()}
        return self

    def score(self, seqs) -> np.ndarray:
        out = np.empty(len(seqs))
        for i, s in enumerate(seqs):
            ps = [self.probs.get(ng, 0.5) for ng in self._ngrams(s, self.k)]
            out[i] = float(np.mean(ps)) if ps else 0.5
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ocdet", "ocnoisy"], required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--families", default="all")
    args = ap.parse_args()

    tag = f"{args.task}_smoke" if args.smoke else args.task
    suffix = "_smoke" if args.smoke else ""
    fams = list(FAMILIES) if args.families == "all" else args.families.split(",")

    def load(split):
        X = pd.read_csv(DATA_ROOT / f"X_{split}_{tag}.csv")["Sequences"].tolist()
        y = pd.read_csv(DATA_ROOT / f"y_{split}_{tag}.csv")
        return X, y["Outcome"].astype(int).values, y["Latent"].astype(int).values

    X_train, y_train, _ = load("train")
    X_test, y_test, ystar_test = load("test")

    out_root = CKPT_ROOT / TASK_DIRNAME[args.task] / "baselines"
    out_root.mkdir(parents=True, exist_ok=True)
    results = []

    for fam in fams:
        kind, builder, model_kind = FAMILIES[fam]
        t0 = time.time()
        Xtr = builder(X_train)
        Xte = builder(X_test)
        model = make_model(model_kind)
        model.fit(Xtr, y_train)
        probs = model.predict_proba(Xte)[:, 1]
        auc_obs = float(roc_auc_score(y_test, probs))
        auc_lat = float(roc_auc_score(ystar_test, probs))
        wall = round(time.time() - t0, 1)
        path = out_root / f"{fam}{suffix}.joblib"
        joblib.dump({"model": model, "family": fam, "task": args.task,
                     "feature_kind": kind}, path)
        results.append({"family": fam, "model": model_kind,
                        "test_auc_obs": auc_obs, "test_auc_latent": auc_lat,
                        "train_s": wall, "path": str(path)})
        append_result(RESULTS_DIR / "training_results.csv", {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": fam, "training_mode": "baseline", "task": args.task,
            "seed": SEED, "train_rows": len(X_train), "epochs_done": 1,
            "best_epoch": 1, "test_auc_obs": round(auc_obs, 6),
            "test_auc_latent": round(auc_lat, 6),
            "threshold": "n/a", "n_params": getattr(
                model, "coef_", np.zeros(0)).size,
            "recipe": f"{model_kind} on {fam} (phase2 settings)",
            "wallclock_s": wall, "checkpoint": str(path),
            "smoke": args.smoke, "host": os.uname().nodename,
        })
        print(f"[baselines] {fam}: auc_obs={auc_obs:.4f} "
              f"auc_latent={auc_lat:.4f} ({wall}s)", flush=True)

    for k in KGRAM_KS:
        t0 = time.time()
        kb = KgramBaseline(k).fit(X_train, y_train)
        probs = kb.score(X_test)
        auc_obs = float(roc_auc_score(y_test, probs))
        auc_lat = float(roc_auc_score(ystar_test, probs))
        wall = round(time.time() - t0, 1)
        path = out_root / f"kgram{k}{suffix}.joblib"
        joblib.dump({"model": kb, "family": f"kgram{k}", "task": args.task,
                     "feature_kind": "kgram"}, path)
        results.append({"family": f"kgram{k}", "model": "prob-avg",
                        "test_auc_obs": auc_obs, "test_auc_latent": auc_lat,
                        "train_s": wall, "path": str(path)})
        append_result(RESULTS_DIR / "training_results.csv", {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": f"kgram{k}", "training_mode": "baseline",
            "task": args.task, "seed": SEED, "train_rows": len(X_train),
            "epochs_done": 1, "best_epoch": 1,
            "test_auc_obs": round(auc_obs, 6),
            "test_auc_latent": round(auc_lat, 6), "threshold": "n/a",
            "n_params": len(kb.probs),
            "recipe": f"contiguous {k}-gram probability averaging "
                      "(signal_decomposition.kgram_baseline)",
            "wallclock_s": wall, "checkpoint": str(path),
            "smoke": args.smoke, "host": os.uname().nodename,
        })
        print(f"[baselines] kgram{k}: auc_obs={auc_obs:.4f} "
              f"auc_latent={auc_lat:.4f} ({wall}s)", flush=True)

    with open(out_root / f"summary{suffix}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
