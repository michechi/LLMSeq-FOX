"""Shared pair-scoring and metric computation for the matched-completion study.

A model enters through a single interface: a `score_fn(list_of_sequence_strings)
-> np.ndarray of shape (N, 2)` returning the two class logits per sequence
(column 1 = positive class). The candidate score used everywhere is the
log-odds `z1 - z0`, which is monotone in the positive-class softmax
probability; margins are computed on it.

margin(pair) = score(positive candidate) - score(negative candidate)
win: margin > tol; tie: |margin| <= tol; loss: margin < -tol
pair accuracy = (wins + 0.5 * ties) / n_pairs
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

TOLERANCES = (1e-8, 1e-6, 1e-10)
MAIN_TOL = 1e-8
BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 20260826


def score_pairs(score_fn: Callable[[List[str]], np.ndarray],
                pairs: pd.DataFrame) -> pd.DataFrame:
    """Score both candidates of every pair. Returns per-pair frame with
    candidate logits, probabilities and the pair margin (log-odds scale)."""
    seqs = pairs["cand0"].tolist() + pairs["cand1"].tolist()
    logits = np.asarray(score_fn(seqs), dtype=np.float64)
    n = len(pairs)
    z0 = logits[:n]          # candidate 0 of each pair
    z1 = logits[n:]          # candidate 1 of each pair
    out = pairs.copy()
    for idx, z in ((0, z0), (1, z1)):
        out[f"cand{idx}_logit0"] = z[:, 0]
        out[f"cand{idx}_logit1"] = z[:, 1]
        out[f"cand{idx}_logodds"] = z[:, 1] - z[:, 0]
        ez = np.exp(z - z.max(axis=1, keepdims=True))
        out[f"cand{idx}_prob1"] = ez[:, 1] / ez.sum(axis=1)
    pos = out["positive_index"].values
    s0 = out["cand0_logodds"].values
    s1 = out["cand1_logodds"].values
    out["score_pos"] = np.where(pos == 1, s1, s0)
    out["score_neg"] = np.where(pos == 1, s0, s1)
    out["margin"] = out["score_pos"] - out["score_neg"]
    return out


def pair_metrics(scored: pd.DataFrame, tol: float = MAIN_TOL) -> Dict:
    """Aggregate pair metrics with bootstrap CI on pair accuracy."""
    m = scored["margin"].values
    n = len(m)
    res: Dict = {"n_pairs": n}
    for t in TOLERANCES:
        wins = (m > t).mean()
        ties = (np.abs(m) <= t).mean()
        acc = wins + 0.5 * ties
        key = "" if t == tol else f"_tol{t:g}"
        res[f"win_rate{key}"] = float(wins)
        res[f"tie_rate{key}"] = float(ties)
        res[f"loss_rate{key}"] = float((m < -t).mean())
        res[f"pair_accuracy{key}"] = float(acc)
    res["mean_margin"] = float(m.mean())
    res["median_margin"] = float(np.median(m))
    res["std_margin"] = float(m.std())

    # flattened candidate AUC: every candidate scored against its latent label
    from sklearn.metrics import roc_auc_score
    scores = np.concatenate([scored["score_pos"].values, scored["score_neg"].values])
    labels = np.concatenate([np.ones(n), np.zeros(n)])
    if np.ptp(scores) == 0:
        res["flattened_auc"] = 0.5
    else:
        res["flattened_auc"] = float(roc_auc_score(labels, scores))

    # pair-level bootstrap CI of pair accuracy at the main tolerance
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_pair = np.where(m > tol, 1.0, np.where(np.abs(m) <= tol, 0.5, 0.0))
    boots = np.empty(BOOTSTRAP_ITERS)
    for i in range(BOOTSTRAP_ITERS):
        boots[i] = per_pair[rng.integers(0, n, size=n)].mean()
    res["pair_accuracy_ci_lo"] = float(np.percentile(boots, 2.5))
    res["pair_accuracy_ci_hi"] = float(np.percentile(boots, 97.5))
    return res


def grouped_metrics(scored: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    rows = []
    for key, grp in scored.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        res = pair_metrics(grp)
        rows.append({**dict(zip(by, key)), **res})
    return pd.DataFrame(rows)


def quick_pair_diag(score_fn, pairs: pd.DataFrame, tol: float = MAIN_TOL) -> Dict:
    """Cheap per-epoch diagnostic: pair accuracy + mean margin only."""
    scored = score_pairs(score_fn, pairs)
    m = scored["margin"].values
    acc = float((m > tol).mean() + 0.5 * (np.abs(m) <= tol).mean())
    return {"completion_pair_acc": acc, "completion_mean_margin": float(m.mean())}
