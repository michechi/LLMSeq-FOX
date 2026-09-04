"""Metrics for the Ordered Compliance exhaustive-replacement audit.

This module is deliberately model agnostic.  Callers provide arrays of
positive-class logits/probabilities for the fixed oracle manifests.  All
correctness labels come from the latent canonical oracle, never from noisy
observed labels.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BOOTSTRAP_SEED = 20260831
BOOTSTRAP_ITERS = 1000


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    return (float(roc_auc_score(y, score)) if np.unique(y).size == 2
            else float("nan"))


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    return (float(average_precision_score(y, score))
            if np.unique(y).size == 2 else float("nan"))


def _mean_or_nan(values: Iterable[float]) -> float:
    a = np.asarray(list(values), dtype=np.float64)
    return float(np.nanmean(a)) if a.size and np.any(np.isfinite(a)) else float("nan")


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_col: str,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_boot: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile interval from resampling whole clusters with replacement."""
    if frame.empty:
        return float("nan"), float("nan")
    groups = [g for _, g in frame.groupby(cluster_col, sort=False)]
    if not groups:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, len(groups), size=len(groups))
        sampled = pd.concat([groups[i] for i in draw], ignore_index=True)
        boots[b] = statistic(sampled)
    finite = boots[np.isfinite(boots)]
    if not finite.size:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(finite, [2.5, 97.5]))


def hole_cluster_bootstrap_ci(
    cluster_ids: Iterable[Any],
    *,
    statistic: str,
    values: Iterable[float] | None = None,
    labels: Iterable[int] | None = None,
    scores: Iterable[float] | None = None,
    n_boot: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> dict[str, float | int | str]:
    """Fast cluster-bootstrap interval for core hole metrics.

    Whole base sequences, rather than their 20 positions or 520 candidates,
    are sampled with replacement.  ``statistic='mean'`` supports metrics that
    are averages of per-position/per-sequence contributions.  ``'auroc'``
    computes a weighted ROC AUC for each cluster draw without materializing a
    duplicated candidate frame.  The latter is important for the full
    10,000-base hole audit.

    The returned interval is percentile-based.  Bootstrap replicates with an
    undefined statistic (for example, a single-class AUC draw) are omitted and
    counted in ``bootstrap_finite_replicates``.
    """
    cluster_array = np.asarray(list(cluster_ids), dtype=object)
    if cluster_array.ndim != 1 or cluster_array.size == 0:
        raise ValueError("cluster_ids must be a non-empty one-dimensional array")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    cluster_codes, unique_clusters = pd.factorize(cluster_array, sort=False)
    if np.any(cluster_codes < 0):
        raise ValueError("hole cluster IDs must not be missing")
    n_clusters = len(unique_clusters)
    rng = np.random.default_rng(seed)
    probability = np.full(n_clusters, 1.0 / n_clusters)
    boots = np.full(n_boot, np.nan, dtype=np.float64)

    if statistic == "mean":
        if values is None:
            raise ValueError("values are required for statistic='mean'")
        value_array = np.asarray(list(values), dtype=np.float64)
        if value_array.shape != cluster_array.shape:
            raise ValueError("values and cluster_ids must have the same shape")
        finite = np.isfinite(value_array)
        sums = np.bincount(
            cluster_codes[finite], weights=value_array[finite], minlength=n_clusters
        )
        counts = np.bincount(cluster_codes[finite], minlength=n_clusters)
        estimate = float(value_array[finite].mean()) if finite.any() else float("nan")
        for b in range(n_boot):
            multiplicity = rng.multinomial(n_clusters, probability)
            denominator = float(multiplicity @ counts)
            if denominator:
                boots[b] = float(multiplicity @ sums) / denominator
    elif statistic == "auroc":
        if labels is None or scores is None:
            raise ValueError("labels and scores are required for statistic='auroc'")
        label_array = np.asarray(list(labels), dtype=np.int8)
        score_array = np.asarray(list(scores), dtype=np.float64)
        if label_array.shape != cluster_array.shape or score_array.shape != cluster_array.shape:
            raise ValueError("labels, scores, and cluster_ids must have the same shape")
        finite = np.isfinite(score_array) & np.isin(label_array, (0, 1))
        label_array = label_array[finite]
        score_array = score_array[finite]
        code_array = cluster_codes[finite]
        estimate = _safe_auc(label_array, score_array)

        # Scores never change across resamples, so sort and identify tie groups
        # once.  Each bootstrap then needs only weighted bincount/cumulative-sum
        # operations and exactly matches half-credit ROC handling for ties.
        order = np.argsort(score_array, kind="stable")
        sorted_scores = score_array[order]
        sorted_labels = label_array[order]
        sorted_codes = code_array[order]
        if len(sorted_scores):
            group_start = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
            score_groups = np.cumsum(group_start) - 1
            n_groups = int(score_groups[-1]) + 1
            for b in range(n_boot):
                multiplicity = rng.multinomial(n_clusters, probability)
                row_weight = multiplicity[sorted_codes].astype(np.float64)
                positive = np.bincount(
                    score_groups,
                    weights=row_weight * sorted_labels,
                    minlength=n_groups,
                )
                negative = np.bincount(
                    score_groups,
                    weights=row_weight * (1 - sorted_labels),
                    minlength=n_groups,
                )
                total_positive = positive.sum()
                total_negative = negative.sum()
                if total_positive and total_negative:
                    negative_below = np.cumsum(negative) - negative
                    numerator = np.sum(
                        positive * (negative_below + 0.5 * negative)
                    )
                    boots[b] = numerator / (total_positive * total_negative)
    else:
        raise ValueError("statistic must be 'mean' or 'auroc'")

    finite_boots = boots[np.isfinite(boots)]
    alpha = 1.0 - confidence_level
    if finite_boots.size:
        low, high = np.quantile(finite_boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    else:
        low = high = float("nan")
    return {
        "estimate": float(estimate),
        "ci_lo": float(low),
        "ci_hi": float(high),
        "confidence_level": float(confidence_level),
        "bootstrap_clusters": int(n_clusters),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_finite_replicates": int(finite_boots.size),
        "bootstrap_statistic": statistic,
    }


def _position_frame(
    hole_manifest: pd.DataFrame,
    oracle_labels: np.ndarray,
    positive_logits: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    n = len(hole_manifest)
    expected = (n, 20, 26)
    for name, a in (("oracle_labels", oracle_labels),
                    ("positive_logits", positive_logits),
                    ("probabilities", probabilities)):
        if np.asarray(a).shape != expected:
            raise ValueError(f"{name} shape {np.asarray(a).shape}, expected {expected}")
    zeros = (oracle_labels == 0).sum(axis=2)
    ones = 26 - zeros
    decisive = (zeros > 0) & (ones > 0)
    fixed_zero = zeros == 26
    fixed_one = ones == 26
    if not np.all(decisive.astype(int) + fixed_zero.astype(int)
                  + fixed_one.astype(int) == 1):
        raise AssertionError("hole categories are not exhaustive and disjoint")
    base_ids = np.repeat(hole_manifest["base_sequence_id"].to_numpy(), 20)
    base_y = np.repeat(hole_manifest["base_Y_star"].to_numpy(dtype=np.int8), 20)
    return pd.DataFrame({
        "base_sequence_id": base_ids,
        "base_Y_star": base_y,
        "position": np.tile(np.arange(20, dtype=np.int8), n),
        "decisive": decisive.ravel(),
        "fixed_zero": fixed_zero.ravel(),
        "fixed_one": fixed_one.ravel(),
        "can_create_compliance": ((base_y.reshape(n, 20) == 0) & decisive).ravel(),
        "can_destroy_compliance": ((base_y.reshape(n, 20) == 1) & decisive).ravel(),
        "n_candidate_zero": zeros.ravel(),
        "n_candidate_one": ones.ravel(),
        "delta": np.ptp(positive_logits, axis=2).ravel(),
        "logit_std": np.std(positive_logits, axis=2).ravel(),
        "probability_range": np.ptp(probabilities, axis=2).ravel(),
    })


def position_localization_metrics(
    hole_manifest: pd.DataFrame,
    oracle_labels: np.ndarray,
    positive_logits: np.ndarray,
    probabilities: np.ndarray,
    *,
    bootstrap_iters: int = 0,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Pooled and within-sequence sensitivity/localization metrics."""
    pos = _position_frame(hole_manifest, oracle_labels, positive_logits,
                          probabilities)
    rows: list[dict[str, Any]] = []
    segments = {
        "all": np.ones(len(pos), dtype=bool),
        "negative_create": pos["base_Y_star"].eq(0).to_numpy(),
        "positive_destroy": pos["base_Y_star"].eq(1).to_numpy(),
    }
    for segment_index, (segment, mask) in enumerate(segments.items()):
        sub = pos.loc[mask]
        y = sub["decisive"].to_numpy(dtype=np.int8)
        score = sub["delta"].to_numpy(dtype=np.float64)
        per_sequence_ap, top1, top3_recall = [], [], []
        sequences_with_decisive = 0
        for _, g in sub.groupby("base_sequence_id", sort=False):
            gy = g["decisive"].to_numpy(dtype=bool)
            gs = g["delta"].to_numpy(dtype=np.float64)
            if not gy.any():
                continue
            sequences_with_decisive += 1
            per_sequence_ap.append(float(average_precision_score(gy, gs)))
            max_score = gs.max()
            # Tie-aware expected correctness among maximizers.
            top1.append(float(gy[gs == max_score].mean()))
            order = np.argsort(-gs, kind="stable")[:3]
            top3_recall.append(float(gy[order].sum() / gy.sum()))
        n_sequences = sub["base_sequence_id"].nunique()
        row: dict[str, Any] = {
            "segment": segment,
            "n_positions": len(sub),
            "n_decisive": int(y.sum()),
            "n_sequences": n_sequences,
            "fraction_sequences_with_decisive": (
                sequences_with_decisive / n_sequences if n_sequences else float("nan")),
            "position_auroc": _safe_auc(y, score),
            "position_auprc": _safe_ap(y, score),
            "within_sequence_average_precision": _mean_or_nan(per_sequence_ap),
            "top1_localization_accuracy": _mean_or_nan(top1),
            "top3_decisive_recall": _mean_or_nan(top3_recall),
            "mean_delta_decisive": _mean_or_nan(sub.loc[sub.decisive, "delta"]),
            "mean_delta_fixed_zero": _mean_or_nan(sub.loc[sub.fixed_zero, "delta"]),
            "mean_delta_fixed_one": _mean_or_nan(sub.loc[sub.fixed_one, "delta"]),
        }
        if bootstrap_iters:
            bootstrap = hole_cluster_bootstrap_ci(
                sub["base_sequence_id"],
                statistic="auroc",
                labels=y,
                scores=score,
                n_boot=bootstrap_iters,
                seed=bootstrap_seed + segment_index,
            )
            row.update({
                "position_auroc_ci_lo": bootstrap["ci_lo"],
                "position_auroc_ci_hi": bootstrap["ci_hi"],
                "bootstrap_clusters": bootstrap["bootstrap_clusters"],
                "bootstrap_replicates": bootstrap["bootstrap_replicates"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _ranking_row(labels: np.ndarray, logits: np.ndarray, target: int) -> dict[str, float]:
    valid = labels == target
    score = logits if target == 1 else -logits
    if valid.sum() in (0, len(valid)):
        return {}
    invalid = ~valid
    auc = float(roc_auc_score(valid.astype(np.int8), score))
    max_score = score.max()
    tied = score == max_score
    order = np.argsort(-score, kind="stable")
    first_rank = int(np.flatnonzero(valid[order])[0]) + 1
    return {
        "candidate_auc": auc,
        "pairwise_win_rate": auc,  # ROC AUC uses half credit for ties.
        "top1_valid": float(valid[int(np.argmax(score))]),
        "tie_aware_top1": float(valid[tied].mean()),
        "mrr_first_valid": 1.0 / first_rank,
        "valid_minus_invalid_margin": float(score[valid].mean() - score[invalid].mean()),
        "random_choice_reference": float(valid.mean()),
    }


def valid_filling_metrics(
    hole_manifest: pd.DataFrame,
    oracle_labels: np.ndarray,
    positive_logits: np.ndarray,
    *,
    bootstrap_iters: int = 0,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Rank all oracle-valid letters at decisive positions for both targets."""
    n = len(hole_manifest)
    if oracle_labels.shape != (n, 20, 26) or positive_logits.shape != (n, 20, 26):
        raise ValueError("hole arrays must have shape [N,20,26]")
    base_y = hole_manifest["base_Y_star"].to_numpy(dtype=np.int8)
    records: list[dict[str, Any]] = []
    excluded = {0: {"empty": 0, "all": 0}, 1: {"empty": 0, "all": 0}}
    for s in range(n):
        for position in range(20):
            labels = oracle_labels[s, position]
            if labels.min() == labels.max():
                for target in (0, 1):
                    count = int((labels == target).sum())
                    excluded[target]["empty" if count == 0 else "all"] += 1
                continue
            for target in (0, 1):
                metrics = _ranking_row(labels, positive_logits[s, position], target)
                case = ("create_compliance" if base_y[s] == 0 and target == 1
                        else "destroy_compliance" if base_y[s] == 1 and target == 0
                        else "preserve_base_label")
                records.append({
                    "sequence_index": s,
                    "base_sequence_id": hole_manifest.iloc[s]["base_sequence_id"],
                    "position": position,
                    "target": target,
                    "case": case,
                    **metrics,
                })
    detail = pd.DataFrame(records)
    rows = []
    if detail.empty:
        return pd.DataFrame(rows)
    selections: list[tuple[str, int | None, pd.Series]] = []
    selections.append(("all_targets", None, pd.Series(True, index=detail.index)))
    for target in (0, 1):
        selections.append((f"target_{target}", target, detail.target.eq(target)))
    for case in ("create_compliance", "destroy_compliance", "preserve_base_label"):
        selections.append((case, None, detail.case.eq(case)))
    for selection_index, (segment, target, mask) in enumerate(selections):
        g = detail.loc[mask]
        if g.empty:
            continue
        # Secondary pooled AUC across candidates in this segment.
        pooled_y, pooled_score = [], []
        for r in g.itertuples(index=False):
            s_idx = int(r.sequence_index)
            labels = oracle_labels[s_idx, int(r.position)]
            this_target = int(r.target)
            scores = positive_logits[s_idx, int(r.position)]
            pooled_y.extend((labels == this_target).tolist())
            pooled_score.extend((scores if this_target == 1 else -scores).tolist())
        row: dict[str, Any] = {
            "segment": segment,
            "target": target if target is not None else "mixed",
            "n_positions": len(g),
            "macro_candidate_auc": float(g.candidate_auc.mean()),
            "pooled_candidate_auc": _safe_auc(np.asarray(pooled_y), np.asarray(pooled_score)),
            "valid_invalid_pairwise_win_rate": float(g.pairwise_win_rate.mean()),
            "top1_valid_filling_accuracy": float(g.top1_valid.mean()),
            "tie_aware_top1_accuracy": float(g.tie_aware_top1.mean()),
            "mrr_first_valid_filling": float(g.mrr_first_valid.mean()),
            "mean_valid_minus_invalid_logit_margin": float(
                g.valid_minus_invalid_margin.mean()),
            "random_choice_reference": float(g.random_choice_reference.mean()),
        }
        if target is not None:
            row["excluded_empty_target_positions"] = excluded[target]["empty"]
            row["excluded_all_valid_positions"] = excluded[target]["all"]
        if bootstrap_iters:
            auc_bootstrap = hole_cluster_bootstrap_ci(
                g["base_sequence_id"],
                statistic="mean",
                values=g["candidate_auc"],
                n_boot=bootstrap_iters,
                seed=bootstrap_seed + 100 + selection_index,
            )
            top1_bootstrap = hole_cluster_bootstrap_ci(
                g["base_sequence_id"],
                statistic="mean",
                values=g["top1_valid"],
                n_boot=bootstrap_iters,
                seed=bootstrap_seed + 200 + selection_index,
            )
            row.update({
                "macro_candidate_auc_ci_lo": auc_bootstrap["ci_lo"],
                "macro_candidate_auc_ci_hi": auc_bootstrap["ci_hi"],
                "top1_valid_filling_accuracy_ci_lo": top1_bootstrap["ci_lo"],
                "top1_valid_filling_accuracy_ci_hi": top1_bootstrap["ci_hi"],
                "bootstrap_clusters": auc_bootstrap["bootstrap_clusters"],
                "bootstrap_replicates": auc_bootstrap["bootstrap_replicates"],
            })
        rows.append(row)
    return pd.DataFrame(rows)


def stability_metrics(
    hole_manifest: pd.DataFrame,
    oracle_labels: np.ndarray,
    positive_logits: np.ndarray,
    probabilities: np.ndarray,
    thresholds: dict[str, float],
    *,
    bootstrap_iters: int = 0,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Score class stability at fixed-zero and fixed-one positions."""
    n = len(hole_manifest)
    if any(a.shape != (n, 20, 26)
           for a in (oracle_labels, positive_logits, probabilities)):
        raise ValueError("hole arrays must have shape [N,20,26]")
    rows = []
    base_ids = hole_manifest["base_sequence_id"].to_numpy()
    position_base_ids = np.broadcast_to(base_ids[:, None], (n, 20))
    for category_index, (category, selector, latent) in enumerate((
        ("fixed_zero", np.all(oracle_labels == 0, axis=2), 0),
        ("fixed_one", np.all(oracle_labels == 1, axis=2), 1),
    )):
        logits = positive_logits[selector]
        probs = probabilities[selector]
        fixed_base_ids = position_base_ids[selector]
        if not len(logits):
            continue
        for threshold_index, (threshold_name, threshold) in enumerate(thresholds.items()):
            pred = probs >= threshold
            correct = pred == latent
            score_correct = probs if latent == 1 else 1.0 - probs
            per_position_flip = np.any(pred != pred[:, :1], axis=1).astype(float)
            row: dict[str, Any] = {
                "category": category,
                "threshold_name": threshold_name,
                "threshold": float(threshold),
                "n_positions": len(logits),
                "mean_logit_range": float(np.ptp(logits, axis=1).mean()),
                "mean_logit_std": float(np.std(logits, axis=1).mean()),
                "mean_probability_range": float(np.ptp(probs, axis=1).mean()),
                "prediction_flip_rate": float(per_position_flip.mean()),
                "all_candidates_correct_fraction": float(np.all(correct, axis=1).mean()),
                "individual_candidate_accuracy": float(correct.mean()),
                "mean_worst_case_correct_class_score": float(
                    np.min(score_correct, axis=1).mean()),
            }
            if bootstrap_iters:
                bootstrap = hole_cluster_bootstrap_ci(
                    fixed_base_ids,
                    statistic="mean",
                    values=per_position_flip,
                    n_boot=bootstrap_iters,
                    seed=(bootstrap_seed + 300 + category_index * 10
                          + threshold_index),
                )
                row.update({
                    "prediction_flip_rate_ci_lo": bootstrap["ci_lo"],
                    "prediction_flip_rate_ci_hi": bootstrap["ci_hi"],
                    "bootstrap_clusters": bootstrap["bootstrap_clusters"],
                    "bootstrap_replicates": bootstrap["bootstrap_replicates"],
                })
            rows.append(row)
    return pd.DataFrame(rows)


def calibrate_tie_tolerance(duplicate_scores: np.ndarray) -> dict[str, float]:
    """Infer a tie tolerance from repeated scores of identical candidates.

    ``duplicate_scores`` is [candidate, repeat].  The tolerance is the maximum
    observed within-candidate range.  Exact deterministic scoring therefore
    legitimately yields a zero tolerance instead of an arbitrary 1e-8.
    """
    scores = np.asarray(duplicate_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("duplicate_scores must be [candidate, repeat>=2]")
    ranges = np.ptp(scores, axis=1)
    return {
        "tie_tolerance": float(ranges.max(initial=0.0)),
        "mean_duplicate_range": float(ranges.mean()),
        "max_duplicate_range": float(ranges.max(initial=0.0)),
        "n_jitter_candidates": int(scores.shape[0]),
        "n_repeats": int(scores.shape[1]),
    }


def strict_pair_metrics(
    scored: pd.DataFrame,
    *,
    tolerance: float,
    cluster_col: str | None = None,
    n_boot: int = BOOTSTRAP_ITERS,
) -> dict[str, Any]:
    """Matched-pair metrics with pair or base-sequence cluster bootstrap."""
    margins = scored["margin"].to_numpy(dtype=np.float64)
    wins = margins > tolerance
    ties = np.abs(margins) <= tolerance
    credits = np.where(wins, 1.0, np.where(ties, 0.5, 0.0))
    labels = np.r_[np.ones(len(scored), dtype=np.int8),
                   np.zeros(len(scored), dtype=np.int8)]
    scores = np.r_[scored["positive_logit"].to_numpy(dtype=float),
                   scored["negative_logit"].to_numpy(dtype=float)]
    cluster = cluster_col if cluster_col and cluster_col in scored else "pair_id"
    bootstrap = hole_cluster_bootstrap_ci(
        scored[cluster],
        statistic="mean",
        values=credits,
        n_boot=n_boot,
    )
    lo, hi = bootstrap["ci_lo"], bootstrap["ci_hi"]
    return {
        "n_pairs": len(scored),
        "tie_tolerance": float(tolerance),
        "strict_win_rate": float(wins.mean()),
        "tie_rate": float(ties.mean()),
        "loss_rate": float((margins < -tolerance).mean()),
        "pair_accuracy": float(credits.mean()),
        "pair_accuracy_ci_lo": lo,
        "pair_accuracy_ci_hi": hi,
        "mean_margin": float(margins.mean()),
        "median_margin": float(np.median(margins)),
        "flattened_candidate_auc": _safe_auc(labels, scores),
        "bootstrap_cluster": cluster,
        "bootstrap_clusters": bootstrap["bootstrap_clusters"],
        "bootstrap_replicates": bootstrap["bootstrap_replicates"],
    }


def standard_observed_oracle_auc(pi: float, latent_prevalence: float) -> float:
    """Closed-form AUC of the binary latent oracle against noisy labels.

    With symmetric flips, the oracle score has two levels.  The expression is
    computed from the empirical latent prevalence rather than hard-coded for a
    single noise setting.
    """
    rho = float(latent_prevalence)
    pi = float(pi)
    p_y1 = rho * (1.0 - pi) + (1.0 - rho) * pi
    p_y0 = 1.0 - p_y1
    if p_y1 <= 0 or p_y0 <= 0:
        return float("nan")
    p_star1_given_y1 = rho * (1.0 - pi) / p_y1
    p_star1_given_y0 = rho * pi / p_y0
    # P(s+ > s-) + .5 P(s+ == s-) for binary scores.
    return float(
        p_star1_given_y1 * (1.0 - p_star1_given_y0)
        + 0.5 * (
            p_star1_given_y1 * p_star1_given_y0
            + (1.0 - p_star1_given_y1) * (1.0 - p_star1_given_y0)
        )
    )
