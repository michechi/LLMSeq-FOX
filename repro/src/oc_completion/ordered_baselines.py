"""Train the Ordered Compliance shortcut baselines on complete sequences.

The artifacts expose the same two-logit scoring interface as neural models so
hole and strict-pair evaluation cannot accidentally use different examples or
metrics.  No hole, pair, witness, or structural annotation enters fitting.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from src.oc_completion.ordered_features import (
    DETERMINISTIC_OCCUPANCY_FAMILIES,
    build_features,
    deterministic_occupancy_score,
)
from src.oc_completion.ordered_io import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    RESULT_ROOT,
    atomic_json_dump,
    load_split,
    pi_slug,
    valid_done,
)
from src.oc_completion.oracle import oc_label_tokens, tokens_of

TRAINED_FAMILIES = {
    "letter_count_logreg": ("letter_count", "logreg"),
    "lag_pair_logreg": ("lag_pair", "logreg"),
    "chain_occupancy_logreg": ("chain_occupancy", "logreg"),
    "chain_occupancy_xgb": ("chain_counts_xgb", "xgb"),
    "position_lag_pair_logreg": ("position_lag_pair", "logreg"),
    "lag_trigram_logreg": ("lag_trigram", "logreg"),
}
ALL_FAMILIES = tuple(TRAINED_FAMILIES) + DETERMINISTIC_OCCUPANCY_FAMILIES + ("oracle",)


def _auc(labels, scores) -> float:
    return (float(roc_auc_score(labels, scores))
            if np.unique(labels).size == 2 else float("nan"))


def select_f1_threshold(probabilities, labels) -> tuple[float, float]:
    """Select a deterministic validation-only probability threshold."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, np.quantile(p, np.linspace(0, 1, 401))])
    scores = np.asarray([f1_score(y, p >= t, zero_division=0) for t in candidates])
    best = np.flatnonzero(scores == scores.max())
    # Stable tie break: closest to 0.5, then lower threshold.
    chosen = best[np.lexsort((candidates[best], np.abs(candidates[best] - 0.5)))[0]]
    return float(candidates[chosen]), float(scores[chosen])


def _make_estimator(kind: str, seed: int, threads: int):
    if kind == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs",
                                  random_state=seed, n_jobs=threads)
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=1.0, tree_method="hist",
            n_jobs=threads, random_state=seed, eval_metric="logloss",
        )
    raise KeyError(kind)


def _score_artifact(artifact: dict, sequences) -> tuple[np.ndarray, np.ndarray]:
    family = artifact["family"]
    if family == "oracle":
        decision = np.asarray([
            float(oc_label_tokens(tokens_of(s) if "\x1f" in s else list(s)))
            for s in sequences
        ])
        probability = decision.copy()
    elif family in DETERMINISTIC_OCCUPANCY_FAMILIES:
        decision = deterministic_occupancy_score(sequences, family)
        # Monotone bounded map; only ranking is intrinsic to these scores.
        probability = 1.0 / (1.0 + np.exp(-decision))
    else:
        X = build_features(sequences, artifact["feature_family"])
        model = artifact.get("estimator", artifact.get("model"))
        if hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(X), dtype=np.float64)
            probability = 1.0 / (1.0 + np.exp(-np.clip(decision, -80, 80)))
        else:
            probability = np.asarray(model.predict_proba(X)[:, 1], dtype=np.float64)
            clipped = np.clip(probability, 1e-12, 1 - 1e-12)
            decision = np.log(clipped / (1 - clipped))
    logits = np.stack([np.zeros(len(decision), dtype=np.float32),
                       np.asarray(decision, dtype=np.float32)], axis=1)
    return logits, np.asarray(probability, dtype=np.float32)


def load_baseline_scorer(path: Path):
    artifact = joblib.load(path)

    def score(sequences):
        logits, _ = _score_artifact(artifact, sequences)
        return logits

    meta = {
        "model": artifact["family"],
        "family": artifact["family"],
        "seed": int(artifact.get("seed", 9550)),
        "noise_pi": float(artifact.get("noise_pi", 0.0)),
        "checkpoint_epoch": 1,
        "validation_threshold": float(
            artifact.get("validation_threshold", 0.5)),
        "training_mode": artifact.get("training_mode", "baseline"),
        "checkpoint_path": str(path),
    }
    return score, meta


def train_family(
    family: str,
    pi: float,
    *,
    data_root: Path,
    checkpoint_root: Path,
    seed: int = 9550,
    threads: int = 8,
    smoke: bool = False,
) -> dict:
    if family not in ALL_FAMILIES:
        raise KeyError(family)
    suffix = "_smoke" if smoke else ""
    out_dir = checkpoint_root / "baselines" / f"pi_{pi_slug(pi)}" / f"{family}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.json"
    identity = {"family": family, "noise_pi": float(pi), "seed": seed,
                "smoke": smoke}
    if valid_done(done_path, identity):
        with open(out_dir / "result.json") as handle:
            return json.load(handle)

    train = load_split(data_root, "train", pi)
    val = load_split(data_root, "val", pi)
    test = load_split(data_root, "test", pi)
    if smoke:
        train, val, test = train.iloc[:2000], val.iloc[:500], test.iloc[:500]
    t0 = time.time()
    if family in TRAINED_FAMILIES:
        feature_family, model_kind = TRAINED_FAMILIES[family]
        X_train = build_features(train.X.tolist(), feature_family)
        model = _make_estimator(model_kind, seed, threads)
        model.fit(X_train, train.Y_observed.to_numpy(dtype=np.int8))
        artifact = {
            **identity, "training_mode": "baseline",
            "feature_family": feature_family, "model_kind": model_kind,
            "model": family, "estimator": model,
        }
    else:
        artifact = {
            **identity,
            "training_mode": "oracle" if family == "oracle" else "deterministic_baseline",
            "feature_family": family, "model_kind": "deterministic",
            "model": family, "estimator": None,
        }

    _, val_prob = _score_artifact(artifact, val.X.tolist())
    threshold, val_f1 = select_f1_threshold(val_prob, val.Y_observed)
    artifact["validation_threshold"] = threshold
    artifact_path = out_dir / "artifact.joblib"
    joblib.dump(artifact, artifact_path)

    _, test_prob = _score_artifact(artifact, test.X.tolist())
    pred = test_prob >= threshold
    result = {
        **identity, "model": family, "training_mode": artifact["training_mode"],
        "train_rows": len(train), "validation_threshold": threshold,
        "validation_f1": val_f1,
        "standard_observed_auc": _auc(test.Y_observed, test_prob),
        "standard_latent_auc": _auc(test.Y_star, test_prob),
        "standard_observed_f1": float(f1_score(test.Y_observed, pred,
                                               zero_division=0)),
        "training_time_s": round(time.time() - t0, 3),
        "hardware": os.uname().nodename,
        "checkpoint": str(artifact_path),
    }
    atomic_json_dump(result, out_dir / "result.json")
    atomic_json_dump({"status": "complete", **identity,
                      "artifact": str(artifact_path)}, done_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", type=float, required=True)
    parser.add_argument("--families", default="all")
    parser.add_argument("--seed", type=int, default=9550)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    families = ALL_FAMILIES if args.families == "all" else tuple(args.families.split(","))
    results = [train_family(
        family, args.pi, data_root=args.data_root,
        checkpoint_root=args.checkpoint_root, seed=args.seed,
        threads=args.threads, smoke=args.smoke,
    ) for family in families]
    run_dir = args.result_root / "runs" / "baselines"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"pi_{pi_slug(args.pi)}{'_smoke' if args.smoke else ''}.json"
    atomic_json_dump(results, out)
    print(f"[ordered_baselines] wrote {out} ({len(results)} families)")


if __name__ == "__main__":
    main()
