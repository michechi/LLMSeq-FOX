"""
K-gram baseline analysis on MIMIC CKD->ESRD cohort (paper Table tab:mimic_ckd_audit, k-gram rows).

For each k in 1..MAX_K: learn P(Y=1|g) for every k-gram g on the training set,
predict each test sequence by averaging its k-gram probabilities, and report
AUC, F1, Precision, Recall plus token/type coverage. Averaged over N_SEEDS
random 80/20 stratified splits.

Inputs:  <MIMIC_PROCESSED>/ckd_cohort_ccs.csv
Outputs: <MIMIC_RESULTS>/kgram_analysis.csv (aggregated)
         <MIMIC_RESULTS>/kgram_analysis_raw.csv (per seed)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.common import MIMIC_PROCESSED, MIMIC_RESULTS

DEFAULT_SEED = 42
DEFAULT_MAX_K = 7
DEFAULT_N_SEEDS = 5


def extract_kgrams(tokens, k):
    if len(tokens) < k:
        return []
    return ["-".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]


def kgram_predict(tokens, probs, k):
    grams = extract_kgrams(tokens, k)
    if not grams:
        return 0.5
    return np.mean([probs.get(g, 0.5) for g in grams])


def run_single_seed(df, seed, max_k):
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )

    results = []
    for k in range(1, max_k + 1):
        stats = defaultdict(lambda: [0, 0])
        train_kgrams_set = set()
        for tokens, label in zip(train_df["tokens"], train_df["label"]):
            for g in extract_kgrams(tokens, k):
                stats[g][int(label)] += 1
                train_kgrams_set.add(g)

        probs = {
            g: c[1] / (c[0] + c[1]) if (c[0] + c[1]) > 0 else 0.5
            for g, c in stats.items()
        }

        test_total = 0
        test_covered = 0
        test_kgrams_set = set()
        for tokens in test_df["tokens"]:
            for g in extract_kgrams(tokens, k):
                test_total += 1
                test_kgrams_set.add(g)
                if g in train_kgrams_set:
                    test_covered += 1

        token_coverage = test_covered / test_total if test_total > 0 else 0.0
        type_coverage = (
            len(test_kgrams_set & train_kgrams_set) / len(test_kgrams_set)
            if test_kgrams_set else 0.0
        )

        preds = [kgram_predict(tokens, probs, k) for tokens in test_df["tokens"]]
        y_true = test_df["label"].values

        auc = roc_auc_score(y_true, preds)

        thresholds = np.linspace(0, 1, 101)
        f1s = [
            f1_score(y_true, (np.array(preds) >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        best_idx = int(np.argmax(f1s))
        best_thr = thresholds[best_idx]
        preds_bin = (np.array(preds) >= best_thr).astype(int)

        results.append({
            "k": k,
            "seed": seed,
            "n_unique_kgrams_train": len(train_kgrams_set),
            "n_unique_kgrams_test": len(test_kgrams_set),
            "token_coverage": token_coverage,
            "type_coverage": type_coverage,
            "AUC": auc,
            "F1": f1s[best_idx],
            "Precision": precision_score(y_true, preds_bin, zero_division=0),
            "Recall": recall_score(y_true, preds_bin, zero_division=0),
            "Best_threshold": best_thr,
        })

    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--cohort", type=Path,
                   default=Path(MIMIC_PROCESSED) / "ckd_cohort_ccs.csv",
                   help="Path to the CCS-mapped cohort CSV.")
    p.add_argument("--output_dir", type=Path, default=Path(MIMIC_RESULTS),
                   help="Where to write kgram_analysis{,_raw}.csv.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n_seeds", type=int, default=DEFAULT_N_SEEDS,
                   help="Number of random 80/20 splits to average over.")
    p.add_argument("--max_k", type=int, default=DEFAULT_MAX_K,
                   help="Maximum k-gram order to evaluate.")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    df = pd.read_csv(args.cohort)
    df["tokens"] = df["codes"].apply(json.loads)
    print(f"Cohort: {len(df)} patients  (Y=1: {df['label'].sum()}, "
          f"Y=0: {(df['label']==0).sum()})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "kgram_analysis.csv"

    all_results = []
    seeds = [args.seed + i for i in range(args.n_seeds)]
    for s in seeds:
        print(f"\n--- Seed {s} ---")
        res = run_single_seed(df, s, args.max_k)
        all_results.extend(res)
        for r in res:
            print(f"  k={r['k']}  AUC={r['AUC']:.4f}  F1={r['F1']:.4f}  "
                  f"cov_token={r['token_coverage']:.4f}  "
                  f"cov_type={r['type_coverage']:.4f}  "
                  f"({r['n_unique_kgrams_train']:,} train k-grams)")

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(output.with_name("kgram_analysis_raw.csv"), index=False)

    agg = raw_df.groupby("k").agg(
        n_kgrams_train=("n_unique_kgrams_train", "mean"),
        token_coverage_mean=("token_coverage", "mean"),
        token_coverage_std=("token_coverage", "std"),
        type_coverage_mean=("type_coverage", "mean"),
        type_coverage_std=("type_coverage", "std"),
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        Precision_mean=("Precision", "mean"),
        Recall_mean=("Recall", "mean"),
    ).reset_index()
    agg.to_csv(output, index=False)

    print("\n" + "=" * 80)
    print("K-GRAM DIAGNOSTIC TABLE  (CKD->ESRD, MIMIC-IV)")
    print("=" * 80)
    print(f"{'k':>2s}  {'|V_k|':>10s}  {'Cov(token)':>12s}  {'Cov(type)':>12s}  "
          f"{'AUC':>14s}  {'F1':>14s}")
    print("-" * 80)
    for _, r in agg.iterrows():
        print(f"{int(r['k']):2d}  {r['n_kgrams_train']:>10,.0f}  "
              f"{r['token_coverage_mean']:>6.1%}+/-{r['token_coverage_std']:>4.1%}  "
              f"{r['type_coverage_mean']:>6.1%}+/-{r['type_coverage_std']:>4.1%}  "
              f"{r['AUC_mean']:.4f}+/-{r['AUC_std']:.4f}  "
              f"{r['F1_mean']:.4f}+/-{r['F1_std']:.4f}")
    print("-" * 80)
    print(f"Averaged over {args.n_seeds} random 80/20 splits.")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
