"""
Descriptive analysis for MIMIC CKD->ESRD cohort (paper Table tab:mimic_ckd_discriminative_codes).

Computes:
  1a. Lag analysis: spacing between consecutive kidney-related codes (CCS_158, CCS_157)
      among progressors. Reports natural lambda (median lag) and a histogram.
  1b. Discriminative codes: per-code prevalence ratio between Y=1 and Y=0.
      Saves the table that appears in the paper appendix.

Inputs:  <MIMIC_PROCESSED>/ckd_cohort_ccs.csv
         <MIMIC_CODES>/CCS_DX_categories.csv
Outputs (under <MIMIC_PROCESSED>):
  lag_analysis.csv
  discriminative_codes.csv
  descriptive_stats.txt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import MIMIC_CODES, MIMIC_PROCESSED

TARGET_CODES = {"CCS_158", "CCS_157"}  # CKD, acute renal failure


def load_data(cohort_path: Path, ccs_cat_path: Path):
    df = pd.read_csv(cohort_path)
    df["codes"] = df["codes"].apply(json.loads)
    ccs = pd.read_csv(ccs_cat_path)
    desc_map = dict(zip("CCS_" + ccs["category_code"].astype(str), ccs["category_desc"]))
    return df, desc_map


def lag_analysis(df: pd.DataFrame) -> pd.DataFrame:
    progressors = df[df["label"] == 1]
    records = []
    for _, row in progressors.iterrows():
        codes = row["codes"]
        positions = [i for i, c in enumerate(codes) if c in TARGET_CODES]
        if len(positions) < 2:
            continue
        for j in range(1, len(positions)):
            lag = positions[j] - positions[j - 1] - 1
            records.append({
                "subject_id": row["subject_id"],
                "pos_prev": positions[j - 1],
                "pos_curr": positions[j],
                "lag": lag,
                "seq_len": row["seq_len"],
            })
    return pd.DataFrame(records)


def report_lag(lag_df: pd.DataFrame) -> str:
    lags = lag_df["lag"]
    n_patients = lag_df["subject_id"].nunique()
    lines = [
        "=" * 60,
        "1a. LAG ANALYSIS  (natural lambda)",
        "=" * 60,
        f"Progressors with >=2 target codes: {n_patients}",
        f"Total consecutive-pair gaps:       {len(lags)}",
        "",
        f"  Median lag:  {lags.median():.1f}",
        f"  IQR:         [{lags.quantile(0.25):.1f}, {lags.quantile(0.75):.1f}]",
        f"  Mean (SD):   {lags.mean():.2f} ({lags.std():.2f})",
        f"  Min / Max:   {lags.min()} / {lags.max()}",
        "",
        "Distribution (histogram bins):",
    ]
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 100, np.inf]
    labels = ["0", "1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51+"]
    hist = pd.cut(lags, bins=bins, labels=labels, right=True).value_counts().sort_index()
    for label, count in hist.items():
        pct = 100 * count / len(lags)
        bar = "#" * int(pct / 2)
        lines.append(f"    {label:>5s}: {count:>6d}  ({pct:5.1f}%)  {bar}")
    lines.append("")
    return "\n".join(lines)


def discriminative_codes(df: pd.DataFrame, desc_map: dict) -> pd.DataFrame:
    y1 = df[df["label"] == 1]
    y0 = df[df["label"] == 0]
    n1, n0 = len(y1), len(y0)

    def code_stats(group):
        presence = Counter()
        total_count = Counter()
        for codes in group["codes"]:
            unique = set(codes)
            for c in unique:
                presence[c] += 1
            for c in codes:
                total_count[c] += 1
        return presence, total_count

    pres1, cnt1 = code_stats(y1)
    pres0, cnt0 = code_stats(y0)
    all_codes = sorted(set(pres1) | set(pres0))

    rows = []
    for c in all_codes:
        f1 = pres1.get(c, 0) / n1
        f0 = pres0.get(c, 0) / n0
        ratio = f1 / f0 if f0 > 0 else np.inf
        log_ratio = np.log2(ratio) if (f0 > 0 and f1 > 0) else np.nan
        diff = f1 - f0
        mean1 = cnt1.get(c, 0) / n1
        mean0 = cnt0.get(c, 0) / n0
        rows.append({
            "code": c,
            "description": desc_map.get(c, ""),
            "freq_Y1": round(f1, 4),
            "freq_Y0": round(f0, 4),
            "ratio": round(ratio, 4),
            "log_ratio": round(log_ratio, 4) if not np.isnan(log_ratio) else np.nan,
            "diff": round(diff, 4),
            "mean_count_Y1": round(mean1, 4),
            "mean_count_Y0": round(mean0, 4),
        })

    disc_df = pd.DataFrame(rows).sort_values(
        "log_ratio", ascending=False, na_position="last"
    ).reset_index(drop=True)
    return disc_df


def report_disc(disc_df: pd.DataFrame) -> str:
    lines = [
        "=" * 60,
        "1b. DISCRIMINATIVE CODES  (empirical S)",
        "=" * 60,
        f"Total unique codes: {len(disc_df)}",
        "",
        "Top 20 codes by log2(freq_Y1/freq_Y0):",
        f"{'Rank':>4s}  {'Code':<12s} {'log2R':>6s} {'freqY1':>7s} {'freqY0':>7s} {'diff':>7s}  Description",
        "-" * 90,
    ]
    top = disc_df.head(20)
    for i, row in top.iterrows():
        lines.append(
            f"{i+1:4d}  {row['code']:<12s} {row['log_ratio']:>6.2f} "
            f"{row['freq_Y1']:>7.3f} {row['freq_Y0']:>7.3f} {row['diff']:>7.3f}  "
            f"{row['description']}"
        )
    lines.append("")

    bottom = disc_df.dropna(subset=["log_ratio"]).tail(10).iloc[::-1]
    lines.append("Bottom 10 codes (enriched in Y=0 non-progressors):")
    lines.append(
        f"{'Rank':>4s}  {'Code':<12s} {'log2R':>6s} {'freqY1':>7s} {'freqY0':>7s} {'diff':>7s}  Description"
    )
    lines.append("-" * 90)
    for i, (_, row) in enumerate(bottom.iterrows()):
        lines.append(
            f"{i+1:4d}  {row['code']:<12s} {row['log_ratio']:>6.2f} "
            f"{row['freq_Y1']:>7.3f} {row['freq_Y0']:>7.3f} {row['diff']:>7.3f}  "
            f"{row['description']}"
        )
    lines.append("")

    if "CCS_158" in disc_df["code"].values:
        ckd_row = disc_df[disc_df["code"] == "CCS_158"].iloc[0]
        lines.append(
            f"Sanity check: CCS_158 (CKD) -- log2R={ckd_row['log_ratio']:.2f}, "
            f"freqY1={ckd_row['freq_Y1']:.3f}, freqY0={ckd_row['freq_Y0']:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--cohort", type=Path,
                   default=Path(MIMIC_PROCESSED) / "ckd_cohort_ccs.csv")
    p.add_argument("--ccs_categories", type=Path,
                   default=Path(MIMIC_CODES) / "CCS_DX_categories.csv")
    p.add_argument("--output_dir", type=Path, default=Path(MIMIC_PROCESSED))
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df, desc_map = load_data(args.cohort, args.ccs_categories)
    n1 = (df["label"] == 1).sum()
    n0 = (df["label"] == 0).sum()
    header = (
        f"CKD->ESRD Cohort: {len(df)} patients  "
        f"(Y=1: {n1}, Y=0: {n0})\n"
        f"Unique CCS tokens: {len(set(c for codes in df['codes'] for c in codes))}\n"
    )
    print(header)

    print("Computing lag analysis...")
    lag_df = lag_analysis(df)
    lag_report = report_lag(lag_df)
    print(lag_report)
    lag_df.to_csv(args.output_dir / "lag_analysis.csv", index=False)

    print("Computing discriminative codes...")
    disc_df = discriminative_codes(df, desc_map)
    disc_report = report_disc(disc_df)
    print(disc_report)
    disc_df.to_csv(args.output_dir / "discriminative_codes.csv", index=False)

    full_report = header + "\n" + lag_report + "\n" + disc_report
    (args.output_dir / "descriptive_stats.txt").write_text(full_report)
    print(f"Outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
