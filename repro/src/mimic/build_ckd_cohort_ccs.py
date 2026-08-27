"""
Build CKD -> ESRD cohort with CCS-mapped codes (paper Section 5, App. F.x).

Tokens are CCS category codes (~293 categories) instead of raw 3-char ICD codes.
Unmapped ICD codes -> "OTHER" token.

Inputs (under MIMIC_RAW, default ./data/mimic/raw/hosp):
  - diagnoses_icd.csv
  - admissions.csv

Outputs (under MIMIC_PROCESSED, default ./data/mimic/processed):
  - ckd_cohort_ccs.csv
  - ckd_cohort_ccs_stats.txt

CCS mapping is read from MIMIC_CODES/CCS_DX_mapping.csv (shipped under data/mimic/codes/).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import MIMIC_CODES, MIMIC_PROCESSED, MIMIC_RAW


def is_ckd(code: str, version: int) -> bool:
    """CKD stages 1-5 (excludes stage 6 / ESRD)."""
    if version == 10:
        return bool(pd.notna(code) and code.startswith("N18") and code not in ("N186", "N189"))
    if version == 9:
        return bool(pd.notna(code) and code.startswith("585") and code not in ("5856", "5859"))
    return False


def is_esrd(code: str, version: int) -> bool:
    """ESRD / dialysis / transplant codes."""
    if version == 10:
        return code.startswith("N186") or code.startswith("Z992") or code.startswith("Z49")
    if version == 9:
        return code.startswith("5856") or code.startswith("V451") or code.startswith("V56")
    return False


def main() -> None:
    raw = Path(MIMIC_RAW)
    out = Path(MIMIC_PROCESSED)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading tables from {raw} ...")
    diag = pd.read_csv(raw / "diagnoses_icd.csv", dtype={"icd_code": str, "icd_version": int})
    adm = pd.read_csv(raw / "admissions.csv", parse_dates=["admittime", "dischtime"])
    ccs_df = pd.read_csv(Path(MIMIC_CODES) / "CCS_DX_mapping.csv")
    print(f"  Diagnoses: {len(diag):,} rows")

    vocab_map = {9: "ICD9CM", 10: "ICD10CM"}
    ccs_lookup = {}
    for _, row in ccs_df.iterrows():
        ccs_lookup[(str(row["code"]), row["vocabulary_id"])] = f"CCS_{int(row['category_code'])}"

    def get_ccs(icd_code: str, icd_version: int) -> str:
        return ccs_lookup.get((icd_code, vocab_map[icd_version]), "OTHER")

    diag["is_ckd"] = [is_ckd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]
    diag["is_esrd"] = [is_esrd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]
    diag["ccs"] = [get_ccs(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]

    ckd_patients = set(diag.loc[diag["is_ckd"], "subject_id"])
    esrd_patients = set(diag.loc[diag["is_esrd"], "subject_id"])
    print(f"CKD patients: {len(ckd_patients):,}")

    diag = diag.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
    diag = diag.sort_values(["subject_id", "admittime", "seq_num"]).reset_index(drop=True)

    print("Building patient sequences...")
    records = []
    excluded_same_admission = 0
    excluded_short = 0

    for sid in sorted(ckd_patients):
        pdf = diag[diag["subject_id"] == sid]

        ckd_rows = pdf[pdf["is_ckd"]]
        first_ckd_hadm = ckd_rows.iloc[0]["hadm_id"]
        first_ckd_admittime = ckd_rows.iloc[0]["admittime"]

        esrd_rows = pdf[pdf["is_esrd"]]

        if len(esrd_rows) > 0 and sid in esrd_patients:
            first_esrd_hadm = esrd_rows.iloc[0]["hadm_id"]
            first_esrd_admittime = esrd_rows.iloc[0]["admittime"]

            if first_esrd_hadm == first_ckd_hadm:
                excluded_same_admission += 1
                continue

            if first_esrd_admittime <= first_ckd_admittime:
                excluded_same_admission += 1
                continue

            seq = pdf[pdf["admittime"] < first_esrd_admittime]
            label = 1
        else:
            seq = pdf
            label = 0

        if len(seq) < 10:
            excluded_short += 1
            continue

        codes = seq["ccs"].tolist()
        hadm_ids = seq["hadm_id"].astype(int).tolist()

        records.append({
            "subject_id": int(sid),
            "label": label,
            "seq_len": len(codes),
            "codes": json.dumps(codes),
            "hadm_ids": json.dumps(hadm_ids),
            "n_admissions": len(set(hadm_ids)),
        })

    print(f"  Excluded (same-admission CKD+ESRD): {excluded_same_admission}")
    print(f"  Excluded (< 10 codes): {excluded_short}")

    cohort = pd.DataFrame(records)

    all_codes = set()
    other_count = 0
    total_count = 0
    for c in cohort["codes"]:
        tokens = json.loads(c)
        all_codes.update(tokens)
        total_count += len(tokens)
        other_count += sum(1 for t in tokens if t == "OTHER")

    print(f"\n{'='*60}")
    print(f"FINAL COHORT (CCS): {len(cohort):,} patients")
    print(f"  Progressors (Y=1): {(cohort['label']==1).sum():,}")
    print(f"  Non-progressors (Y=0): {(cohort['label']==0).sum():,}")
    print(f"  Class balance: {cohort['label'].mean():.4f}")
    print(f"\nAlphabet size: {len(all_codes)} CCS categories (includes OTHER)")
    print(f"  OTHER tokens: {other_count:,} / {total_count:,} ({other_count/total_count*100:.2f}%)")
    print(f"\nSequence length distribution:")
    print(cohort["seq_len"].describe().to_string())

    cohort_path = out / "ckd_cohort_ccs.csv"
    cohort.to_csv(cohort_path, index=False)
    print(f"\nSaved cohort to {cohort_path}")

    stats_lines = [
        "CKD -> ESRD Cohort Statistics (CCS-mapped)",
        "=" * 45,
        f"Total patients: {len(cohort):,}",
        f"Progressors (Y=1): {(cohort['label']==1).sum():,}",
        f"Non-progressors (Y=0): {(cohort['label']==0).sum():,}",
        f"Class balance (frac Y=1): {cohort['label'].mean():.4f}",
        "",
        f"Alphabet size: {len(all_codes)} CCS categories",
        f"OTHER tokens: {other_count:,} / {total_count:,} ({other_count/total_count*100:.2f}%)",
        "",
        "Sequence lengths:",
        cohort["seq_len"].describe().to_string(),
        "",
        "Admissions per patient:",
        cohort["n_admissions"].describe().to_string(),
        "",
        f"Excluded (same-admission onset): {excluded_same_admission}",
        f"Excluded (< 10 codes): {excluded_short}",
    ]
    stats_path = out / "ckd_cohort_ccs_stats.txt"
    stats_path.write_text("\n".join(stats_lines))
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
