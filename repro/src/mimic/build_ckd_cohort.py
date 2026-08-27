"""
Build CKD -> ESRD cohort from MIMIC-IV with raw 3-character ICD tokens.

Cohort definition (same as build_ckd_cohort_ccs.py):
  - Inclusion: patients with at least one CKD code (ICD-10 N18.1-N18.5, ICD-9 585.1-585.5)
  - Label Y=1: ESRD codes appear AFTER the first CKD code (progressor); sequence truncated
    before the first ESRD admission
  - Label Y=0: no ESRD codes ever (non-progressor)
  - Sequences ordered by (admittime, seq_num)
  - Filtering: >=10 codes after flattening; exclude same-admission CKD+ESRD onset

Output: <MIMIC_PROCESSED>/ckd_cohort.csv
        <MIMIC_PROCESSED>/ckd_cohort_stats.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import MIMIC_PROCESSED, MIMIC_RAW
from src.mimic.build_ckd_cohort_ccs import is_ckd, is_esrd


def main() -> None:
    raw = Path(MIMIC_RAW)
    out = Path(MIMIC_PROCESSED)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading tables from {raw} ...")
    diag = pd.read_csv(raw / "diagnoses_icd.csv", dtype={"icd_code": str, "icd_version": int})
    adm = pd.read_csv(raw / "admissions.csv", parse_dates=["admittime", "dischtime"])
    pat = pd.read_csv(raw / "patients.csv")

    print(f"  Diagnoses: {len(diag):,} rows")
    print(f"  Admissions: {len(adm):,} rows")
    print(f"  Patients: {len(pat):,} unique patients")

    diag["is_ckd"] = [is_ckd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]
    diag["is_esrd"] = [is_esrd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]

    ckd_patients = set(diag.loc[diag["is_ckd"], "subject_id"])
    esrd_patients = set(diag.loc[diag["is_esrd"], "subject_id"])
    print(f"\nCKD patients: {len(ckd_patients):,}")
    print(f"ESRD patients (all): {len(esrd_patients):,}")

    diag = diag.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
    diag = diag.sort_values(["subject_id", "admittime", "seq_num"]).reset_index(drop=True)

    print("\nBuilding patient sequences...")
    records = []
    excluded_same_admission = 0
    excluded_short = 0

    for sid in sorted(ckd_patients):
        pdf = diag[diag["subject_id"] == sid].copy()

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

        codes = []
        hadm_ids = []
        for _, row in seq.iterrows():
            c = row["icd_code"]
            v = row["icd_version"]
            token = c[:3] if v == 10 else "ICD9_" + c[:3]
            codes.append(token)
            hadm_ids.append(int(row["hadm_id"]))

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
    print(f"\n{'='*60}")
    print(f"FINAL COHORT: {len(cohort):,} patients")
    print(f"  Progressors (Y=1): {(cohort['label']==1).sum():,}")
    print(f"  Non-progressors (Y=0): {(cohort['label']==0).sum():,}")
    print(f"  Class balance: {cohort['label'].mean():.3f}")

    print("\nSequence length distribution:")
    print(cohort["seq_len"].describe().to_string())

    print("\nAdmissions per patient:")
    print(cohort["n_admissions"].describe().to_string())

    all_codes = set()
    for c in cohort["codes"]:
        all_codes.update(json.loads(c))
    print(f"\nAlphabet size (unique 3-char codes): {len(all_codes)}")

    cohort_path = out / "ckd_cohort.csv"
    cohort.to_csv(cohort_path, index=False)
    print(f"\nSaved to {cohort_path}")

    stats_lines = [
        "CKD -> ESRD Cohort Statistics",
        "=" * 40,
        f"Total patients: {len(cohort):,}",
        f"Progressors (Y=1): {(cohort['label']==1).sum():,}",
        f"Non-progressors (Y=0): {(cohort['label']==0).sum():,}",
        f"Class balance (frac Y=1): {cohort['label'].mean():.4f}",
        "",
        "Sequence lengths:",
        cohort["seq_len"].describe().to_string(),
        "",
        "Admissions per patient:",
        cohort["n_admissions"].describe().to_string(),
        "",
        f"Alphabet size: {len(all_codes)}",
        "",
        f"Excluded (same-admission onset): {excluded_same_admission}",
        f"Excluded (< 10 codes): {excluded_short}",
    ]
    (out / "ckd_cohort_stats.txt").write_text("\n".join(stats_lines))
    print(f"Saved stats to {out / 'ckd_cohort_stats.txt'}")


if __name__ == "__main__":
    main()
