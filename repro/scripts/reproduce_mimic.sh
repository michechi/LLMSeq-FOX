#!/bin/bash
# Reproduce the MIMIC-IV CKD->ESRD audit (paper Section 5 + Appendix F.x).
#
# Pipeline:
#   1. Build the cohort from raw MIMIC-IV (CCS-mapped).
#   2. Optional: descriptive analysis (lag, discriminative codes).
#   3. K-gram diagnostic (Table tab:mimic_ckd_audit, k-gram rows).
#   4. Order-invariant + k-gram shuffle test.
#   5. Prepare 60/20/20 splits (ordered + shuffled) for neural models.
#   6. (Outside this script) submit neural models via scripts/slurm/submit_all_mimic.sh
#      or run a single model locally with `python -m src.mimic.train_mimic ...`.
#
# Required raw inputs at $MIMIC_RAW (default ./data/mimic/raw/hosp/):
#   diagnoses_icd.csv, admissions.csv, patients.csv  (PhysioNet credentialed access)

set -euo pipefail

REPRO_ROOT="${REPRO_ROOT:-$PWD}"
cd "$REPRO_ROOT"

echo "[mimic] 1/5  Build cohort (CCS) ..."
python -m src.mimic.build_ckd_cohort_ccs

echo "[mimic] 2/5  Descriptive analysis (lag + discriminative codes) ..."
python -m src.mimic.descriptive_analysis

echo "[mimic] 3/5  K-gram diagnostic ..."
python -m src.mimic.kgram_analysis

echo "[mimic] 4/5  Shuffle test (BoC + k-gram) ..."
python -m src.mimic.shuffle_test

echo "[mimic] 5/5  Prepare neural training splits ..."
python -m src.mimic.prepare_training_data

echo
echo "[mimic] Done. Now run sequence models on a GPU node:"
echo "  bash scripts/slurm/submit_all_mimic.sh"
echo "or single-shot for one model:"
echo "  python -m src.mimic.train_mimic --model transformer --data ordered"
