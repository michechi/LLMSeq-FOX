#!/bin/bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=0-06:00:00
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --output=slurm-%j-%x.out

# Run a single mechanism-ID phase inside an apptainer container.

set -o errexit
set -o nounset
set -o pipefail

# Usage:
#   PHASE=1 sbatch -J mechid_p1 src/analysis/mechanism_id/scripts/slurm/run_phase.sh
#   PHASE=2 EXTRA="--rows_train 200000" sbatch -J mechid_p2 src/analysis/mechanism_id/scripts/slurm/run_phase.sh

PHASE="${PHASE:?ERROR: set PHASE=1..6}"
EXTRA="${EXTRA:-}"
REPO="${REPRO_ROOT:-$PWD}"

echo "=== PHASE $PHASE  job=$SLURM_JOB_ID  node=$(hostname) ==="
echo "=== $(date)  cpus=$SLURM_CPUS_ON_NODE mem=$SLURM_MEM_PER_NODE ==="

# Replace with your container path; any image with Python 3.10+, torch, transformers,
# scikit-learn, xgboost, pandas, matplotlib will work.
CONTAINER="${CONTAINER:?ERROR: export CONTAINER=/path/to/your/pytorch.sif}"
BINDS="--bind $REPO:$REPO"

CASE_SCRIPT=""
case "$PHASE" in
    1) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase1_label_audit.py" ;;
    1b) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase1b_rule_from_logreg.py" ;;
    2) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase2_baseline_ladder.py" ;;
    2c) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase2c_lag_agnostic.py" ;;
    3) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase3_matched_histogram.py" ;;
    4) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase4_heldout_rule.py" ;;
    5) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase5_parity.py" ;;
    6) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/phase6_oracle.py" ;;
    plots) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/make_plots.py" ;;
    summary) CASE_SCRIPT="$REPO/src/analysis/mechanism_id/scripts/make_summary_csv.py" ;;
    *) echo "Unknown PHASE=$PHASE"; exit 2 ;;
esac

echo "=== running $CASE_SCRIPT $EXTRA ==="

srun apptainer exec \
    $BINDS \
    --env PYTHONUNBUFFERED=1 \
    --env OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE \
    --env REPRO_ROOT=$REPO \
    "$CONTAINER" \
    python3 -u "$CASE_SCRIPT" $EXTRA

echo "=== DONE $(date) ==="
