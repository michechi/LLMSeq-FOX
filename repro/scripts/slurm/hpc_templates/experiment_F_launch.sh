#!/bin/bash
# Launcher for Experiment F (length curriculum on main-paper parity).
#
# Submits the four SLURM arrays with correct --dependency wiring:
#   Stage 1 DL         (task 0 generates data, all 6 tasks save checkpoints)
#   Stage 1 BERT       (waits on Stage 1 DL's data-gen task)
#   Stage 2 DL         (depends on Stage 1 DL)
#   Stage 2 BERT       (depends on Stage 1 BERT)
#
# Requires: HF_TOKEN set in the environment (for BERT).
# Usage:    bash scripts/slurm/hpc_templates/experiment_F_launch.sh
#
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set in the environment (needed for BERT)." >&2
    echo "       export HF_TOKEN=<your-token> first." >&2
    exit 1
fi

cd "$(dirname "$0")"

echo "Submitting Stage 1 DL..."
S1_DL_ID=$(sbatch --parsable experiment_F_stage1_dl.slurm)
echo "  Stage 1 DL job id: $S1_DL_ID"

echo "Submitting Stage 1 BERT (depends on Stage 1 DL data-gen)..."
S1_BERT_ID=$(sbatch --parsable \
    --dependency=afterok:${S1_DL_ID}_0 \
    experiment_F_stage1_bert.slurm)
echo "  Stage 1 BERT job id: $S1_BERT_ID"

echo "Submitting Stage 2 DL (depends on Stage 1 DL)..."
S2_DL_ID=$(sbatch --parsable \
    --dependency=afterok:${S1_DL_ID} \
    experiment_F_stage2_dl.slurm)
echo "  Stage 2 DL job id: $S2_DL_ID"

echo "Submitting Stage 2 BERT (depends on Stage 1 BERT)..."
S2_BERT_ID=$(sbatch --parsable \
    --dependency=afterok:${S1_BERT_ID} \
    experiment_F_stage2_bert.slurm)
echo "  Stage 2 BERT job id: $S2_BERT_ID"

echo ""
echo "All four arrays submitted. Chain:"
echo "  $S1_DL_ID  (Stage 1 DL)"
echo "  $S1_BERT_ID  (Stage 1 BERT, waits on $S1_DL_ID task 0)"
echo "  $S2_DL_ID  (Stage 2 DL, waits on $S1_DL_ID)"
echo "  $S2_BERT_ID  (Stage 2 BERT, waits on $S1_BERT_ID)"
echo ""
echo "Monitor with:  squeue --me --format='%.10i %.14j %.2t %.6M'"
echo "After both Stage 2 arrays finish, run:"
echo "  sbatch scripts/slurm/hpc_templates/plot_experiment_F.slurm"
