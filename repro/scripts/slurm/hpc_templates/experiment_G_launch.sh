#!/bin/bash
# Launcher for Experiment G (length-difficulty curve at the balanced parity
# regime: l=26, |K|=13, n in {10,15,30}).
#
# Submits two arrays with correct --dependency wiring:
#   DL   (Transformer + LSTM)  -- task 0 generates anchor CSVs at all 3 lengths
#   BERT                       -- waits on DL task 0 for the CSVs
#
# After both arrays finish, run:
#   sbatch scripts/slurm/hpc_templates/plot_experiment_G.slurm
# to (re-)aggregate JSONs into the figure / CSV / LaTeX outputs under
# analysis/experiment_G/figures/.
#
# Requires: HF_TOKEN exported in the environment (for BERT).
#
# Usage:    bash scripts/slurm/hpc_templates/experiment_G_launch.sh
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set in the environment (needed for BERT)." >&2
    echo "       export HF_TOKEN=<your-token> first." >&2
    exit 1
fi

cd "$(dirname "$0")"

echo "Submitting Exp G length-DL array..."
DL_ID=$(sbatch --parsable experiment_G_length_dl.slurm)
echo "  DL job id: $DL_ID"

echo "Submitting Exp G length-BERT array (depends on DL task 0 for CSVs)..."
BERT_ID=$(sbatch --parsable \
    --dependency=afterok:${DL_ID}_0 \
    experiment_G_length_bert.slurm)
echo "  BERT job id: $BERT_ID"

echo ""
echo "Both arrays submitted. Chain:"
echo "  $DL_ID    (DL,  18 tasks, generates CSVs on task 0)"
echo "  $BERT_ID  (BERT, 9 tasks, waits on DL task 0)"
echo ""
echo "Monitor with:  squeue --me --format='%.10i %.14j %.2t %.6M'"
echo "After both arrays finish, run:"
echo "  sbatch scripts/slurm/hpc_templates/plot_experiment_G.slurm"
