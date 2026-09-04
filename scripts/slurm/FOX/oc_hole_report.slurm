#!/bin/bash -l
# Assemble the Ordered Compliance paper tables, figures, CSVs, and report.
# Submit only after the evaluation array completes; missing/failed cells remain
# explicitly missing rather than being inferred from partial artifacts.
#
#   mkdir -p logs/oc_hole
#   sbatch --dependency=afterok:<evaluation_job> \
#       scripts/slurm/FOX/oc_hole_report.slurm

#SBATCH --job-name=oc_hole_report
#SBATCH --account=ec12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hole/report_%j.out

set -o errexit
set -o nounset
set -o pipefail

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
DATA_DIR="$(realpath -e -- "$DATA_DIR")"
HF_HOME="${HF_HOME:-$DATA_DIR/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
OC_RESULT_ROOT="${OC_RESULT_ROOT:-$REPO_ROOT/results/oc_hole_audit}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/work/projects/ec12/michechi/checkpoints}"
OC_CHECKPOINT_ROOT="${OC_CHECKPOINT_ROOT:-$CHECKPOINT_DIR/oc_hole_audit}"
OC_DATA_ROOT="$DATA_DIR/simulation/oc_hole_audit"

if [[ ! -f "$REPO_ROOT/repro/src/oc_completion/ordered_report.py" ]]; then
    echo "ERROR: invalid REPO_ROOT or missing report builder: $REPO_ROOT" >&2
    exit 2
fi
for required in \
    "$OC_RESULT_ROOT/dataset_manifest.json" \
    "$OC_RESULT_ROOT/noise_manifest.json" \
    "$OC_RESULT_ROOT/hole_manifest.parquet" \
    "$OC_RESULT_ROOT/hole_oracle_labels.npz" \
    "$OC_RESULT_ROOT/strict_pair_manifest.parquet"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing report input: $required" >&2
        exit 2
    fi
done

mkdir -p "$OC_RESULT_ROOT"
export REPO_ROOT DATA_DIR HF_HOME HF_HUB_CACHE CHECKPOINT_DIR OC_CHECKPOINT_ROOT
export LLMSEQ_ROOT="$REPO_ROOT"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$REPO_ROOT/repro"
python3 -u -m src.oc_completion.ordered_report \
    --results-root "$OC_RESULT_ROOT" \
    --checkpoint-root "$OC_CHECKPOINT_ROOT" \
    --data-root "$OC_DATA_ROOT"

if [[ ! -f "$OC_RESULT_ROOT/report_manifest.json" || ! -f "$OC_RESULT_ROOT/report.md" ]]; then
    echo "ERROR: report builder did not produce its required completion artifacts" >&2
    exit 1
fi
echo "[oc_hole_report] complete: $OC_RESULT_ROOT/report.md"
