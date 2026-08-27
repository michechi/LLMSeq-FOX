#!/bin/bash -l
#SBATCH --job-name=mimic_aki_h200
#SBATCH --account=ec12
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/mimic_aki_h200_%j.out

set -o errexit
set -o nounset

# ============== MODULES ==============
module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-nlptools/01-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
AKI_RUN_ROOT="${AKI_RUN_ROOT:-$REPO_ROOT/data/processed/mimic_aki/aki-audit-20260812-fox-h200-v3}"

CONFIG="$AKI_RUN_ROOT/protocol/mimic_aki_fox_h200_smd015_amendment_v3.yaml"
PREPARED_DIR="$AKI_RUN_ROOT/prepared"
OUTPUT_DIR="$AKI_RUN_ROOT/experiment"

export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Run root: $AKI_RUN_ROOT"
echo "Models: logistic_regression xgboost lstm transformer"

cd "$REPO_ROOT"

python3 -u -m src.mimic.aki train \
    --config "$CONFIG" \
    --prepared-dir "$PREPARED_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --models logistic_regression xgboost lstm transformer

echo "=== JOB FINISHED ==="
