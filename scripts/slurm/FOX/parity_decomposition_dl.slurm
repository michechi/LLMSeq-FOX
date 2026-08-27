#!/bin/bash
#
# Parity decomposition — DL sweep (Transformer, LSTM).
#
# 3 variants × 2 models × 2 sizes × 3 seeds = 36 runs. Use the SLURM array
# index to select the configuration. Submit with:
#
#   sbatch scripts/slurm/FOX/parity_decomposition_dl.slurm
#
# Outputs land in ${SCRATCH}/results/parity_decomp/.
#
#SBATCH --job-name=parity_decomp_dl
#SBATCH --account=ec12
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=a100:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-35
#SBATCH --output=logs/parity_decomp_dl_%A_%a.out

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8

REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/simulation/parity_decomp}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
mkdir -p "$OUTPUT_DIR" logs

# Task 0 builds variant CSVs; other tasks wait to avoid 36-way race.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$DATA_DIR/X_train_parity_raw_400K.csv" ]; then
    echo "[rank $SLURM_ARRAY_TASK_ID] building full variant CSVs in $DATA_DIR"
    python3 -m src.data.parity_variants --build_full \
        --data_dir "${REPO_ROOT}/data/simulation/tested" \
        --output_dir "$DATA_DIR"
else
    for i in $(seq 1 60); do
        [ -f "$DATA_DIR/X_train_parity_raw_400K.csv" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for variant CSVs... ($i/60)"
        sleep 15
    done
fi

# Configuration grid: (variant, model, size, seed)
VARIANTS=("raw" "masked" "bitonly")
MODELS=("Transformer" "LSTM")
SIZES=("40K" "400K")
SEEDS=(0 1 2)

N_V=${#VARIANTS[@]}
N_M=${#MODELS[@]}
N_SIZES=${#SIZES[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL=$((N_V * N_M * N_SIZES * N_SEEDS))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL — exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_SEEDS ))
IDX=$(( IDX / N_SEEDS ))
SIZE_IDX=$(( IDX % N_SIZES ))
IDX=$(( IDX / N_SIZES ))
MODEL_IDX=$(( IDX % N_M ))
VARIANT_IDX=$(( IDX / N_M ))

VARIANT=${VARIANTS[$VARIANT_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
SIZE=${SIZES[$SIZE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[rank $SLURM_ARRAY_TASK_ID] variant=$VARIANT model=$MODEL size=$SIZE seed=$SEED"

python3 -u -m src.experiments.parity_decomposition_dl \
    --variant "$VARIANT" \
    --model "$MODEL" \
    --size "$SIZE" \
    --seed "$SEED" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 30 \
    --patience 3 \
    --batch_size 64
