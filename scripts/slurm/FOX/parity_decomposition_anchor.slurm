#!/bin/bash
#
# Parity decomposition — binary anchor sweep.
#
# ell ∈ {2, 4} × (DL Transformer, LSTM + Llama-3.2-1B) × 3 seeds.
#
# DL tasks: --array=0-11  (2 ell × 2 models × 3 seeds = 12)
# LLM tasks: --array=12-17 (2 ell × 1 model × 3 seeds = 6)
#
# Set --array=0-5 to run only ell=2 controls.
#
#SBATCH --job-name=parity_anchor
#SBATCH --account=ec12
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --partition=accel --gpus=a100:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-17
#SBATCH --output=logs/parity_anchor_%A_%a.out

set -o errexit
set -o nounset

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-transformers/4.47.1-foss-2022b-Python-3.10.8
module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module load nlpl-datasets/3.2.0-foss-2022b-Python-3.10.8
module load nlpl-nlptools/04-foss-2022b-Python-3.10.8
module load nlpl-scikit-bundle/1.3.2-foss-2022b-Python-3.10.8
module load nlpl-bitsandbytes/0.42.0-foss-2022b-Python-3.10.8

REPO_ROOT="${REPO_ROOT:-${HOME}/MIMICIV}"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/simulation/parity_decomp}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CACHE_DIR="${CACHE_DIR:-${SCRATCH:-/tmp}/cache}"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" logs

# Task 0 builds anchors; other tasks wait.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    for ELL in 2 4; do
        K=$(( ELL / 2 ))
        if [ "$ELL" = "2" ]; then K=1; fi
        if [ ! -f "$DATA_DIR/X_train_anchor_l${ELL}_k${K}.csv" ]; then
            echo "[rank $SLURM_ARRAY_TASK_ID] building anchor ell=$ELL k=$K"
            python3 -m src.data.parity_variants --build_anchor \
                --anchor_ell "$ELL" --anchor_k "$K" --output_dir "$DATA_DIR"
        fi
    done
else
    for i in $(seq 1 60); do
        [ -f "$DATA_DIR/X_train_anchor_l2_k1.csv" ] && [ -f "$DATA_DIR/X_train_anchor_l4_k2.csv" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for anchor CSVs... ($i/60)"
        sleep 15
    done
fi

ELLS=(2 4)
DL_MODELS=("Transformer" "LSTM")
SEEDS=(0 1 2)

N_ELL=${#ELLS[@]}
N_DL=${#DL_MODELS[@]}
N_SEEDS=${#SEEDS[@]}

DL_TOTAL=$(( N_ELL * N_DL * N_SEEDS ))        # 12
LLM_TOTAL=$(( N_ELL * N_SEEDS ))              # 6

IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -lt "$DL_TOTAL" ]; then
    # DL: ell / model / seed
    SEED_IDX=$(( IDX % N_SEEDS ))
    IDX=$(( IDX / N_SEEDS ))
    MODEL_IDX=$(( IDX % N_DL ))
    ELL_IDX=$(( IDX / N_DL ))
    ELL=${ELLS[$ELL_IDX]}
    MODEL=${DL_MODELS[$MODEL_IDX]}
    SEED=${SEEDS[$SEED_IDX]}

    echo "[rank $SLURM_ARRAY_TASK_ID] DL anchor ell=$ELL model=$MODEL seed=$SEED"
    python3 -u -m src.experiments.parity_decomposition_dl \
        --variant bitonly \
        --model "$MODEL" \
        --size "binary${ELL}" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 30 --patience 3 --batch_size 64
else
    LLM_IDX=$(( IDX - DL_TOTAL ))
    if [ "$LLM_IDX" -ge "$LLM_TOTAL" ]; then
        echo "IDX $SLURM_ARRAY_TASK_ID out of range — exiting."
        exit 0
    fi
    SEED_IDX=$(( LLM_IDX % N_SEEDS ))
    ELL_IDX=$(( LLM_IDX / N_SEEDS ))
    ELL=${ELLS[$ELL_IDX]}
    SEED=${SEEDS[$SEED_IDX]}

    echo "[rank $SLURM_ARRAY_TASK_ID] LLM anchor ell=$ELL seed=$SEED"
    python3 -u -m src.experiments.parity_decomposition_llm \
        --variant bitonly \
        --size "binary${ELL}" \
        --seed "$SEED" \
        --model_name meta-llama/Llama-3.2-1B \
        --peft \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$CACHE_DIR" \
        --batch_size 64 --max_length 128 \
        --epochs 20 --patience 3 --lr 2e-5 --early loss
fi
