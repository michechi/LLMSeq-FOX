#!/bin/bash
#
# Experiment F, Stage 1 (DL): cold training on main-paper parity at n=10.
#
# Grid: {Transformer, LSTM} x seed in {0,1,2} = 6 tasks.
# Task 0 generates the n=10 and n=20 main-paper parity CSVs; siblings wait.
# Saves each run's best-state checkpoint to $CKPT_DIR for Stage 2 warm-start.
#
# Submit:  cd scripts/slurm/hpc_templates && sbatch experiment_F_stage1_dl.slurm
#
#SBATCH --job-name=expF_stage1_dl
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=01:30:00
#SBATCH --partition=accel
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-5
#SBATCH --output=logs/expF_stage1_dl_%A_%a.out
#SBATCH --error=logs/expF_stage1_dl_%A_%a.err

set -o errexit
set -o nounset

export PYTHONUNBUFFERED=1

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/curriculum}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CKPT_DIR="${CKPT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp/ckpt_expF}"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" "$DATA_DIR" logs

CONTAINER_ENV="PYTHONUNBUFFERED=1"

BINDS="--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

# ============== DATA PREP ==============
# Task 0 generates both n=10 and n=20 CSVs; siblings file-poll.
N10_SENTINEL="$DATA_DIR/X_train_main_parity_n10.csv"
N20_SENTINEL="$DATA_DIR/X_train_main_parity_n20.csv"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$N20_SENTINEL" ]; then
    echo "[rank 0] building main-parity n=10 and n=20 CSVs in $DATA_DIR"
    cd "$REPO_ROOT"
    srun apptainer exec --nv $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
        python3 -m src.data.main_parity_variants --n 10 --n 20 \
            --n_train 100000 --n_val 20000 --n_test 20000 \
            --output_dir "$DATA_DIR"
else
    for i in $(seq 1 80); do
        [ -f "$N10_SENTINEL" ] && [ -f "$N20_SENTINEL" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for curriculum CSVs... ($i/80)"
        sleep 15
    done
fi

# ============== GRID ==============
MODELS=("Transformer" "LSTM")
SEEDS=(0 1 2)

N_M=${#MODELS[@]}
N_S=${#SEEDS[@]}
TOTAL=$((N_M * N_S))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL, exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_S ))
MODEL_IDX=$(( IDX / N_S ))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: model=$MODEL seed=$SEED stage=1 (n=10)"

cd "$REPO_ROOT"

srun apptainer exec --nv $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_dl \
        --variant raw \
        --size main_n10 \
        --model "$MODEL" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 30 \
        --patience 3 \
        --batch_size 64 \
        --save_checkpoint "$CKPT_DIR/stage1_${MODEL}_seed${SEED}.pt"

echo "=== JOB FINISHED ==="
