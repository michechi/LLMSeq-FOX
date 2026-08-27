#!/bin/bash
#
# Experiment F, Stage 2 (DL): warm-started training on main-paper parity at n=20.
#
# Grid: {Transformer, LSTM} x seed in {0,1,2} = 6 tasks.
# Warm-starts from matched-seed Stage 1 checkpoint in $CKPT_DIR.
#
# Submit with dependency on Stage 1 jobid (see experiment_F_launch.sh) or,
# once Stage 1 is done, directly:
#   cd scripts/slurm/hpc_templates && sbatch experiment_F_stage2_dl.slurm
#
#SBATCH --job-name=expF_stage2_dl
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=01:30:00
#SBATCH --partition=accel
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-5
#SBATCH --output=logs/expF_stage2_dl_%A_%a.out
#SBATCH --error=logs/expF_stage2_dl_%A_%a.err

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

mkdir -p "$OUTPUT_DIR" logs

CONTAINER_ENV="PYTHONUNBUFFERED=1"

BINDS="--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

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

CKPT="$CKPT_DIR/stage1_${MODEL}_seed${SEED}.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint $CKPT missing. Run experiment_F_stage1_dl.slurm first."
    exit 1
fi

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: model=$MODEL seed=$SEED stage=2 (n=20, warm from n=10)"
echo "Init checkpoint: $CKPT"

cd "$REPO_ROOT"

srun apptainer exec --nv $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_dl \
        --variant raw \
        --size main_n20 \
        --model "$MODEL" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 30 \
        --patience 3 \
        --batch_size 64 \
        --init_checkpoint "$CKPT"

echo "=== JOB FINISHED ==="
