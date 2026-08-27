#!/bin/bash
#
# Experiment C: binary parity regime sweep at ell=2.
#
# Grid: n in {10, 20} x regime in {streaming, fixed} x seed in {0..9} = 40 tasks.
# Mirrors the reviewer's reference setup (sBLg): 2-layer TransformerEncoder,
# d_model=64, nhead=8, no positional encoding, CLS-token read, BCE, Adam 1e-3,
# 1000 steps, batch 1024.
#
# Submit:
#   cd scripts/slurm/hpc_templates && sbatch experiment_C.slurm
#
#SBATCH --job-name=experiment_C
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=00:30:00
#SBATCH --partition=accel
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-39
#SBATCH --output=logs/experiment_C_%A_%a.out
#SBATCH --error=logs/experiment_C_%A_%a.err

set -o errexit
set -o nounset

export PYTHONUNBUFFERED=1

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_streaming}"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

mkdir -p "$OUTPUT_DIR" logs

CONTAINER_ENV="PYTHONUNBUFFERED=1"

BINDS="--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

# ============== GRID ==============
# Outer-to-inner: N, REGIME, SEED.
NS=(10 20)
REGIMES=("streaming" "fixed")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

N_N=${#NS[@]}
N_R=${#REGIMES[@]}
N_S=${#SEEDS[@]}
TOTAL=$((N_N * N_R * N_S))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL, exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_S ))
IDX=$(( IDX / N_S ))
REGIME_IDX=$(( IDX % N_R ))
N_IDX=$(( IDX / N_R ))

N=${NS[$N_IDX]}
REGIME=${REGIMES[$REGIME_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: n=$N regime=$REGIME seed=$SEED"
echo "Container: $CONTAINER"

cd "$REPO_ROOT"

srun apptainer exec --nv \
    $BINDS \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.experiments.parity_streaming \
        --n "$N" \
        --regime "$REGIME" \
        --n_train 100000 \
        --steps 1000 \
        --batch_size 1024 \
        --lr 0.001 \
        --val_samples 4096 \
        --val_every 100 \
        --test_samples 10000 \
        --seed "$SEED" \
        --output_dir "$OUTPUT_DIR"

echo "=== JOB FINISHED ==="
