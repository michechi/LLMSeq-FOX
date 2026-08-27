#!/bin/bash
#
# Experiment A: parity vocabulary sweep, BERT-base side.
#
# Grid: ell in {2,4,8,16,26} x seed in {0,1,2} = 15 tasks, BERT-base
# fine-tuned at |K|=ell/2 (balanced regime), 100K train, 20K val, 20K test.
#
# Assumes the DL array (experiment_A_dl.slurm) already generated the anchor
# CSVs. Submit after the DL array has at least kicked off:
#
#   HF_TOKEN=... sbatch scripts/slurm/hpc_templates/experiment_A_bert.slurm
#
# Outputs land in ${SCRATCH:-/tmp}/results/parity_decomp/.
#
#SBATCH --job-name=experiment_A_bert
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=06:00:00
#SBATCH --partition=accel
#SBATCH --mem=90G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-14
#SBATCH --output=logs/experiment_A_bert_%A_%a.out
#SBATCH --error=logs/experiment_A_bert_%A_%a.err

set -o errexit
set -o nounset

# ============== CACHE + TOKEN ==============
export SCRATCH_CACHE="$SCRATCH/hf_cache_$SLURM_JOB_ID"
mkdir -p "$SCRATCH_CACHE" logs

export HF_HOME="$SCRATCH_CACHE"
export HF_DATASETS_CACHE="$SCRATCH_CACHE"
export XDG_CACHE_HOME="$SCRATCH_CACHE"
export PYTHONUNBUFFERED=1
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:?ERROR: Set HF_TOKEN before submitting}"

export http_proxy=http://10.63.2.48:3128/
export https_proxy=http://10.63.2.48:3128/
export HTTP_PROXY=http://10.63.2.48:3128/
export HTTPS_PROXY=http://10.63.2.48:3128/

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/parity_decomp}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}
# Shared userbase for pip-installed arm64 packages (transformers etc.). First
# task to see it missing installs; others reuse.
BERT_PKG_DIR="${SCRATCH:-/tmp}/python_packages/arm64_pt2506"

mkdir -p "$OUTPUT_DIR" "$DATA_DIR" "$PERMANENT_CACHE" "$BERT_PKG_DIR"

CONTAINER_ENV="HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
CONTAINER_ENV="$CONTAINER_ENV,HF_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$SCRATCH_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUSERBASE=$BERT_PKG_DIR"
CONTAINER_ENV="$CONTAINER_ENV,http_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,https_proxy=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTP_PROXY=http://10.63.2.48:3128/"
CONTAINER_ENV="$CONTAINER_ENV,HTTPS_PROXY=http://10.63.2.48:3128/"

BINDS="--bind $SCRATCH_CACHE:$SCRATCH_CACHE:rw"
BINDS="$BINDS --bind $PERMANENT_CACHE:$PERMANENT_CACHE:rw"
BINDS="$BINDS --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

# ============== GRID ==============
# Varying axes (outer to inner): ELL, SEED.
ELLS=(2 4 8 16 26)
SEEDS=(0 1 2)

N_E=${#ELLS[@]}
N_S=${#SEEDS[@]}
TOTAL=$((N_E * N_S))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL, exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_S ))
ELL_IDX=$(( IDX / N_S ))

ELL=${ELLS[$ELL_IDX]}
SEED=${SEEDS[$SEED_IDX]}
SIZE="binary${ELL}"
K=$(( ELL / 2 ))

# Wait for the DL array to finish generating the anchor CSVs.
SENTINEL="$DATA_DIR/X_train_anchor_l${ELL}_k${K}.csv"
for i in $(seq 1 60); do
    [ -f "$SENTINEL" ] && break
    echo "[rank $SLURM_ARRAY_TASK_ID] waiting for anchor CSV ($SENTINEL)... ($i/60)"
    sleep 30
done
if [ ! -f "$SENTINEL" ]; then
    echo "ERROR: anchor CSV $SENTINEL never appeared. Did the DL array (experiment_A_dl.slurm) run first?"
    exit 1
fi

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: ell=$ELL seed=$SEED size=$SIZE"
echo "Container: $CONTAINER"
echo "Pip user base: $BERT_PKG_DIR"

cd "$REPO_ROOT"

# Packages must have been installed beforehand by
# scripts/slurm/hpc_templates/install_bert_deps.slurm. Fail fast with a clear message
# if the sentinel is missing.
PKG_SENTINEL="$BERT_PKG_DIR/.installed_ok"
if [ ! -f "$PKG_SENTINEL" ]; then
    echo "ERROR: $PKG_SENTINEL not found."
    echo "Run 'sbatch scripts/slurm/hpc_templates/install_bert_deps.slurm' first and wait for it to finish."
    exit 1
fi

srun apptainer exec --nv \
    $BINDS \
    --env "$CONTAINER_ENV" \
    "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_bert \
        --variant raw \
        --size "$SIZE" \
        --seed "$SEED" \
        --model_name google-bert/bert-base-uncased \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$SCRATCH_CACHE" \
        --batch_size 32 \
        --max_length 128 \
        --epochs 20 \
        --patience 3 \
        --lr 2e-5 \
        --early loss

echo "=== JOB FINISHED ==="
