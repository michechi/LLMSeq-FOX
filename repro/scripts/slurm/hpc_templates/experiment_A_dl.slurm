#!/bin/bash
#
# Experiment A: parity vocabulary sweep, DL side (Transformer, LSTM).
#
# Grid: ell in {2,4,8,16,26} x model in {Transformer, LSTM} x seed in {0,1,2}
# = 30 tasks at |K|=ell/2 (balanced regime), 100K train, 20K val, 20K test.
#
# Task 0 also generates the five anchor CSVs the first time through.
# Submit:
#
#   cd scripts/slurm/hpc_templates && sbatch experiment_A_dl.slurm
#
# Smoke-test one task first:
#
#   sbatch --array=0 experiment_A_dl.slurm
#
# Outputs land in ${SCRATCH:-/tmp}/results/parity_decomp/.
#
#SBATCH --job-name=experiment_A_dl
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=04:00:00
#SBATCH --partition=accel
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-29
#SBATCH --output=logs/experiment_A_dl_%A_%a.out
#SBATCH --error=logs/experiment_A_dl_%A_%a.err

set -o errexit
set -o nounset

# ============== CACHE ==============
export PERMANENT_CACHE="${SCRATCH:-/tmp}/cache"
export HF_HOME="$PERMANENT_CACHE"
export HF_DATASETS_CACHE="$PERMANENT_CACHE"
export XDG_CACHE_HOME="$PERMANENT_CACHE"
export PYTHONUNBUFFERED=1

mkdir -p logs "$PERMANENT_CACHE"

# ============== MODULES ==============
module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

# ============== CONFIG ==============
REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/parity_decomp}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

N_TRAIN="${N_TRAIN:-100000}"
N_VAL="${N_VAL:-20000}"
N_TEST="${N_TEST:-20000}"
ANCHOR_ELLS="${ANCHOR_ELLS:-2,4,8,16,26}"

mkdir -p "$OUTPUT_DIR" "$DATA_DIR"

CONTAINER_ENV="HF_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,HF_DATASETS_CACHE=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,XDG_CACHE_HOME=$PERMANENT_CACHE"
CONTAINER_ENV="$CONTAINER_ENV,PYTHONUNBUFFERED=1"

BINDS="--bind $PERMANENT_CACHE:$PERMANENT_CACHE:rw"
BINDS="$BINDS --bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

# ============== ANCHOR DATA BUILD (task 0) ==============
# Task 0 generates anchor CSVs for every requested ell; other tasks wait for
# the largest ell's train CSV to exist.
BIGGEST_ELL=$(echo "$ANCHOR_ELLS" | awk -F',' '{print $NF}')
BIGGEST_K=$(( BIGGEST_ELL / 2 ))
SENTINEL="$DATA_DIR/X_train_anchor_l${BIGGEST_ELL}_k${BIGGEST_K}.csv"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$SENTINEL" ]; then
    echo "[rank $SLURM_ARRAY_TASK_ID] building anchor CSVs for ells=$ANCHOR_ELLS -> $DATA_DIR"
    cd "$REPO_ROOT"
    # Plain apptainer exec (no srun) so the GPU allocation is kept intact for
    # the training srun below. Data build is pure CPU.
    apptainer exec \
        $BINDS \
        --env "$CONTAINER_ENV" \
        "$CONTAINER" \
        python3 -m src.data.parity_variants --build_anchor \
            --anchor_ells "$ANCHOR_ELLS" \
            --anchor_n_train "$N_TRAIN" \
            --anchor_n_val "$N_VAL" \
            --anchor_n_test "$N_TEST" \
            --output_dir "$DATA_DIR"
else
    for i in $(seq 1 60); do
        [ -f "$SENTINEL" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for anchor CSVs ($SENTINEL)... ($i/60)"
        sleep 15
    done
fi

# ============== GRID ==============
# Varying axes (outer to inner): ELL, MODEL, SEED.
ELLS=(2 4 8 16 26)
MODELS=("Transformer" "LSTM")
SEEDS=(0 1 2)

N_E=${#ELLS[@]}
N_M=${#MODELS[@]}
N_S=${#SEEDS[@]}
TOTAL=$((N_E * N_M * N_S))

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL, exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_S ))
IDX=$(( IDX / N_S ))
MODEL_IDX=$(( IDX % N_M ))
ELL_IDX=$(( IDX / N_M ))

ELL=${ELLS[$ELL_IDX]}
MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}
SIZE="binary${ELL}"

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Config: ell=$ELL model=$MODEL seed=$SEED size=$SIZE"
echo "Container: $CONTAINER"

echo "=== HOST GPU STATE ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
nvidia-smi -L 2>&1 | head -5 || echo "host: nvidia-smi -L failed"

cd "$REPO_ROOT"

srun apptainer exec --nv \
    $BINDS \
    --env "$CONTAINER_ENV,MODEL=$MODEL,SIZE=$SIZE,SEED=$SEED,DATA_DIR=$DATA_DIR,OUTPUT_DIR=$OUTPUT_DIR" \
    "$CONTAINER" \
    bash -c '
echo "=== CONTAINER GPU STATE ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L 2>&1 | head -5 || echo "container: nvidia-smi -L failed"
python3 -c "import torch; print(\"torch.cuda.is_available:\", torch.cuda.is_available(), \"device_count:\", torch.cuda.device_count())" || echo "container: torch import failed"
python3 -c "import numpy, pandas, sklearn; print(\"numpy\", numpy.__version__, \"pandas\", pandas.__version__, \"sklearn\", sklearn.__version__)" 2>&1 | head -5 || echo "container: numpy/pandas/sklearn import failed"
python3 -c "import transformers; print(\"transformers\", transformers.__version__)" 2>&1 | head -3 || echo "container: transformers import failed"
echo "=== STARTING TRAINING ==="
exec python3 -u -m src.experiments.parity_decomposition_dl \
        --variant raw \
        --model "$MODEL" \
        --size "$SIZE" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 30 \
        --patience 3 \
        --batch_size 64
'

echo "=== JOB FINISHED ==="
