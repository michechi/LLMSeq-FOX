#!/bin/bash
#
# Experiment G: length-difficulty curve at the balanced parity regime
# (l=26, |K|=13). DL only: Transformer + LSTM x 3 seeds x 3 lengths {10,15,30}.
#
# Grid: 2 archs * 3 seeds * 3 lengths = 18 tasks (array 0-17).
# Task 0 generates the anchor_l26_k13_n{10,15,30} CSVs; siblings file-poll.
# n=20 is reused from the existing anchor_l26_k13_*.json runs and is NOT
# rebuilt or re-trained here.
#
# Submit:  cd scripts/slurm/hpc_templates && sbatch experiment_G_length_dl.slurm
#
#SBATCH --job-name=expG_len_dl
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --time=01:30:00
#SBATCH --partition=accel
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --array=0-17
#SBATCH --output=logs/expG_len_dl_%A_%a.out
#SBATCH --error=logs/expG_len_dl_%A_%a.err

set -o errexit
set -o nounset

export PYTHONUNBUFFERED=1

module purge
# module load <YOUR_GPU_MODULE>  # adapt to your cluster
if ! command -v apptainer &> /dev/null; then module load Apptainer; fi

REPO_ROOT="${REPO_ROOT:-${REPRO_ROOT:-$PWD}}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data/simulation/parity_decomp}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH:-/tmp}/results/parity_decomp}"
CONTAINER="${CONTAINER:?set CONTAINER=/path/to/your/pytorch.sif}

mkdir -p "$OUTPUT_DIR" "$DATA_DIR" logs

CONTAINER_ENV="PYTHONUNBUFFERED=1"

BINDS="--bind ${SCRATCH:-/tmp}:${SCRATCH:-/tmp}:rw"
BINDS="$BINDS --bind ${REPRO_ROOT:-$PWD}:${REPRO_ROOT:-$PWD}:rw"

# ============== DATA PREP ==============
# Task 0 builds anchor CSVs at n in {10,15,30}; siblings poll for the last
# sentinel (n=30) before proceeding.
SENTINEL="$DATA_DIR/X_train_anchor_l26_k13_n30.csv"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ] && [ ! -f "$SENTINEL" ]; then
    echo "[rank 0] building anchor_l26_k13_n{10,15,30} CSVs in $DATA_DIR"
    cd "$REPO_ROOT"
    srun apptainer exec --nv $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
        python3 -m src.data.parity_variants --build_anchor \
            --anchor_ell 26 --anchor_k 13 --anchor_lens 10,15,30 \
            --anchor_n_train 100000 \
            --anchor_n_val 20000 \
            --anchor_n_test 20000 \
            --output_dir "$DATA_DIR"
else
    for i in $(seq 1 80); do
        [ -f "$SENTINEL" ] && break
        echo "[rank $SLURM_ARRAY_TASK_ID] waiting for anchor CSVs... ($i/80)"
        sleep 15
    done
fi

# ============== GRID ==============
MODELS=("Transformer" "LSTM")
SEEDS=(0 1 2)
NS=(10 15 30)

N_M=${#MODELS[@]}
N_S=${#SEEDS[@]}
N_N=${#NS[@]}
TOTAL=$(( N_M * N_S * N_N ))   # 18

if [ "$SLURM_ARRAY_TASK_ID" -ge "$TOTAL" ]; then
    echo "ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID >= TOTAL $TOTAL, exiting."
    exit 0
fi

IDX=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( IDX % N_S ))
IDX=$(( IDX / N_S ))
N_IDX=$(( IDX % N_N ))
MODEL_IDX=$(( IDX / N_N ))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}
N=${NS[$N_IDX]}

echo "=== JOB CONFIGURATION ==="
echo "Job: $SLURM_JOB_ID / Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: model=$MODEL seed=$SEED n=$N (l=26, |K|=13)"

cd "$REPO_ROOT"

srun apptainer exec --nv $BINDS --env "$CONTAINER_ENV" "$CONTAINER" \
    python3 -u -m src.experiments.parity_decomposition_dl \
        --variant bitonly \
        --size "binary26k13_n${N}" \
        --model "$MODEL" \
        --seed "$SEED" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs 30 \
        --patience 3 \
        --batch_size 64

echo "=== JOB FINISHED ==="
