#!/bin/bash -l
# Ordered Compliance Stage B scratch training on FOX.
#
# Array 0-35 maps model x pi x seed, with seed varying fastest:
#   model_index = task / 12       -> LSTM, Transformer, RNNTransformer
#   pi_index    = (task % 12) / 3 -> 0.0, 0.1, 0.2, 0.3
#   seed_index  = task % 3        -> 9550, 9551, 9552
#
# Submit from the repository root after creating the log directory:
#   mkdir -p logs/oc_hole
#   sbatch scripts/slurm/FOX/oc_hole_train_dl.slurm
#
# Required shared environment contract:
#   REPO_ROOT   repository checkout
#   DATA_DIR    parent data directory containing simulation/oc_hole_audit
#   CHECKPOINT_DIR defaults to /cluster/work/projects/ec12/michechi/checkpoints
# HF_HOME and HF_HUB_CACHE are preserved/exported for a common job environment
# even though scratch training does not access pretrained weights.

#SBATCH --job-name=oc_hole_dl
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=nvidia_h200_nvl:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-35%4
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hole/train_dl_%A_%a.out

set -o errexit
set -o nounset
set -o pipefail

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
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

if [[ ! -f "$REPO_ROOT/repro/src/oc_completion/ordered_train_dl.py" ]]; then
    echo "ERROR: invalid REPO_ROOT or missing ordered scratch trainer: $REPO_ROOT" >&2
    exit 2
fi
for required in \
    "$OC_DATA_ROOT/splits/train.parquet" \
    "$OC_DATA_ROOT/splits/val.parquet" \
    "$OC_DATA_ROOT/splits/test.parquet" \
    "$OC_DATA_ROOT/noise/train.parquet" \
    "$OC_DATA_ROOT/noise/val.parquet" \
    "$OC_DATA_ROOT/noise/test.parquet"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: missing ordered-data input: $required" >&2
        exit 2
    fi
done

mkdir -p "$OC_RESULT_ROOT" "$OC_CHECKPOINT_ROOT"
export REPO_ROOT DATA_DIR HF_HOME HF_HUB_CACHE CHECKPOINT_DIR OC_CHECKPOINT_ROOT
export LLMSEQ_ROOT="$REPO_ROOT"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

MODELS=(LSTM Transformer RNNTransformer)
PIS=(0.0 0.1 0.2 0.3)
SEEDS=(9550 9551 9552)
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( INDEX < 0 || INDEX >= 36 )); then
    echo "ERROR: scratch array index must be in 0-35: $INDEX" >&2
    exit 2
fi
MODEL="${MODELS[$((INDEX / 12))]}"
PI="${PIS[$(((INDEX % 12) / 3))]}"
SEED="${SEEDS[$((INDEX % 3))]}"
PI_TAG="${PI/./p}"
MODEL_DIR="${MODEL,,}"
RUN_DIR="$OC_CHECKPOINT_ROOT/$MODEL_DIR/pi_$PI_TAG/seed_$SEED"

valid_done() {
    python3 - "$RUN_DIR" "$MODEL" "$PI" "$SEED" <<'PY'
import json
import pathlib
import sys

run = pathlib.Path(sys.argv[1])
try:
    done = json.loads((run / "done.json").read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
expected = {
    "status": "complete",
    "model_name": sys.argv[2],
    "noise_pi": float(sys.argv[3]),
    "model_seed": int(sys.argv[4]),
}
required = ("best_standard.pt", "last.pt", "config.json", "history.csv")
ok = all(done.get(key) == value for key, value in expected.items())
ok = ok and all((run / name).is_file() for name in required)
raise SystemExit(0 if ok else 1)
PY
}

echo "[oc_hole_train_dl] index=$INDEX model=$MODEL pi=$PI seed=$SEED"
echo "[oc_hole_train_dl] node=$(hostname) data=$OC_DATA_ROOT run=$RUN_DIR"
if valid_done; then
    echo "[oc_hole_train_dl] valid completed run exists; skipping"
    exit 0
fi

cd "$REPO_ROOT/repro"
python3 -u -m src.oc_completion.ordered_train_dl \
    --model "$MODEL" \
    --noise-pi "$PI" \
    --seed "$SEED" \
    --data-root "$OC_DATA_ROOT" \
    --output-root "$OC_CHECKPOINT_ROOT" \
    --max-epochs 30 \
    --patience 5 \
    --num-workers "$SLURM_CPUS_PER_TASK" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --device cuda \
    --resume

if ! valid_done; then
    echo "ERROR: trainer returned without a valid done.json: $RUN_DIR" >&2
    exit 1
fi
echo "[oc_hole_train_dl] complete: $RUN_DIR"
