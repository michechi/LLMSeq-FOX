#!/bin/bash -l
# Ordered Compliance Stage C/D pretrained LoRA training on FOX H200 GPUs.
#
# Time-constrained endpoint array (0-3), using OC_HF_MODEL_SEED=9950:
#   0-1  BERT-LoRA: pi 0.0,0.3
#   2-3  Llama-3.2-1B-LoRA: pi 0.0,0.3
#
# Commands:
#   mkdir -p logs/oc_hole
#   sbatch scripts/slurm/FOX/oc_hole_train_hf.slurm
#
# REPO_ROOT, DATA_DIR, HF_HOME, and HF_HUB_CACHE must point to persistent
# project storage. CHECKPOINT_DIR defaults to the ec12 project checkpoint
# directory. Llama access must already be cached or authenticated.

#SBATCH --job-name=oc_hole_hf
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hole/train_hf_%A_%a.out

set -o errexit
set -o nounset
set -o pipefail

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
DATA_DIR="$(realpath -e -- "$DATA_DIR")"
: "${HF_HOME:?Set HF_HOME to persistent project storage before submission}"
: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to persistent project storage before submission}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
HF_HOME="$(realpath -e -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -e -- "$HF_HUB_CACHE")"
OC_RESULT_ROOT="${OC_RESULT_ROOT:-$REPO_ROOT/results/oc_hole_audit}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/work/projects/ec12/michechi/checkpoints}"
OC_CHECKPOINT_ROOT="${OC_CHECKPOINT_ROOT:-$CHECKPOINT_DIR/oc_hole_audit}"
OC_HF_MODEL_SEED="${OC_HF_MODEL_SEED:-9950}"
OC_DATA_ROOT="$DATA_DIR/simulation/oc_hole_audit"

if [[ ! -f "$REPO_ROOT/repro/src/oc_completion/ordered_train_hf.py" ]]; then
    echo "ERROR: invalid REPO_ROOT or missing ordered HF trainer: $REPO_ROOT" >&2
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
export OC_HF_MODEL_SEED
export LLMSEQ_ROOT="$REPO_ROOT"
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

CONFIGS=(
    "bert_lora 0.0"
    "bert_lora 0.3"
    "llama_lora 0.0"
    "llama_lora 0.3"
)
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( INDEX < 0 || INDEX >= ${#CONFIGS[@]} )); then
    echo "ERROR: HF array index must be in 0-3: $INDEX" >&2
    exit 2
fi
read -r ARM PI <<< "${CONFIGS[$INDEX]}"
SEED="$OC_HF_MODEL_SEED"
PI_TAG="${PI/./p}"
RUN_DIR="$OC_CHECKPOINT_ROOT/$ARM/pi_$PI_TAG/seed_$SEED"

valid_done() {
    python3 - "$RUN_DIR" "$ARM" "$PI" "$SEED" <<'PY'
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
    "arm": sys.argv[2],
    "noise_pi": float(sys.argv[3]),
    "model_seed": int(sys.argv[4]),
    "selected_checkpoint": "best_standard.pt",
}
required = (
    "best_standard.pt", "last.pt", "config.json", "history.csv",
    "tokenization_audit.json",
)
ok = all(done.get(key) == value for key, value in expected.items())
ok = ok and all((run / name).is_file() for name in required)
raise SystemExit(0 if ok else 1)
PY
}

echo "[oc_hole_train_hf] index=$INDEX arm=$ARM pi=$PI seed=$SEED"
echo "[oc_hole_train_hf] node=$(hostname) data=$OC_DATA_ROOT run=$RUN_DIR"
if valid_done; then
    echo "[oc_hole_train_hf] valid completed run exists; skipping"
    exit 0
fi

python3 - <<'PY'
import accelerate
import peft
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated GPU")
print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"peft={peft.__version__} accelerate={accelerate.__version__} "
    f"gpu={torch.cuda.get_device_name(0)}"
)
PY

cd "$REPO_ROOT/repro"
python3 -u -m src.oc_completion.ordered_train_hf \
    --arm "$ARM" \
    --pi "$PI" \
    --model-seed "$SEED" \
    --data-root "$OC_DATA_ROOT" \
    --checkpoint-root "$OC_CHECKPOINT_ROOT" \
    --hf-cache "$HF_HUB_CACHE" \
    --max-epochs 20 \
    --patience 3 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --threads "$SLURM_CPUS_PER_TASK" \
    --resume

if ! valid_done; then
    echo "ERROR: trainer returned without a valid done.json: $RUN_DIR" >&2
    exit 1
fi
echo "[oc_hole_train_hf] complete: $RUN_DIR"
