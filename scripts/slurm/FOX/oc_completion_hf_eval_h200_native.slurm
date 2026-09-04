#!/bin/bash -l
# Pair evaluation for completed OC Hugging Face runs on FOX.
# Submit through submit_oc_completion_hf_h200_native.sh so this array starts
# only after its corresponding training array has completed successfully.
#
#SBATCH --job-name=oc_hf_eval
#SBATCH --account=ec12
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accel --gpus=nvidia_h200_nvl:1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-13%1
#SBATCH --open-mode=append
#SBATCH --output=logs/oc_hf_eval_%A_%a.out

set -o errexit
set -o nounset
set -o pipefail

CONFIGS=(
    "bert_lora ocdet 9550"
    "bert_lora ocdet 9551"
    "bert_lora ocdet 9552"
    "bert_lora ocnoisy 9550"
    "bert_lora ocnoisy 9551"
    "bert_lora ocnoisy 9552"
    "llama_lora ocdet 9550"
    "llama_lora ocdet 9551"
    "llama_lora ocdet 9552"
    "llama_lora ocnoisy 9550"
    "llama_lora ocnoisy 9551"
    "llama_lora ocnoisy 9552"
    "llama_full ocdet 9550"
    "llama_full ocnoisy 9550"
    "llama_full ocdet 9551"
    "llama_full ocnoisy 9551"
    "llama_full ocdet 9552"
    "llama_full ocnoisy 9552"
)

IDX="$SLURM_ARRAY_TASK_ID"
if (( IDX < 0 || IDX >= ${#CONFIGS[@]} )); then
    echo "ERROR: unsupported array index $IDX" >&2
    exit 2
fi
read -r ARM TASK SEED <<< "${CONFIGS[$IDX]}"

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-pytorch/2.6.0-foss-2024a-cuda-12.6.0-Python-3.12.3
module load nlpl-transformers/4.55.4-foss-2024a-Python-3.12.3
module load nlpl-llmtools/01-foss-2024a-Python-3.12.3
module load nlpl-datasets/3.6.0-foss-2024a-Python-3.12.3
module load nlpl-scikit-bundle/1.6.1-foss-2024a-Python-3.12.3

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(realpath -e -- "$REPO_ROOT")"
if [[ ! -d "$REPO_ROOT/repro/src/oc_completion" ]]; then
    echo "ERROR: REPO_ROOT is not an LLMSeq checkout: $REPO_ROOT" >&2
    exit 2
fi

: "${OC_RUN_ROOT:?ERROR: export OC_RUN_ROOT to a project-storage directory}"
if [[ "$OC_RUN_ROOT" != /* ]]; then
    echo "ERROR: OC_RUN_ROOT must be absolute: $OC_RUN_ROOT" >&2
    exit 2
fi
OC_RUN_ROOT="$(realpath -e -- "$OC_RUN_ROOT")"
HOME_ROOT="$(realpath -e -- "$HOME")"
case "$OC_RUN_ROOT/" in
    "$HOME_ROOT/"*)
        echo "ERROR: full-run outputs must not use quota-limited home storage." >&2
        exit 2
        ;;
esac
export DATA_DIR="$REPO_ROOT/data"
export LLMSEQ_ROOT="$OC_RUN_ROOT"
export HF_HOME="${HF_HOME:-$OC_RUN_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [[ "$HF_HOME" != /* || "$HF_HUB_CACHE" != /* ]]; then
    echo "ERROR: HF_HOME and HF_HUB_CACHE must be absolute paths" >&2
    exit 2
fi
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
HF_HOME="$(realpath -e -- "$HF_HOME")"
HF_HUB_CACHE="$(realpath -e -- "$HF_HUB_CACHE")"
export HF_HOME HF_HUB_CACHE
for cache_path in "$HF_HOME" "$HF_HUB_CACHE"; do
    case "$cache_path/" in
        "$HOME_ROOT/"*)
            echo "ERROR: Hugging Face cache must use project storage: $cache_path" >&2
            exit 2
            ;;
    esac
done
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

TASK_DIR="oc_deterministic"
if [[ "$TASK" == "ocnoisy" ]]; then
    TASK_DIR="oc_noisy"
fi
RUN_DIR="$OC_RUN_ROOT/checkpoints/oc_completion/$TASK_DIR/$ARM/seed_$SEED"
DONE_MARKER="$RUN_DIR/done.json"
BEST_CHECKPOINT="$RUN_DIR/best.pt"
EVAL_MARKER="$RUN_DIR/pair_eval.done"
RESULTS_DIR="$OC_RUN_ROOT/results/matched_completion"
mkdir -p "$RESULTS_DIR"

for required in "$DONE_MARKER" "$BEST_CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: training artifact is missing: $required" >&2
        exit 2
    fi
done

if [[ -f "$EVAL_MARKER" ]]; then
    echo "Pair evaluation already complete: $ARM / $TASK / seed $SEED"
    exit 0
fi

python3 -u - <<'PY'
import peft
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see the allocated H200")
print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"peft={peft.__version__} gpu={torch.cuda.get_device_name(0)}"
)
PY

cd "$REPO_ROOT/repro"

# eval_pairs appends shared CSVs.  Serialize all HF evaluations and recheck
# the per-run marker after acquiring the lock.
exec 8>"$RESULTS_DIR/.hf_pair_eval.lock"
flock 8
if [[ -f "$EVAL_MARKER" ]]; then
    echo "Pair evaluation completed while this task waited for the lock."
    exit 0
fi

python3 -u -m src.oc_completion.eval_pairs \
    --model_kind hf \
    --checkpoint "$BEST_CHECKPOINT" \
    --task "$TASK" \
    --hf_batch_size 256 \
    --threads "$SLURM_CPUS_PER_TASK"

touch "$EVAL_MARKER"
echo "=== PAIR EVALUATION COMPLETE: $ARM / $TASK / seed $SEED ==="
