#!/bin/bash -l
#SBATCH --job-name=mimic-aki-h200-v3
#SBATCH --partition=accel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue

# Train-only MIMIC-IV AKI v3 audit on FOX. Extraction and preparation must
# already have completed on the CPU VM. Submit only with the companion wrapper.

set -o errexit
set -o nounset
set -o pipefail
umask 077

readonly CONFIG_NAME="mimic_aki_fox_h200_smd015_amendment_v3.yaml"
readonly SOURCE_MANIFEST_NAME="mimic_aki_fox_h200_smd015_amendment_v3.sources.sha256"
readonly TRANSFER_MANIFEST_NAME="cpu-to-fox-transfer.sha256"
readonly EXPECTED_CONFIG_BYTE_SHA256="75d259be65637112bf6e50f31e54c63b572812f5bc69cd24d5facf712d000bdc"
readonly EXPECTED_CONFIG_SEMANTIC_SHA256="d27681de235eaa6d0421f3720b6253c2dc138a046be5c536d6ba7f498edf9345"

: "${SLURM_JOB_ID:?ERROR: submit this workflow through sbatch}"
: "${SLURM_JOB_PARTITION:?ERROR: missing Slurm partition metadata}"
: "${SLURM_JOB_GPUS:?ERROR: Slurm did not identify the allocated GPU}"
: "${SLURM_CPUS_PER_TASK:?ERROR: missing Slurm CPU allocation metadata}"
: "${AKI_REPO:?ERROR: missing approved repository path}"
: "${AKI_RUN_ROOT:?ERROR: missing transferred private v3 run root}"
: "${AKI_SLURM_LOG_DIR:?ERROR: missing private Slurm log path}"
: "${AKI_CONTAINER:?ERROR: missing pinned ARM64 Apptainer image}"
: "${AKI_CONTAINER_SHA256:?ERROR: missing container SHA-256}"
: "${AKI_SOURCE_MANIFEST_SHA256:?ERROR: missing source manifest SHA-256}"
: "${AKI_TRANSFER_MANIFEST:?ERROR: missing transfer checksum manifest}"
: "${AKI_TRANSFER_MANIFEST_SHA256:?ERROR: missing transfer manifest SHA-256}"

fail() {
    echo "stage=workflow status=FAIL reason=$1" >&2
    exit 2
}

require_absolute() {
    [[ "$1" = /* ]] || fail "$2-must-be-absolute"
}

reject_symlink() {
    [[ ! -L "$1" ]] || fail "$2-must-not-be-a-symlink"
}

require_private() {
    if find "$1" -maxdepth 0 -perm /0077 -print -quit | grep -q .; then
        fail "$2-has-group-or-world-permissions"
    fi
}

is_within() {
    case "$1/" in
        "$2/"*) return 0 ;;
        *) return 1 ;;
    esac
}

for variable_name in \
    AKI_REPO \
    AKI_RUN_ROOT \
    AKI_SLURM_LOG_DIR \
    AKI_CONTAINER \
    AKI_TRANSFER_MANIFEST; do
    require_absolute "${!variable_name}" "$variable_name"
    reject_symlink "${!variable_name}" "$variable_name"
done

AKI_REPO="$(realpath -e -- "$AKI_REPO")"
AKI_RUN_ROOT="$(realpath -e -- "$AKI_RUN_ROOT")"
AKI_SLURM_LOG_DIR="$(realpath -e -- "$AKI_SLURM_LOG_DIR")"
AKI_CONTAINER="$(realpath -e -- "$AKI_CONTAINER")"
AKI_TRANSFER_MANIFEST="$(realpath -e -- "$AKI_TRANSFER_MANIFEST")"

[[ -d "$AKI_REPO" ]] || fail "repository-path-is-not-a-directory"
[[ -d "$AKI_RUN_ROOT" ]] || fail "run-root-is-not-a-directory"
[[ -d "$AKI_SLURM_LOG_DIR" ]] || fail "Slurm-log-path-is-not-a-directory"
[[ -f "$AKI_CONTAINER" ]] || fail "container-path-is-not-a-file"
[[ -f "$AKI_TRANSFER_MANIFEST" ]] || fail "transfer-manifest-is-not-a-file"
[[ ! -e "$AKI_REPO/data" && ! -e "$AKI_REPO/.venv" \
    && ! -e "$AKI_REPO/sitecustomize.py" \
    && ! -e "$AKI_REPO/usercustomize.py" ]] \
    || fail "FOX-source-tree-is-not-a-clean-data-free-bundle"
[[ "$AKI_TRANSFER_MANIFEST" \
    = "$AKI_RUN_ROOT/protocol/$TRANSFER_MANIFEST_NAME" ]] \
    || fail "transfer-manifest-path-is-not-canonical"

is_within "$AKI_RUN_ROOT" "$AKI_REPO" \
    && fail "run-root-must-be-outside-repository"
is_within "$AKI_REPO" "$AKI_RUN_ROOT" \
    && fail "repository-must-be-outside-run-root"
is_within "$AKI_SLURM_LOG_DIR" "$AKI_REPO" \
    && fail "Slurm-logs-must-be-outside-repository"
is_within "$AKI_REPO" "$AKI_SLURM_LOG_DIR" \
    && fail "repository-must-be-outside-Slurm-logs"
is_within "$AKI_RUN_ROOT" "$AKI_SLURM_LOG_DIR" \
    && fail "run-root-and-Slurm-logs-must-be-disjoint"
is_within "$AKI_SLURM_LOG_DIR" "$AKI_RUN_ROOT" \
    && fail "run-root-and-Slurm-logs-must-be-disjoint"
case "$AKI_RUN_ROOT/" in
    /tmp/*|/var/tmp/*) fail "run-root-must-use-persistent-storage" ;;
esac

require_private "$AKI_RUN_ROOT" run-root
require_private "$AKI_SLURM_LOG_DIR" Slurm-log-directory
require_private "$AKI_TRANSFER_MANIFEST" transfer-manifest
if find "$AKI_RUN_ROOT" -type l -print -quit | grep -q .; then
    fail "transferred-run-tree-contains-a-symbolic-link"
fi
if find "$AKI_RUN_ROOT" -type f \( \
    -name labevents.csv -o -name labevents.csv.gz -o \
    -name admissions.csv -o -name admissions.csv.gz -o \
    -name patients.csv -o -name patients.csv.gz -o \
    -name d_labitems.csv -o -name d_labitems.csv.gz \
    \) -print -quit | grep -q .; then
    fail "transferred-run-tree-contains-a-raw-MIMIC-source-table"
fi
if find "$AKI_RUN_ROOT" ! -type d ! -type f -print -quit | grep -q .; then
    fail "transferred-run-tree-has-an-unsupported-entry-type"
fi
if find "$AKI_RUN_ROOT" -perm /0077 -print -quit | grep -q .; then
    fail "transferred-run-tree-has-group-or-world-permissions"
fi

for required_directory in protocol cohort prepared logs tmp; do
    [[ -d "$AKI_RUN_ROOT/$required_directory" ]] \
        || fail "transferred-run-root-is-incomplete"
done
top_level_count=0
while IFS= read -r -d '' top_level_entry; do
    top_level_count=$((top_level_count + 1))
    case "$(basename -- "$top_level_entry")" in
        protocol|cohort|prepared|logs|tmp) ;;
        *) fail "transferred-run-root-has-an-unexpected-top-level-entry" ;;
    esac
done < <(find "$AKI_RUN_ROOT" -mindepth 1 -maxdepth 1 -print0)
[[ "$top_level_count" -eq 5 ]] \
    || fail "transferred-run-root-top-level-set-is-not-exact"
[[ ! -e "$AKI_RUN_ROOT/experiment" ]] \
    || fail "experiment-output-already-exists"
[[ -z "$(find "$AKI_RUN_ROOT/tmp" -mindepth 1 -print -quit)" ]] \
    || fail "transferred-temporary-directory-is-not-empty"

for fox_output in \
    fox-preflight.console.log \
    fox-environment-and-cuda-smoke.console.log \
    fox-unit-tests.console.log \
    fox-synthetic-e2e.console.log \
    fox-validate-config.console.log \
    fox-pretraining-integrity-gate.console.log \
    fox-train.console.log \
    fox-train.gpu-telemetry.csv \
    fox-train.gpu-telemetry.stderr.log \
    fox-posttraining-integrity-gate.console.log \
    fox-completed.log; do
    [[ ! -e "$AKI_RUN_ROOT/logs/$fox_output" ]] \
        || fail "FOX-output-log-already-exists"
done
[[ ! -e "$AKI_RUN_ROOT/protocol/fox-training-output.sha256" ]] \
    || fail "FOX-output-manifest-already-exists"

[[ "$AKI_CONTAINER_SHA256" =~ ^[[:xdigit:]]{64}$ ]] \
    || fail "container-SHA256-is-invalid"
[[ "$AKI_SOURCE_MANIFEST_SHA256" =~ ^[[:xdigit:]]{64}$ ]] \
    || fail "source-manifest-SHA256-is-invalid"
[[ "$AKI_TRANSFER_MANIFEST_SHA256" =~ ^[[:xdigit:]]{64}$ ]] \
    || fail "transfer-manifest-SHA256-is-invalid"

config_source="$AKI_REPO/configs/$CONFIG_NAME"
frozen_config="$AKI_RUN_ROOT/protocol/$CONFIG_NAME"
source_manifest="$AKI_REPO/configs/$SOURCE_MANIFEST_NAME"
frozen_source_manifest="$AKI_RUN_ROOT/protocol/$SOURCE_MANIFEST_NAME"
gate_source="$AKI_REPO/scripts/mimic_aki_integrity_gate.py"
job_source="$AKI_REPO/scripts/slurm/FOX/mimic_aki_h200_v3.slurm"
submit_source="$AKI_REPO/scripts/slurm/FOX/submit_mimic_aki_h200_v3.sh"
for required_file in \
    "$AKI_REPO/src/mimic/aki/__main__.py" \
    "$AKI_REPO/scripts/mimic_aki_verify_source_manifest.py" \
    "$AKI_REPO/scripts/mimic_aki_verify_transfer_manifest.py" \
    "$config_source" \
    "$frozen_config" \
    "$source_manifest" \
    "$frozen_source_manifest" \
    "$gate_source" \
    "$job_source" \
    "$submit_source" \
    "$AKI_RUN_ROOT/protocol/mimic_aki_integrity_gate.py" \
    "$AKI_RUN_ROOT/protocol/mimic_aki_verify_transfer_manifest.py" \
    "$AKI_RUN_ROOT/protocol/mimic_aki_h200_v3.slurm" \
    "$AKI_RUN_ROOT/protocol/submit_mimic_aki_h200_v3.sh"; do
    [[ -f "$required_file" && ! -L "$required_file" ]] \
        || fail "required-source-or-provenance-file-is-missing"
done

sha256_of() {
    sha256sum "$1" | awk '{print $1}'
}

verify_sources() {
    [[ "$(sha256_of "$source_manifest")" = "$AKI_SOURCE_MANIFEST_SHA256" ]] \
        || fail "source-manifest-SHA256-mismatch"
    [[ "$(sha256_of "$frozen_source_manifest")" \
        = "$AKI_SOURCE_MANIFEST_SHA256" ]] \
        || fail "frozen-source-manifest-SHA256-mismatch"
    cmp -s "$source_manifest" "$frozen_source_manifest" \
        || fail "source-and-frozen-source-manifests-differ"
    (cd "$AKI_REPO" && sha256sum --quiet -c "$source_manifest") \
        >/dev/null 2>&1 \
        || fail "source-files-do-not-match-pinned-manifest"
    python3 -B "$AKI_REPO/scripts/mimic_aki_verify_source_manifest.py" \
        --repo-root "$AKI_REPO" --manifest "$source_manifest" \
        >/dev/null 2>&1 || fail "source-manifest-exact-verification-failed"
    cmp -s "$gate_source" \
        "$AKI_RUN_ROOT/protocol/mimic_aki_integrity_gate.py" \
        || fail "integrity-gate-differs-from-frozen-copy"
    cmp -s "$job_source" "$AKI_RUN_ROOT/protocol/mimic_aki_h200_v3.slurm" \
        || fail "Slurm-job-differs-from-frozen-copy"
    cmp -s "${BASH_SOURCE[0]}" \
        "$AKI_RUN_ROOT/protocol/mimic_aki_h200_v3.slurm" \
        || fail "executing-Slurm-spool-script-differs-from-frozen-copy"
    cmp -s "$submit_source" \
        "$AKI_RUN_ROOT/protocol/submit_mimic_aki_h200_v3.sh" \
        || fail "submit-wrapper-differs-from-frozen-copy"
}

verify_protocol() {
    [[ "$(sha256_of "$config_source")" = "$EXPECTED_CONFIG_BYTE_SHA256" ]] \
        || fail "source-protocol-byte-SHA256-mismatch"
    [[ "$(sha256_of "$frozen_config")" = "$EXPECTED_CONFIG_BYTE_SHA256" ]] \
        || fail "frozen-protocol-byte-SHA256-mismatch"
    cmp -s "$config_source" "$frozen_config" \
        || fail "source-and-frozen-protocols-differ"
}

verify_container() {
    [[ "$(sha256_of "$AKI_CONTAINER")" = "$AKI_CONTAINER_SHA256" ]] \
        || fail "container-SHA256-mismatch"
}

verify_locked_state() {
    verify_sources
    verify_protocol
    verify_container
}

verify_locked_state
if find "$AKI_REPO/src" "$AKI_REPO/tests" \
    \( -type d -name __pycache__ -o -type f -name '*.pyc' \) \
    -print -quit | grep -q .; then
    fail "source-tree-contains-Python-bytecode-cache"
fi
[[ "$(sha256_of "$AKI_TRANSFER_MANIFEST")" \
    = "$AKI_TRANSFER_MANIFEST_SHA256" ]] \
    || fail "transfer-manifest-SHA256-mismatch"

(cd "$AKI_RUN_ROOT" && sha256sum --quiet -c "$AKI_TRANSFER_MANIFEST") \
    >/dev/null 2>&1 \
    || fail "transferred-files-do-not-match-checksum-manifest"
python3 -B "$AKI_REPO/scripts/mimic_aki_verify_transfer_manifest.py" \
    --run-root "$AKI_RUN_ROOT" --manifest "$AKI_TRANSFER_MANIFEST" \
    >/dev/null 2>&1 || fail "exact-transfer-manifest-verification-failed"

[[ "$SLURM_JOB_PARTITION" = "accel" ]] || fail "unexpected-Slurm-partition"
[[ "$SLURM_JOB_GPUS" != *,* && -n "$SLURM_JOB_GPUS" ]] \
    || fail "allocation-must-contain-exactly-one-GPU"
: "${CUDA_VISIBLE_DEVICES:?ERROR: Slurm did not set CUDA_VISIBLE_DEVICES}"
[[ "$CUDA_VISIBLE_DEVICES" != *,* ]] \
    || fail "exactly-one-CUDA-device-must-be-visible"
[[ "$(uname -m)" = "aarch64" ]] || fail "H200-node-must-be-ARM64"

allocated_gpu="$SLURM_JOB_GPUS"
gpu_record="$(nvidia-smi -i "$allocated_gpu" \
    --query-gpu=uuid,name --format=csv,noheader,nounits)" \
    || fail "allocated-GPU-query-failed"
[[ "$(printf '%s\n' "$gpu_record" | wc -l)" -eq 1 ]] \
    || fail "allocated-GPU-query-returned-multiple-devices"
IFS=',' read -r allocated_gpu_uuid allocated_gpu_name unexpected_gpu_field \
    <<< "$gpu_record"
allocated_gpu_uuid="${allocated_gpu_uuid//[[:space:]]/}"
allocated_gpu_name="$(printf '%s' "$allocated_gpu_name" | xargs)"
[[ -z "${unexpected_gpu_field:-}" && -n "$allocated_gpu_uuid" ]] \
    || fail "allocated-GPU-record-is-invalid"
[[ "${allocated_gpu_name^^}" = *H200* ]] \
    || fail "allocated-GPU-is-not-H200-or-GH200"

if command -v module >/dev/null 2>&1; then
    module --quiet purge >/dev/null 2>&1 || module purge >/dev/null 2>&1
fi
if ! command -v apptainer >/dev/null 2>&1; then
    command -v module >/dev/null 2>&1 || fail "module-command-is-unavailable"
    module load Apptainer >/dev/null 2>&1 \
        || fail "Apptainer-module-load-failed"
fi
command -v apptainer >/dev/null 2>&1 || fail "Apptainer-is-unavailable"

exec 9>"$AKI_RUN_ROOT/.fox-run.lock"
flock -n 9 || fail "run-root-lock-is-held"

export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export CUBLAS_WORKSPACE_CONFIG=:4096:8

common_exec=(
    srun --nodes=1 --ntasks=1
    apptainer exec --cleanenv --no-home --nv
    --bind "$AKI_REPO:$AKI_REPO:ro"
    --bind "$AKI_RUN_ROOT:$AKI_RUN_ROOT:ro"
    --bind "$AKI_RUN_ROOT/logs:$AKI_RUN_ROOT/logs:rw"
    --bind "$AKI_RUN_ROOT/tmp:$AKI_RUN_ROOT/tmp:rw"
    --bind "$AKI_RUN_ROOT/tmp:/tmp:rw"
    --pwd "$AKI_REPO"
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    --env "PYTHONHASHSEED=$PYTHONHASHSEED"
    --env "PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE"
    --env "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
    --env "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
    --env "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    --env "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    --env "MKL_NUM_THREADS=$MKL_NUM_THREADS"
    --env "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
    --env "OMP_DYNAMIC=$OMP_DYNAMIC"
    --env "MKL_DYNAMIC=$MKL_DYNAMIC"
    --env "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
    --env "TMPDIR=$AKI_RUN_ROOT/tmp"
    "$AKI_CONTAINER"
)

run_stage() {
    local stage_name="$1"
    local stage_log="$2"
    shift 2
    echo "stage=$stage_name status=STARTED"
    if "$@" >"$stage_log" 2>&1; then
        chmod 0600 "$stage_log"
        echo "stage=$stage_name status=PASS"
    else
        local status=$?
        chmod 0600 "$stage_log" 2>/dev/null || true
        echo "stage=$stage_name status=FAIL exit_code=$status" >&2
        return "$status"
    fi
}

printf '%s\n' \
    "started_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "slurm_job_id=$SLURM_JOB_ID" \
    "slurm_cluster=${SLURM_CLUSTER_NAME:-unknown}" \
    "slurm_partition=$SLURM_JOB_PARTITION" \
    "architecture=$(uname -m)" \
    "gpu_name=$allocated_gpu_name" \
    "gpu_uuid=$allocated_gpu_uuid" \
    "config_byte_sha256=$EXPECTED_CONFIG_BYTE_SHA256" \
    "config_semantic_sha256=$EXPECTED_CONFIG_SEMANTIC_SHA256" \
    "source_manifest_sha256=$AKI_SOURCE_MANIFEST_SHA256" \
    "transfer_manifest_sha256=$AKI_TRANSFER_MANIFEST_SHA256" \
    "container_sha256=$AKI_CONTAINER_SHA256" \
    > "$AKI_RUN_ROOT/logs/fox-preflight.console.log"
chmod 0600 "$AKI_RUN_ROOT/logs/fox-preflight.console.log"
echo "stage=preflight status=PASS"

run_stage environment_and_cuda_smoke \
    "$AKI_RUN_ROOT/logs/fox-environment-and-cuda-smoke.console.log" \
    "${common_exec[@]}" python3 -B -c '
import importlib.metadata as metadata
import platform
import torch

packages = (
    "torch", "pandas", "numpy", "pyarrow", "PyYAML",
    "scikit-learn", "xgboost", "scipy", "pytest",
)
for package in packages:
    print(f"{package}={metadata.version(package)}")
print(f"python={platform.python_version()}")
print(f"machine={platform.machine()}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
assert platform.machine() == "aarch64"
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
print(f"cuda_device_name={name}")
assert "H200" in name.upper()
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)
x = torch.randn(256, 256, device="cuda:0")
y = x @ x.T
lstm = torch.nn.LSTM(8, 16, batch_first=True).to("cuda:0")
z, _ = lstm(torch.randn(4, 32, 8, device="cuda:0"))
layer = torch.nn.TransformerEncoderLayer(
    d_model=16, nhead=4, batch_first=True
).to("cuda:0")
q = layer(torch.randn(4, 32, 16, device="cuda:0"))
torch.cuda.synchronize()
print(
    "cuda_smoke=ok "
    f"matmul={tuple(y.shape)} lstm={tuple(z.shape)} transformer={tuple(q.shape)}"
)
'

verify_locked_state
run_stage validate_config_cli \
    "$AKI_RUN_ROOT/logs/fox-validate-config.console.log" \
    "${common_exec[@]}" python3 -B -m src.mimic.aki validate-config \
    --config "$frozen_config"
grep -q "$EXPECTED_CONFIG_SEMANTIC_SHA256" \
    "$AKI_RUN_ROOT/logs/fox-validate-config.console.log" \
    || fail "validated-semantic-hash-is-unexpected"

verify_locked_state
run_stage unit_tests \
    "$AKI_RUN_ROOT/logs/fox-unit-tests.console.log" \
    "${common_exec[@]}" python3 -B -m pytest -q \
    tests/mimic/aki tests/test_aki_cohort.py \
    --ignore=tests/mimic/aki/test_end_to_end_synthetic.py -p no:cacheprovider

verify_locked_state
run_stage synthetic_end_to_end_tests \
    "$AKI_RUN_ROOT/logs/fox-synthetic-e2e.console.log" \
    "${common_exec[@]}" python3 -B -m pytest -q \
    tests/mimic/aki/test_end_to_end_synthetic.py -p no:cacheprovider

verify_locked_state
run_stage pretraining_integrity_gate \
    "$AKI_RUN_ROOT/logs/fox-pretraining-integrity-gate.console.log" \
    "${common_exec[@]}" python3 -B scripts/mimic_aki_integrity_gate.py pretrain \
    --config "$frozen_config" \
    --cohort-dir "$AKI_RUN_ROOT/cohort" \
    --prepared-dir "$AKI_RUN_ROOT/prepared"
grep -q '"training_authorized": true' \
    "$AKI_RUN_ROOT/logs/fox-pretraining-integrity-gate.console.log" \
    || fail "pretraining-gate-did-not-authorize-training"

mkdir "$AKI_RUN_ROOT/experiment"
chmod 0700 "$AKI_RUN_ROOT/experiment"
common_exec=(
    srun --nodes=1 --ntasks=1
    apptainer exec --cleanenv --no-home --nv
    --bind "$AKI_REPO:$AKI_REPO:ro"
    --bind "$AKI_RUN_ROOT:$AKI_RUN_ROOT:ro"
    --bind "$AKI_RUN_ROOT/logs:$AKI_RUN_ROOT/logs:rw"
    --bind "$AKI_RUN_ROOT/experiment:$AKI_RUN_ROOT/experiment:rw"
    --bind "$AKI_RUN_ROOT/tmp:$AKI_RUN_ROOT/tmp:rw"
    --bind "$AKI_RUN_ROOT/tmp:/tmp:rw"
    --pwd "$AKI_REPO"
    --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    --env "PYTHONHASHSEED=$PYTHONHASHSEED"
    --env "PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE"
    --env "PYTHONNOUSERSITE=$PYTHONNOUSERSITE"
    --env "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
    --env "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    --env "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    --env "MKL_NUM_THREADS=$MKL_NUM_THREADS"
    --env "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
    --env "OMP_DYNAMIC=$OMP_DYNAMIC"
    --env "MKL_DYNAMIC=$MKL_DYNAMIC"
    --env "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
    --env "TMPDIR=$AKI_RUN_ROOT/tmp"
    "$AKI_CONTAINER"
)

gpu_telemetry="$AKI_RUN_ROOT/logs/fox-train.gpu-telemetry.csv"
gpu_telemetry_errors="$AKI_RUN_ROOT/logs/fox-train.gpu-telemetry.stderr.log"
printf 'uuid,name,utilization_gpu_percent,memory_used_mib\n' > "$gpu_telemetry"
nvidia-smi -i "$allocated_gpu" \
    --query-gpu=uuid,name,utilization.gpu,memory.used \
    --format=csv,noheader,nounits --loop=1 \
    >> "$gpu_telemetry" 2> "$gpu_telemetry_errors" &
telemetry_pid=$!

cleanup_telemetry() {
    if [[ -n "${telemetry_pid:-}" ]]; then
        kill "$telemetry_pid" 2>/dev/null || true
        wait "$telemetry_pid" 2>/dev/null || true
        telemetry_pid=""
    fi
}
trap cleanup_telemetry EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

verify_locked_state
run_stage train_complete_matrix \
    "$AKI_RUN_ROOT/logs/fox-train.console.log" \
    "${common_exec[@]}" python3 -u -m src.mimic.aki train \
    --config "$frozen_config" \
    --prepared-dir "$AKI_RUN_ROOT/prepared" \
    --output-dir "$AKI_RUN_ROOT/experiment" \
    --models logistic_regression xgboost lstm transformer

cleanup_telemetry
chmod 0600 "$gpu_telemetry" "$gpu_telemetry_errors"

verify_locked_state
run_stage posttraining_integrity_gate \
    "$AKI_RUN_ROOT/logs/fox-posttraining-integrity-gate.console.log" \
    "${common_exec[@]}" python3 -B scripts/mimic_aki_integrity_gate.py posttrain \
    --config "$frozen_config" \
    --prepared-dir "$AKI_RUN_ROOT/prepared" \
    --experiment-dir "$AKI_RUN_ROOT/experiment" \
    --train-log "$AKI_RUN_ROOT/logs/fox-train.console.log" \
    --gpu-telemetry "$gpu_telemetry"
grep -q '"overall_gate": "PASS"' \
    "$AKI_RUN_ROOT/logs/fox-posttraining-integrity-gate.console.log" \
    || fail "posttraining-gate-did-not-pass"

printf 'completed_at_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    > "$AKI_RUN_ROOT/logs/fox-completed.log"
chmod 0600 "$AKI_RUN_ROOT/logs/fox-completed.log"
(cd "$AKI_RUN_ROOT" && find experiment logs -type f -print0 \
    | sort -z | xargs -0 sha256sum) \
    > "$AKI_RUN_ROOT/protocol/fox-training-output.sha256"
chmod 0400 "$AKI_RUN_ROOT/protocol/fox-training-output.sha256"

if find "$AKI_RUN_ROOT" -type l -print -quit | grep -q .; then
    fail "completed-run-tree-contains-a-symbolic-link"
fi
if find "$AKI_RUN_ROOT" ! -type d ! -type f -print -quit | grep -q .; then
    fail "transferred-run-tree-has-an-unsupported-entry-type"
fi
if find "$AKI_RUN_ROOT" -perm /0077 -print -quit | grep -q .; then
    fail "completed-run-tree-has-group-or-world-permissions"
fi
echo "stage=workflow status=PASS"
