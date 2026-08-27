#!/bin/bash

# Submit train-only v3 AKI analysis from a FOX login node.

set -o errexit
set -o nounset
set -o pipefail
umask 077

: "${FOX_ACCOUNT:?ERROR: export your active FOX Slurm account}"
: "${AKI_REPO:?ERROR: export the absolute FOX repository path}"
: "${AKI_RUN_ROOT:?ERROR: export the transferred private v3 run root}"
: "${AKI_SLURM_LOG_DIR:?ERROR: export a private persistent Slurm log directory}"
: "${AKI_CONTAINER:?ERROR: export an ARM64 CUDA/PyTorch Apptainer image path}"
: "${AKI_CONTAINER_SHA256:?ERROR: export the container SHA-256}"
: "${AKI_SOURCE_MANIFEST_SHA256:?ERROR: export the source manifest SHA-256}"
: "${AKI_TRANSFER_MANIFEST:?ERROR: export the transfer manifest path}"
: "${AKI_TRANSFER_MANIFEST_SHA256:?ERROR: export the transfer manifest SHA-256}"

command -v sbatch >/dev/null 2>&1 || {
    echo "ERROR: sbatch is unavailable; submit only from FOX" >&2
    exit 2
}

for variable_name in \
    AKI_REPO \
    AKI_RUN_ROOT \
    AKI_SLURM_LOG_DIR \
    AKI_CONTAINER \
    AKI_TRANSFER_MANIFEST; do
    value="${!variable_name}"
    [[ "$value" = /* ]] || {
        echo "ERROR: $variable_name must be absolute" >&2
        exit 2
    }
    [[ "$value" != *','* && "$value" != *$'\n'* ]] || {
        echo "ERROR: $variable_name contains an unsupported character" >&2
        exit 2
    }
done

[[ "$FOX_ACCOUNT" != *','* && "$FOX_ACCOUNT" != *$'\n'* ]] || {
    echo "ERROR: FOX_ACCOUNT contains an unsupported character" >&2
    exit 2
}
for hash_name in AKI_CONTAINER_SHA256 AKI_SOURCE_MANIFEST_SHA256 \
    AKI_TRANSFER_MANIFEST_SHA256; do
    [[ "${!hash_name}" =~ ^[[:xdigit:]]{64}$ ]] || {
        echo "ERROR: $hash_name is not a full SHA-256" >&2
        exit 2
    }
done
for input_path in "$AKI_REPO" "$AKI_RUN_ROOT" "$AKI_CONTAINER" \
    "$AKI_TRANSFER_MANIFEST"; do
    [[ ! -L "$input_path" ]] || {
        echo "ERROR: input paths must not be symbolic links" >&2
        exit 2
    }
done
[[ -d "$AKI_REPO" && -d "$AKI_RUN_ROOT" ]] || {
    echo "ERROR: repository or transferred run root is missing" >&2
    exit 2
}
[[ ! -e "$AKI_REPO/data" && ! -e "$AKI_REPO/.venv" \
    && ! -e "$AKI_REPO/sitecustomize.py" \
    && ! -e "$AKI_REPO/usercustomize.py" ]] || {
    echo "ERROR: FOX source tree must exclude data, .venv, and Python customizers" >&2
    exit 2
}
AKI_REPO="$(realpath -e -- "$AKI_REPO")"
AKI_RUN_ROOT="$(realpath -e -- "$AKI_RUN_ROOT")"
AKI_CONTAINER="$(realpath -e -- "$AKI_CONTAINER")"
AKI_TRANSFER_MANIFEST="$(realpath -e -- "$AKI_TRANSFER_MANIFEST")"
if [[ -e "$AKI_SLURM_LOG_DIR" ]]; then
    [[ -d "$AKI_SLURM_LOG_DIR" && ! -L "$AKI_SLURM_LOG_DIR" ]] || {
        echo "ERROR: Slurm log path must be a non-symlink directory" >&2
        exit 2
    }
    AKI_SLURM_LOG_DIR="$(realpath -e -- "$AKI_SLURM_LOG_DIR")"
else
    log_parent="$(realpath -e -- "$(dirname -- "$AKI_SLURM_LOG_DIR")")"
    log_name="$(basename -- "$AKI_SLURM_LOG_DIR")"
    [[ "$log_name" != "." && "$log_name" != ".." && -n "$log_name" ]] || {
        echo "ERROR: invalid Slurm log directory name" >&2
        exit 2
    }
    AKI_SLURM_LOG_DIR="$log_parent/$log_name"
fi
case "$AKI_RUN_ROOT/" in
    "$AKI_REPO/"*)
        echo "ERROR: run root must be outside the repository" >&2
        exit 2
        ;;
    /tmp/*|/var/tmp/*)
        echo "ERROR: run root must use persistent storage" >&2
        exit 2
        ;;
esac
case "$AKI_REPO/" in
    "$AKI_RUN_ROOT/"*)
        echo "ERROR: repository and run root must be separate trees" >&2
        exit 2
        ;;
esac
case "$AKI_SLURM_LOG_DIR/" in
    "$AKI_REPO/"*|"$AKI_RUN_ROOT/"*)
        echo "ERROR: Slurm logs must be outside repository and run root" >&2
        exit 2
        ;;
    /tmp/*|/var/tmp/*)
        echo "ERROR: Slurm logs must use persistent storage" >&2
        exit 2
        ;;
esac
case "$AKI_REPO/" in
    "$AKI_SLURM_LOG_DIR/"*)
        echo "ERROR: repository and Slurm log directory must be disjoint" >&2
        exit 2
        ;;
esac
case "$AKI_RUN_ROOT/" in
    "$AKI_SLURM_LOG_DIR/"*)
        echo "ERROR: run root and Slurm log directory must be disjoint" >&2
        exit 2
        ;;
esac
if find "$AKI_RUN_ROOT" -maxdepth 0 -perm /0077 -print -quit | grep -q .; then
    echo "ERROR: run root has group/world permission bits" >&2
    exit 2
fi
[[ ! -e "$AKI_RUN_ROOT/experiment" ]] || {
    echo "ERROR: experiment output already exists; refusing overwrite" >&2
    exit 2
}
[[ -f "$AKI_CONTAINER" && -f "$AKI_TRANSFER_MANIFEST" ]] || {
    echo "ERROR: container or transfer manifest is missing" >&2
    exit 2
}
[[ "$(sha256sum "$AKI_CONTAINER" | awk '{print $1}')" \
    = "$AKI_CONTAINER_SHA256" ]] || {
    echo "ERROR: container SHA-256 mismatch" >&2
    exit 2
}
[[ "$(sha256sum "$AKI_TRANSFER_MANIFEST" | awk '{print $1}')" \
    = "$AKI_TRANSFER_MANIFEST_SHA256" ]] || {
    echo "ERROR: transfer manifest SHA-256 mismatch" >&2
    exit 2
}
source_manifest="$AKI_REPO/configs/mimic_aki_fox_h200_smd015_amendment_v3.sources.sha256"
[[ -f "$source_manifest" && ! -L "$source_manifest" ]] || {
    echo "ERROR: source manifest is missing or a symbolic link" >&2
    exit 2
}
[[ "$(sha256sum "$source_manifest" | awk '{print $1}')" \
    = "$AKI_SOURCE_MANIFEST_SHA256" ]] || {
    echo "ERROR: source manifest SHA-256 mismatch" >&2
    exit 2
}
(cd "$AKI_REPO" && sha256sum --quiet -c "$source_manifest") \
    >/dev/null 2>&1 || {
    echo "ERROR: source files do not match the pinned manifest" >&2
    exit 2
}
python3 -B "$AKI_REPO/scripts/mimic_aki_verify_source_manifest.py" \
    --repo-root "$AKI_REPO" --manifest "$source_manifest" >/dev/null || {
    echo "ERROR: exact source manifest verification failed" >&2
    exit 2
}
if find "$AKI_REPO/src" "$AKI_REPO/tests" \
    \( -type d -name __pycache__ -o -type f -name '*.pyc' \) \
    -print -quit | grep -q .; then
    echo "ERROR: clean source transfer contains Python bytecode cache" >&2
    exit 2
fi
[[ "$AKI_TRANSFER_MANIFEST" \
    = "$AKI_RUN_ROOT/protocol/cpu-to-fox-transfer.sha256" ]] || {
    echo "ERROR: transfer manifest is not the frozen run-root manifest" >&2
    exit 2
}
(cd "$AKI_RUN_ROOT" && sha256sum --quiet -c "$AKI_TRANSFER_MANIFEST") \
    >/dev/null 2>&1 || {
    echo "ERROR: transferred files do not match the checksum manifest" >&2
    exit 2
}
python3 -B "$AKI_REPO/scripts/mimic_aki_verify_transfer_manifest.py" \
    --run-root "$AKI_RUN_ROOT" --manifest "$AKI_TRANSFER_MANIFEST" \
    >/dev/null || {
    echo "ERROR: exact transfer manifest verification failed" >&2
    exit 2
}

mkdir -p "$AKI_SLURM_LOG_DIR"
chmod 0700 "$AKI_SLURM_LOG_DIR"

job_script="$AKI_REPO/scripts/slurm/FOX/mimic_aki_h200_v3.slurm"
[[ -f "$job_script" ]] || {
    echo "ERROR: FOX AKI job script is missing" >&2
    exit 2
}

export_spec="AKI_REPO=$AKI_REPO"
export_spec+=",AKI_RUN_ROOT=$AKI_RUN_ROOT"
export_spec+=",AKI_SLURM_LOG_DIR=$AKI_SLURM_LOG_DIR"
export_spec+=",AKI_CONTAINER=$AKI_CONTAINER"
export_spec+=",AKI_CONTAINER_SHA256=$AKI_CONTAINER_SHA256"
export_spec+=",AKI_SOURCE_MANIFEST_SHA256=$AKI_SOURCE_MANIFEST_SHA256"
export_spec+=",AKI_TRANSFER_MANIFEST=$AKI_TRANSFER_MANIFEST"
export_spec+=",AKI_TRANSFER_MANIFEST_SHA256=$AKI_TRANSFER_MANIFEST_SHA256"

sbatch \
    --account="$FOX_ACCOUNT" \
    --chdir="$AKI_REPO" \
    --output="$AKI_SLURM_LOG_DIR/mimic-aki-h200-v3-%j.out" \
    --error="$AKI_SLURM_LOG_DIR/mimic-aki-h200-v3-%j.err" \
    --umask=0077 \
    --export="$export_spec" \
    "$job_script"
