#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for the local RWKV-7 evaluation wrappers.
: "${MODEL_PATH:=/workspace/.hf_home/hub/models--RWKV--RWKV7-G1j-1.5B-20260831/snapshots/2c18b29ab7fbece25ff6112281eea0fa41fcb30f}"
: "${HF_HOME:=/workspace/.hf_home}"
: "${OUTPUT_ROOT:=/workspace/results/rwkv7-olmo-eval}"
: "${OLMO_EVAL_BIN:=/venv/main/bin/olmo-eval}"
: "${DTYPE:=bfloat16}"
: "${NUM_GPUS:=1}"
: "${PARALLELISM:=1}"
export HF_HOME

run_eval() {
    local run_name="$1"
    local task_spec="$2"
    local harness="${3:-default}"
    shift 3

    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "Model path does not exist: ${MODEL_PATH}" >&2
        exit 1
    fi
    if [[ ! -x "${OLMO_EVAL_BIN}" ]]; then
        echo "olmo-eval executable does not exist: ${OLMO_EVAL_BIN}" >&2
        exit 1
    fi

    local output_dir="${OUTPUT_ROOT}/${run_name}"
    mkdir -p "${output_dir}"

    local -a task_overrides=()
    if [[ -n "${LIMIT:-}" ]]; then
        task_overrides+=(-o "limit=${LIMIT}")
    fi

    "${OLMO_EVAL_BIN}" run \
        -H "${harness}" \
        -o provider.kind=hf \
        -o provider.trust_remote_code=true \
        -o "provider.dtype=${DTYPE}" \
        -m "${MODEL_PATH}" \
        -t "${task_spec}" \
        "${task_overrides[@]}" \
        --num-gpus "${NUM_GPUS}" \
        --parallelism "${PARALLELISM}" \
        -O "${output_dir}" \
        "$@"
}

unsupported_eval() {
    local label="$1"
    local reason="$2"
    echo "[${label}] unavailable: ${reason}" >&2
    exit 2
}

