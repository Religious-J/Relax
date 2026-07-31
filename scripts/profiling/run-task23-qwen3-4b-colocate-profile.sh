#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
#
# Memory-guarded microbenchmark for Task 23 (Qwen3-4B colocate text).
#
# Outputs:
#   traces/<run>/phase_timeline/  cross-process phase timeline
#   traces/<run>/train_trace/     Megatron torch-profiler traces
#   traces/<run>/sglang_trace/    SGLang torch-profiler traces

set -e
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RELAX_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
cd "${RELAX_ROOT}"

run_stamp="$(date +%Y%m%d_%H%M%S)"
PROFILE_RUN_NAME="${PROFILE_RUN_NAME:-task23-qwen3-4b-h20-${run_stamp}}"
PROFILE_ROOT="${PROFILE_ROOT:-${RELAX_ROOT}/traces/${PROFILE_RUN_NAME}}"
PROFILE_MAX_GPU_MIB="${PROFILE_MAX_GPU_MIB:-90000}"
PROFILE_MIN_HOST_AVAILABLE_MIB="${PROFILE_MIN_HOST_AVAILABLE_MIB:-131072}"
PROFILE_GPU_LIMIT_STRIKES="${PROFILE_GPU_LIMIT_STRIKES:-3}"
PROFILE_MONITOR_INTERVAL="${PROFILE_MONITOR_INTERVAL:-2}"
PROFILE_RAY_NUM_CPUS="${PROFILE_RAY_NUM_CPUS:-64}"
PROFILE_NOFILE_LIMIT="${PROFILE_NOFILE_LIMIT:-65535}"
PROFILE_ENABLE_TORCH="${PROFILE_ENABLE_TORCH:-1}"
PROFILE_ENABLE_SGLANG="${PROFILE_ENABLE_SGLANG:-1}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-1}"

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
DATA_DIR="${DATA_DIR:-/root/data}"
EXP_DIR="${EXP_DIR:-/root/data/task23-profile-exps/${PROFILE_RUN_NAME}}"
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
PROFILE_ROLLOUT_BATCH_SIZE="${PROFILE_ROLLOUT_BATCH_SIZE:-4}"
PROFILE_N_SAMPLES_PER_PROMPT="${PROFILE_N_SAMPLES_PER_PROMPT:-2}"
PROFILE_GLOBAL_BATCH_SIZE="${PROFILE_GLOBAL_BATCH_SIZE:-8}"
PROFILE_MAX_RESPONSE_LEN="${PROFILE_MAX_RESPONSE_LEN:-512}"
PROFILE_MAX_TOKENS_PER_GPU="${PROFILE_MAX_TOKENS_PER_GPU:-2048}"
PROFILE_LOG_PROBS_MAX_TOKENS_PER_GPU="${PROFILE_LOG_PROBS_MAX_TOKENS_PER_GPU:-4096}"
PROFILE_KL_LOSS_COEF="${PROFILE_KL_LOSS_COEF:-0.001}"
PROFILE_SGLANG_MEM_FRACTION_STATIC="${PROFILE_SGLANG_MEM_FRACTION_STATIC:-0.55}"
PROFILE_ROLLOUT_NUM_GPUS_PER_ENGINE="${PROFILE_ROLLOUT_NUM_GPUS_PER_ENGINE:-8}"
PROFILE_SGLANG_ROUTER_POLICY="${PROFILE_SGLANG_ROUTER_POLICY:-}"
PROFILE_USE_GROUP_AFFINITY_ROUTER="${PROFILE_USE_GROUP_AFFINITY_ROUTER:-0}"
PROFILE_TRAIN_TARGET="${PROFILE_TRAIN_TARGET:-train_overall}"
PROFILE_STEP_START="${PROFILE_STEP_START:-1}"
PROFILE_STEP_END="${PROFILE_STEP_END:-1}"
PROFILE_SGLANG_STEPS="${PROFILE_SGLANG_STEPS:-1}"
PROFILE_SGLANG_NUM_STEPS="${PROFILE_SGLANG_NUM_STEPS:-3}"

test -f "${MODEL_DIR}/Qwen3-4B/config.json"
test -f "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
test -f "${DATA_DIR}/aime-2024/aime-2024.jsonl"
mkdir -p "${PROFILE_ROOT}" "${EXP_DIR}"

ulimit -n "${PROFILE_NOFILE_LIMIT}"
if ! timeout 5 ray status >/dev/null 2>&1; then
    ray stop --grace-period 15 2>/dev/null || ray stop --force 2>/dev/null || true
    pkill -9 sglang 2>/dev/null || true
    ray start --head \
        --node-ip-address 127.0.0.1 \
        --num-cpus "${PROFILE_RAY_NUM_CPUS}" \
        --num-gpus 8 \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
    export RAY_ADDRESS=127.0.0.1:6379
fi

if [ -z "${RELAX_ENTRYPOINT_MODE:-}" ]; then
    source "${RELAX_ROOT}/scripts/entrypoint/local.sh"
fi
# local.sh enables shell tracing for launch diagnostics. Disable it here so the
# one-second resource monitor does not flood run logs or perturb benchmarks.
set +x
if [ "${NCCL_NVLS_ENABLE:-}" = "0" ]; then
    export HAS_NVLINK=0
    RUNTIME_ENV_JSON="$(
        printf '%s' "${RUNTIME_ENV_JSON}" \
            | sed 's/"NCCL_NVLS_ENABLE": "1"/"NCCL_NVLS_ENABLE": "0"/'
    )"
fi
source "${MODEL_CONFIG_DIR}/qwen3-4B.sh"

monitor_pid=""
cleanup() {
    rc=$?
    if [ -n "${monitor_pid}" ]; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
    ray stop --grace-period 30 2>/dev/null || ray stop --force 2>/dev/null || true
    pkill -TERM sglang 2>/dev/null || pkill -9 sglang 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT INT TERM

monitor_resources() {
    parent_pid=$1
    gpu_strikes=0
    host_strikes=0
    monitor_file="${PROFILE_ROOT}/resource_monitor.csv"
    host_monitor_file="${PROFILE_ROOT}/host_memory_monitor.csv"
    echo "timestamp,gpu,index,memory_used_mib,memory_total_mib,utilization_gpu_pct" >"${monitor_file}"
    echo "timestamp,cgroup_used_mib,cgroup_limit_mib,host_available_mib,effective_available_mib" >"${host_monitor_file}"
    while kill -0 "${parent_pid}" 2>/dev/null; do
        timestamp="$(date --iso-8601=seconds)"
        gpu_rows="$(
            nvidia-smi \
                --query-gpu=name,index,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits
        )"
        while IFS= read -r row; do
            echo "${timestamp},${row}" >>"${monitor_file}"
        done <<<"${gpu_rows}"

        max_used="$(
            printf '%s\n' "${gpu_rows}" \
                | awk -F',' '{value=$3+0; if (value > max) max=value} END {print max+0}'
        )"
        if [ "${max_used}" -ge "${PROFILE_MAX_GPU_MIB}" ]; then
            gpu_strikes=$((gpu_strikes + 1))
        else
            gpu_strikes=0
        fi

        cgroup_used_bytes=0
        cgroup_limit_bytes=0
        if [ -r /sys/fs/cgroup/memory.current ]; then
            cgroup_used_bytes="$(cat /sys/fs/cgroup/memory.current)"
            cgroup_limit_raw="$(cat /sys/fs/cgroup/memory.max)"
            if [ "${cgroup_limit_raw}" != "max" ]; then
                cgroup_limit_bytes="${cgroup_limit_raw}"
            fi
        elif [ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]; then
            cgroup_used_bytes="$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes)"
            cgroup_limit_bytes="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
        fi
        cgroup_used_mib=$((cgroup_used_bytes / 1024 / 1024))
        cgroup_limit_mib=$((cgroup_limit_bytes / 1024 / 1024))
        host_available_mib="$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)"
        effective_available_mib="${host_available_mib}"
        if [ "${cgroup_limit_mib}" -gt 0 ]; then
            cgroup_available_mib=$((cgroup_limit_mib - cgroup_used_mib))
            if [ "${cgroup_available_mib}" -lt "${effective_available_mib}" ]; then
                effective_available_mib="${cgroup_available_mib}"
            fi
        fi
        echo "${timestamp},${cgroup_used_mib},${cgroup_limit_mib},${host_available_mib},${effective_available_mib}" \
            >>"${host_monitor_file}"

        if [ "${effective_available_mib}" -le "${PROFILE_MIN_HOST_AVAILABLE_MIB}" ]; then
            host_strikes=$((host_strikes + 1))
        else
            host_strikes=0
        fi

        if [ "${gpu_strikes}" -ge "${PROFILE_GPU_LIMIT_STRIKES}" ]; then
            {
                echo "GPU memory guard triggered."
                echo "max_used_mib=${max_used}"
                echo "limit_mib=${PROFILE_MAX_GPU_MIB}"
                echo "consecutive_samples=${gpu_strikes}"
            } >"${PROFILE_ROOT}/OOM_GUARD_TRIGGERED.txt"
            ray stop --force 2>/dev/null || true
            pkill -9 sglang 2>/dev/null || true
            kill -TERM "${parent_pid}" 2>/dev/null || true
            return
        fi
        if [ "${host_strikes}" -ge "${PROFILE_GPU_LIMIT_STRIKES}" ]; then
            {
                echo "Host memory guard triggered."
                echo "effective_available_mib=${effective_available_mib}"
                echo "limit_mib=${PROFILE_MIN_HOST_AVAILABLE_MIB}"
                echo "consecutive_samples=${host_strikes}"
            } >"${PROFILE_ROOT}/OOM_GUARD_TRIGGERED.txt"
            ray stop --force 2>/dev/null || true
            pkill -9 sglang 2>/dev/null || true
            kill -TERM "${parent_pid}" 2>/dev/null || true
            return
        fi
        sleep "${PROFILE_MONITOR_INTERVAL}"
    done
}

monitor_resources "$$" &
monitor_pid=$!

PROJECT_NAME="${PROJECT_NAME:-Relax/task23-profile}"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}/Qwen3-4B/"
    --ref-load "${MODEL_DIR}/Qwen3-4B/"
    --load "${MODEL_DIR}/Qwen3-4B/"
    --megatron-to-hf-mode bridge
    --warm-hf-checkpoint-page-cache
)

ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type dapo
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${PROFILE_ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${PROFILE_N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${PROFILE_MAX_RESPONSE_LEN}"
    --rollout-temperature 1
    --global-batch-size "${PROFILE_GLOBAL_BATCH_SIZE}"
    --balance-data
    --use-fault-tolerance
)

EVAL_ARGS=(
    --skip-eval-before-train
    --log-passrate
    --eval-interval 20
    --eval-prompt-data aime "${DATA_DIR}/aime-2024/aime-2024.jsonl"
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 16384
    --eval-top-p 0.7
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --calculate-per-token-loss
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${PROFILE_MAX_TOKENS_PER_GPU}"
    --log-probs-max-tokens-per-gpu "${PROFILE_LOG_PROBS_MAX_TOKENS_PER_GPU}"
    --train-memory-margin-bytes 8589934592
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef "${PROFILE_KL_LOSS_COEF}"
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${PROFILE_ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${PROFILE_SGLANG_MEM_FRACTION_STATIC}"
)
if [ -n "${PROFILE_SGLANG_ROUTER_POLICY}" ]; then
    SGLANG_ARGS+=(--sglang-router-policy "${PROFILE_SGLANG_ROUTER_POLICY}")
fi
if [ "${PROFILE_USE_GROUP_AFFINITY_ROUTER}" = "1" ]; then
    if [ -n "${PROFILE_SGLANG_ROUTER_POLICY}" ]; then
        echo "PROFILE_USE_GROUP_AFFINITY_ROUTER and PROFILE_SGLANG_ROUTER_POLICY are mutually exclusive" >&2
        exit 2
    fi
    SGLANG_ARGS+=(--use-slime-router --slime-router-sticky)
fi
if [ "${PROFILE_ENABLE_SGLANG}" = "1" ]; then
    SGLANG_ARGS+=(
        --sglang-profile
        --sglang-profile-steps "${PROFILE_SGLANG_STEPS}"
        --sglang-profile-num-steps "${PROFILE_SGLANG_NUM_STEPS}"
        --sglang-profile-by-stage
        --sglang-profile-record-shapes
        --sglang-profile-output-dir "${PROFILE_ROOT}/sglang_trace"
    )
    if [ "${PROFILE_WITH_STACK}" = "1" ]; then
        SGLANG_ARGS+=(--sglang-profile-with-stack)
    fi
fi

PROFILE_ARGS=(
    --timeline-dump-dir "${PROFILE_ROOT}/phase_timeline"
)
if [ "${PROFILE_ENABLE_TORCH}" = "1" ]; then
    PROFILE_ARGS+=(
        --use-pytorch-profiler
        --profile-target "${PROFILE_TRAIN_TARGET}"
        --profile-step-start "${PROFILE_STEP_START}"
        --profile-step-end "${PROFILE_STEP_END}"
    )
    if [ "${PROFILE_WITH_STACK}" = "1" ]; then
        PROFILE_ARGS+=(--profile-with-stack)
    fi
fi

TRACKING_ARGS=(
    --use-metrics-service
    --tb-project-name "${PROJECT_NAME}"
    --tb-experiment-name "${PROFILE_ROOT}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

ray job submit --address="http://127.0.0.1:8265" \
    ${WORKING_DIR:+--working-dir "${WORKING_DIR}"} \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -m relax.entrypoints.train \
    --resource '{"actor": [1, 8], "rollout": [1, 8]}' \
    --max-staleness 0 \
    --num-data-storage-units 1 \
    --colocate \
    --use-health-check \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${TRACKING_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    2>&1 | tee "${PROFILE_ROOT}/run.log"

find "${PROFILE_ROOT}" -type f -printf '%p %s bytes\n' | sort | tee "${PROFILE_ROOT}/artifacts.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    | tee "${PROFILE_ROOT}/gpu_processes_after_run.csv"
