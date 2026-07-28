#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -uo pipefail

cd /root/Relax

run_id="${RUN_ID:-$(date '+%Y-%m-%d-%H:%M:%S')}"
gpu_log="log/qwen3-4b-GRPO-gpu4-pro5000-baseline-${run_id}-gpu.csv"
meta_log="log/qwen3-4b-GRPO-gpu4-pro5000-baseline-${run_id}-meta.txt"

mkdir -p log
{
    echo "run_id=${run_id}"
    echo "launcher_pid=$$"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "gpu_log=${gpu_log}"
    echo "start_time=$(date --iso-8601=seconds)"
} >"${meta_log}"

(
    echo "timestamp,index,name,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB],power.draw [W],temperature.gpu"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits
        sleep 1
    done
) >"${gpu_log}" 2>&1 &
monitor_pid=$!

cleanup() {
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

NUM_GPUS=4 \
MODEL_DIR=/workspace/models \
DATA_DIR=/root/data \
NUM_ROLLOUT="${NUM_ROLLOUT:-6}" \
bash scripts/training/text/run-qwen3-4B-4xgpu-pro5000-baseline.sh
status=$?

{
    echo "end_time=$(date --iso-8601=seconds)"
    echo "exit_status=${status}"
} >>"${meta_log}"

exit "${status}"
