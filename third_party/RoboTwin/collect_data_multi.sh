#!/bin/bash
# Multi-task data collection wrapper around collect_data.sh.
# Distributes tasks round-robin across N workers (= n_gpus * workers_per_gpu)
# and runs them in parallel. Multiple workers can share a GPU.
#
# Usage:
#   ./collect_data_multi.sh [flags] <task_config> <gpu_list> [task1 task2 ...]
#
# Flags:
#   --dry-run               Print plan and exit.
#   --skip-done             Drop tasks already at episode_num (per seed.txt).
#   --workers-per-gpu N     Run N parallel workers per GPU (default 1).
#                           Test GPU memory headroom before going above 2.
#
# Examples:
#   ./collect_data_multi.sh demo_randomized 0,1                      # 1 worker × 2 GPUs
#   ./collect_data_multi.sh --workers-per-gpu 2 demo_randomized 0    # 2 workers × 1 GPU
#   ./collect_data_multi.sh --workers-per-gpu 2 demo_randomized 0,1  # 2 workers × 2 GPUs = 4
#   ./collect_data_multi.sh --dry-run --skip-done demo_randomized 0  # preview, skip done

# ./collect_data_multi.sh demo_randomized_replay 0 <task_name>
# or with your active task list + multi-GPU/worker setup


set -euo pipefail

# Use the CUDA-capable OIDN build instead of the system one.
# Silences "OIDN Error: unsupported device type: CUDA" + the cascade of
# "invalid handle" errors from svulkan2's denoiser.
export LD_LIBRARY_PATH="${HOME}/oidn:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${HOME}/oidn/libOpenImageDenoise.so.2.3.3${LD_PRELOAD:+:${LD_PRELOAD}}"

# Restrict Vulkan to the NVIDIA ICD. Silences MESA's noisy Intel iGPU
# enumeration ("MESA: warning: Driver does not support the 0x7d67 PCI ID"
# + INCOMPATIBLE_DRIVER errors). SAPIEN only uses NVIDIA anyway.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# ----------------------------------------------------------------------
# Edit this list to pick a subset of tasks to run.
# - Leave EMPTY to auto-discover and run ALL tasks in envs/.
# - Tasks passed as CLI positional args override this list entirely.
# ----------------------------------------------------------------------
TASKS=(
    # lift_pot
    # place_shoe
    # place_dual_shoes
    # put_object_cabinet
    # adjust_bottle
    stack_blocks_three
    stack_bowls_three
    # place_can_basket
    
    # pick_diverse_bottles
    # pick_dual_bottles
    # place_empty_cup
    # move_can_pot
    # hanging_mug
)

dry_run=0
skip_done=0
workers_per_gpu=1

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)         dry_run=1; shift ;;
        --skip-done)       skip_done=1; shift ;;
        --workers-per-gpu) workers_per_gpu=$2; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

if [ $# -lt 2 ]; then
    echo "usage: $0 [--dry-run] [--skip-done] [--workers-per-gpu N] <task_config> <gpu_list> [task1 task2 ...]" >&2
    exit 1
fi

if ! [[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
    echo "--workers-per-gpu must be a positive integer (got: $workers_per_gpu)" >&2
    exit 1
fi

task_config=$1
gpu_list=$2
shift 2

IFS=',' read -ra gpus <<< "$gpu_list"
n_gpus=${#gpus[@]}
n_workers=$((n_gpus * workers_per_gpu))

# Task list precedence: CLI args > in-script TASKS array > auto-discover all
if [ $# -gt 0 ]; then
    tasks=("$@")
elif [ "${#TASKS[@]}" -gt 0 ]; then
    tasks=("${TASKS[@]}")
else
    mapfile -t tasks < <(ls envs/*.py | xargs -n1 basename | sed 's/\.py$//' | grep -vE '^(_|test_)')
fi

# Read episode_num from the task config
config_path="task_config/${task_config}.yml"
if [ ! -f "$config_path" ]; then
    echo "config not found: $config_path" >&2
    exit 1
fi
episode_num=$(awk '/^episode_num:/ {print $2; exit}' "$config_path")
if [ -z "$episode_num" ]; then
    echo "could not parse episode_num from $config_path" >&2
    exit 1
fi

# Optionally filter out tasks already at or above episode_num
filtered=()
skipped=()
for t in "${tasks[@]}"; do
    seed_file="data/${t}/${task_config}/seed.txt"
    done_count=0
    if [ -f "$seed_file" ]; then
        done_count=$(wc -w < "$seed_file" 2>/dev/null || echo 0)
    fi
    if [ "$skip_done" -eq 1 ] && [ "$done_count" -ge "$episode_num" ]; then
        skipped+=("$t")
    else
        filtered+=("$t")
    fi
done
tasks=("${filtered[@]}")

echo "Config:           ${task_config} (episode_num=${episode_num})"
echo "GPUs (${n_gpus}):         ${gpus[*]}"
echo "Workers/GPU:      ${workers_per_gpu}  (total workers: ${n_workers})"
echo "Tasks (${#tasks[@]}):       ${tasks[*]:-<none>}"
if [ "${#skipped[@]}" -gt 0 ]; then
    echo "Skipped done (${#skipped[@]}): ${skipped[*]}"
fi

if [ "${#tasks[@]}" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

# Worker w runs on GPU gpus[w % n_gpus]
declare -a worker_gpu
declare -a worker_tasks
for ((w=0; w<n_workers; w++)); do
    worker_gpu[w]=${gpus[$((w % n_gpus))]}
    worker_tasks[w]=""
done

# Round-robin tasks across workers
for i in "${!tasks[@]}"; do
    w=$((i % n_workers))
    worker_tasks[w]+="${tasks[$i]} "
done

echo
echo "Plan:"
for ((w=0; w<n_workers; w++)); do
    echo "  worker $w (GPU ${worker_gpu[w]}): ${worker_tasks[w]:-<none>}"
done

if [ "$dry_run" -eq 1 ]; then
    echo
    echo "[dry-run] not executing."
    exit 0
fi

mkdir -p logs

# Force python to flush stdout/stderr line-by-line so logs are watchable in real time
export PYTHONUNBUFFERED=1

# Launch one shell per worker
pids=()
for ((w=0; w<n_workers; w++)); do
    gpu=${worker_gpu[w]}
    wid=$w
    (
        worker_rc=0
        for t in ${worker_tasks[wid]}; do
            log="logs/${t}_${task_config}_gpu${gpu}_w${wid}.log"
            echo "[w${wid}/GPU $gpu] >>> $t  (log: $log)"
            # pipefail propagates collect_data.sh's exit code through sed
            ./collect_data.sh "$t" "$task_config" "$gpu" 2>&1 \
                | sed -ur 's/\x1b\[[0-9;]*[mGKHF]//g' > "$log"
            task_rc=${PIPESTATUS[0]}
            echo "[w${wid}/GPU $gpu] <<< $t  (exit=$task_rc)"
            [ "$task_rc" -ne 0 ] && worker_rc=$task_rc
        done
        exit $worker_rc
    ) &
    pids+=($!)
done

# Wait for all workers; propagate first non-zero exit code
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
done

echo "All workers complete (rc=$rc)."
exit $rc
