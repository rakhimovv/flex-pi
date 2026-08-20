#!/usr/bin/env bash
# Convert RoboTwin raw HDF5 episodes into a LeRobot v2.1 dataset.
#
# Two modes:
#
#   1) ALL tasks (default, no positional args) — produces ONE combined dataset
#      matching the reference robotwin2.0 layout (global episode_index across
#      every task/config, single meta/info.json, single tasks.jsonl):
#        bash script/convert_robotwin_to_lerobot.sh
#
#   2) Single task (backwards-compatible):
#        bash script/convert_robotwin_to_lerobot.sh <task_name> [config] [out_dir] [depth_codec] [episodes...]
#
# Options:
#   --out-dir DIR        Output dataset dir for all-tasks mode
#                        (default: <flex-pi>/data/robotwin2.0_3d)
#   --configs "A B ..."  Configs to include in order (default: "demo_randomized demo_clean")
#   --depth-codec CODEC  ffv1 | x264rgb | both | none   (default: both)
#   --dry-run            Print the constructed Python command and exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"      # third_party/RoboTwin
FLEXPI_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)"      # the flex-pi project dir

OUT_DIR="${FLEXPI_ROOT}/data/robotwin2.0_3d"
DEPTH_CODEC="both"
CONFIGS=("demo_randomized" "demo_clean")
WORKERS=2
# Drive task order from the reference map so episode indices line up with the
# existing task_episode_map.json (reusable without regeneration). Set to an
# empty string to fall back to alphabetical.
TASK_ORDER_MAP="${FLEXPI_ROOT}/src/flexpi/datasets/task_episode_map.json"
DRY_RUN=0

# Parse flags
while [ $# -gt 0 ]; do
    case "$1" in
        --out-dir)      OUT_DIR="$2"; shift 2 ;;
        --configs)      IFS=' ' read -ra CONFIGS <<< "$2"; shift 2 ;;
        --depth-codec)  DEPTH_CODEC="$2"; shift 2 ;;
        --workers)      WORKERS="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

# Remaining positional args (backwards-compat single-task mode)
SINGLE_TASK="${1:-}"
SINGLE_CONFIG="${2:-}"
SINGLE_OUT="${3:-}"
SINGLE_DEPTH="${4:-}"
if [ $# -ge 4 ]; then shift 4; else shift $#; fi
SINGLE_EPISODES=( "$@" )

CONVERT_PY="${REPO_ROOT}/script/convert_robotwin_to_lerobot.py"

PY_BIN="${PY_BIN:-python}"
if [[ ! -x "${PY_BIN}" ]]; then
    PY_BIN="$(command -v python)"
fi

# -------- SINGLE-TASK MODE (backwards-compatible) --------
if [ -n "$SINGLE_TASK" ]; then
    cfg="${SINGLE_CONFIG:-demo_randomized}"
    out="${SINGLE_OUT:-/tmp/${SINGLE_TASK}_lerobot}"
    depth="${SINGLE_DEPTH:-${DEPTH_CODEC}}"
    raw_dir="${REPO_ROOT}/data/${SINGLE_TASK}/${cfg}"
    if [[ ! -d "${raw_dir}/data" ]]; then
        echo "ERROR: raw data dir not found: ${raw_dir}/data" >&2
        exit 1
    fi
    cmd=(
        "${PY_BIN}" "${CONVERT_PY}"
        --raw-dir    "${raw_dir}"
        --out-dir    "${out}"
        --task-name  "${SINGLE_TASK}"
        --depth-codec "${depth}"
        --workers    "${WORKERS}"
        --overwrite
    )
    if [[ ${#SINGLE_EPISODES[@]} -gt 0 ]]; then
        cmd+=( --episodes "${SINGLE_EPISODES[@]}" )
    fi
    echo "[run] ${cmd[*]}"
    [ "$DRY_RUN" -eq 1 ] && exit 0
    "${cmd[@]}"
    exit 0
fi

# -------- ALL-TASKS MODE (one combined dataset) --------

# Determine task iteration order. Prefer the reference map's ordering so that
# episode indices in the new dataset match the existing task_episode_map.json.
if [ -n "$TASK_ORDER_MAP" ] && [ -f "$TASK_ORDER_MAP" ]; then
    mapfile -t ORDERED_TASKS < <(
        "${PY_BIN}" -c "
import json, sys
m = json.load(open('$TASK_ORDER_MAP'))
for name, info in sorted(m['tasks'].items(), key=lambda kv: kv[1]['episode_start']):
    print(name)
"
    )
    echo "[task order] using reference map ($TASK_ORDER_MAP): ${#ORDERED_TASKS[@]} tasks"
else
    mapfile -t ORDERED_TASKS < <(ls -1 "${REPO_ROOT}/data/" | sort)
    echo "[task order] alphabetical fallback (no task_episode_map.json): ${#ORDERED_TASKS[@]} tasks"
fi

raw_dirs=()
task_names=()
skipped_tasks=()
for t in "${ORDERED_TASKS[@]}"; do
    task_dir="${REPO_ROOT}/data/${t}"
    if [ ! -d "$task_dir" ]; then
        skipped_tasks+=("$t (no data dir)")
        continue
    fi
    added=0
    for cfg in "${CONFIGS[@]}"; do
        if [ -d "${task_dir}/${cfg}/data" ]; then
            raw_dirs+=("${task_dir}/${cfg}")
            task_names+=("${t}")
            added=$((added+1))
        fi
    done
    if [ "$added" -eq 0 ]; then
        skipped_tasks+=("$t (no configs)")
    fi
done

if [ "${#raw_dirs[@]}" -eq 0 ]; then
    echo "no task/config directories found under ${REPO_ROOT}/data/" >&2
    exit 1
fi

echo "plan: merge ${#raw_dirs[@]} (task, config) sources into ONE combined dataset"
echo "  out-dir:     ${OUT_DIR}"
echo "  configs:     ${CONFIGS[*]}"
echo "  depth-codec: ${DEPTH_CODEC}"
echo "  workers:     ${WORKERS}"
echo "  task order:  ${TASK_ORDER_MAP:-alphabetical}"
echo "  sources:"
for i in "${!raw_dirs[@]}"; do
    printf "    [%3d] %s\n" "$i" "${raw_dirs[$i]#${REPO_ROOT}/}"
done
if [ "${#skipped_tasks[@]}" -gt 0 ]; then
    echo "  skipped: ${skipped_tasks[*]}"
fi
echo

cmd=(
    "${PY_BIN}" "${CONVERT_PY}"
    --raw-dirs   "${raw_dirs[@]}"
    --task-names "${task_names[@]}"
    --out-dir    "${OUT_DIR}"
    --depth-codec "${DEPTH_CODEC}"
    --workers    "${WORKERS}"
    --overwrite
)

echo "[run] ${cmd[*]:0:6} ... (${#raw_dirs[@]} sources) ..."
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] not executing."
    exit 0
fi

"${cmd[@]}"
echo "done."
