#!/bin/bash
# Wrapper around collect_data_multi.sh for the replay configs (demo_randomized_replay
# / demo_clean_replay). Builds a task list by:
#   1. Including tasks that have seed.txt in data_replay/<task>/<config>/
#   2. Skipping tasks already fully collected in the ORIGINAL data/ tree
#   3. Skipping tasks listed in EXCLUDE below
#
# Usage:
#   ./collect_data_replay.sh [--dry-run] [--workers-per-gpu N] <replay_config> <gpu_list>
#
# Examples:
#   ./collect_data_replay.sh demo_randomized_replay 0
#   ./collect_data_replay.sh demo_clean_replay 0
#   ./collect_data_replay.sh --workers-per-gpu 2 demo_clean_replay 0,1
#   ./collect_data_replay.sh --dry-run demo_randomized_replay 0

set -euo pipefail

# ----------------------------------------------------------------------
# If INCLUDE is non-empty, ONLY these tasks are considered (still subject
# to seed.txt / done-in-data checks). Leave empty to consider all tasks
# found under data_replay/*/<replay_config>/.
# ----------------------------------------------------------------------
INCLUDE=(
    # rotate_qrcode
    # scan_object
    # shake_bottle
    # shake_bottle_horizontally
    # stamp_seal
    # turn_switch
    # stack_blocks_two
    stack_bowls_two
    # put_object_cabinet
    # press_stapler
    # put_bottles_dustbin
)

# ----------------------------------------------------------------------
# Edit this list to exclude tasks from replay.
# Typical reasons: task currently being collected in data/, known-broken
# planner, or just not wanted right now.
# Ignored when INCLUDE is non-empty.
# ----------------------------------------------------------------------
EXCLUDE=(
    rotate_qrcode
    scan_object
    shake_bottle
    shake_bottle_horizontally
    stamp_seal
    turn_switch
    stack_blocks_two
    stack_bowls_two
    put_object_cabinet
    press_stapler
    put_bottles_dustbin
    
    # stack_blocks_three     # currently being collected in data/
    # stack_bowls_three     # planner regression; investigate before replay
)

dry_run=0
workers_per_gpu=2

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)         dry_run=1; shift ;;
        --workers-per-gpu) workers_per_gpu=$2; shift 2 ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *) break ;;
    esac
done

if [ $# -lt 2 ]; then
    echo "usage: $0 [--dry-run] [--workers-per-gpu N] <replay_config> <gpu_list>" >&2
    exit 1
fi

replay_config=$1
gpu_list=$2

# Map replay_config -> original task_config + episode_num for the "done in data/" check
case "$replay_config" in
    demo_randomized_replay) orig_config=demo_randomized; episode_num=500 ;;
    demo_clean_replay)      orig_config=demo_clean;      episode_num=50  ;;
    *) echo "unknown replay_config: $replay_config" >&2; exit 1 ;;
esac

# Build include / exclude sets for O(1) lookup. INCLUDE takes precedence:
# when non-empty, EXCLUDE is ignored and only included tasks are candidates.
declare -A included excluded
for t in "${INCLUDE[@]}"; do included[$t]=1; done
if [ "${#INCLUDE[@]}" -eq 0 ]; then
    for t in "${EXCLUDE[@]}"; do excluded[$t]=1; done
fi

# Gather candidate tasks + filter
tasks=()
skipped_excluded=()
skipped_not_included=()
skipped_done=()
skipped_no_seed=()

for d in data_replay/*/"${replay_config}"/; do
    [ -d "$d" ] || continue
    t=$(basename "$(dirname "$d")")

    if [ ! -f "$d/seed.txt" ]; then
        skipped_no_seed+=("$t")
        continue
    fi
    if [ "${#INCLUDE[@]}" -gt 0 ] && [ -z "${included[$t]:-}" ]; then
        skipped_not_included+=("$t")
        continue
    fi
    if [ -n "${excluded[$t]:-}" ]; then
        skipped_excluded+=("$t")
        continue
    fi

    orig_dir="data/${t}/${orig_config}/data"
    orig_hdf5=0
    if [ -d "$orig_dir" ]; then
        orig_hdf5=$(find "$orig_dir" -maxdepth 1 -name 'episode*.hdf5' 2>/dev/null | wc -l)
    fi
    if [ "$orig_hdf5" -ge "$episode_num" ]; then
        skipped_done+=("$t ($orig_hdf5/$episode_num)")
        continue
    fi

    tasks+=("$t")
done

echo "Replay config:    $replay_config  (episode_num=$episode_num)"
echo "Workers/GPU:      $workers_per_gpu"
echo "Included (${#tasks[@]}):  ${tasks[*]:-<none>}"
[ ${#skipped_not_included[@]} -gt 0 ] && echo "Not in INCLUDE (${#skipped_not_included[@]}):    ${skipped_not_included[*]}"
[ ${#skipped_excluded[@]}  -gt 0 ] && echo "Excluded by list (${#skipped_excluded[@]}): ${skipped_excluded[*]}"
[ ${#skipped_done[@]}      -gt 0 ] && echo "Skipped (done in data/, ${#skipped_done[@]}):    ${skipped_done[*]}"
[ ${#skipped_no_seed[@]}   -gt 0 ] && echo "Skipped (no seed.txt, ${#skipped_no_seed[@]}):   ${skipped_no_seed[*]}"

if [ "${#tasks[@]}" -eq 0 ]; then
    echo "Nothing to do."
    exit 0
fi

cmd=(./collect_data_multi.sh --workers-per-gpu "$workers_per_gpu" "$replay_config" "$gpu_list" "${tasks[@]}")
echo
echo "Command:"
echo "  ${cmd[*]}"

if [ "$dry_run" -eq 1 ]; then
    echo "[dry-run] not executing."
    exit 0
fi

exec "${cmd[@]}"
