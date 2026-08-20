#!/usr/bin/env bash
# DA3 metric-depth labeling pass over an RGB-only LeRobot dataset.
# Thin wrapper around label_depth.py: resolves the interpreter and the
# checkpoint from env.sh (written by setup.sh) so callers only name a dataset.
#
#   bash scripts/da3_depth/setup.sh                            # once
#   CHECK=1 bash scripts/da3_depth/run.sh <dataset_dir>        # validate, no GPU
#   bash scripts/da3_depth/run.sh <dataset_dir> [flags...]     # label
#   NUM_SHARDS=4 SHARD_ID=$i bash scripts/da3_depth/run.sh <d> # one per GPU
#   PATCH_INFO_ONLY=1 bash scripts/da3_depth/run.sh <d>        # finalize after shards
#
# DA3_PY / DA3_WEIGHTS set in the environment override env.sh.
set -euo pipefail

DS="${1:?usage: run.sh <lerobot_dataset_dir> [flags...]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${HERE}/label_depth.py"

# shellcheck disable=SC1091
[[ -f "${HERE}/env.sh" ]] && source "${HERE}/env.sh"
: "${DA3_PY:=}"; : "${DA3_WEIGHTS:=}"
if [[ ! -x "${DA3_PY}" ]]; then
  echo "ERROR: no DA3 interpreter (DA3_PY='${DA3_PY}'). Run: bash ${HERE}/setup.sh" >&2
  exit 1
fi

NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# --check and --patch-info-only need neither GPU nor checkpoint.
if [[ "${CHECK:-0}" == "1" ]]; then
  exec "${DA3_PY}" "${PY_SCRIPT}" --dataset-dir "${DS}" --check "${@:2}"
fi
if [[ "${PATCH_INFO_ONLY:-0}" == "1" ]]; then
  exec "${DA3_PY}" "${PY_SCRIPT}" --dataset-dir "${DS}" --patch-info-only "${@:2}"
fi

if [[ ! -f "${DA3_WEIGHTS}/config.json" ]]; then
  echo "ERROR: no DA3 checkpoint at DA3_WEIGHTS='${DA3_WEIGHTS}'. Run: bash ${HERE}/setup.sh" >&2
  exit 1
fi

exec "${DA3_PY}" "${PY_SCRIPT}" \
  --dataset-dir "${DS}" \
  --da3-weights "${DA3_WEIGHTS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-shards "${NUM_SHARDS}" --shard-id "${SHARD_ID}" \
  "${@:2}"
