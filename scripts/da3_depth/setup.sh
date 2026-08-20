#!/usr/bin/env bash
# =============================================================================
# One-time setup for the DA3 depth labeling pass: build the env, install
# Depth Anything 3, download the DA3METRIC checkpoint, and verify that the
# checkpoint actually LOADS and produces metric depth.
#
# On success it writes scripts/da3_depth/env.sh with the resolved interpreter
# and weights paths; run.sh sources that, so nothing downstream hardcodes a
# path. Idempotent — re-running skips whatever already exists.
#
#   bash scripts/da3_depth/setup.sh
#
# Overrides (all optional):
#   CONDA_DIR    conda install to build the env in   (default: $HOME/miniforge3)
#   ENV_NAME     env name                            (default: da3_depth)
#   DA3_SRC      where to clone Depth-Anything-3     (default: $HOME/projects/Depth-Anything-3)
#   DA3_WEIGHTS  where to put the checkpoint         (default: $HOME/weights/DA3METRIC-LARGE)
#   DA3_MODEL    HF repo id                          (default: depth-anything/DA3METRIC-LARGE)
#   PY_VER       python version                      (default: 3.12)
#
# Pinned stack (the verified-working set — torch 2.10/cu128 is what the DA3
# attention path was exercised against; older torch may work but is untested):
#   python 3.12 · torch 2.10.0+cu128 · torchvision 0.25.0 · xformers 0.0.35
#   numpy 1.26.4 (<2 required) · ffmpeg 7 (LGPL, from conda-forge)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-da3_depth}"
DA3_SRC="${DA3_SRC:-$HOME/projects/Depth-Anything-3}"
DA3_WEIGHTS="${DA3_WEIGHTS:-$HOME/weights/DA3METRIC-LARGE}"
DA3_MODEL="${DA3_MODEL:-depth-anything/DA3METRIC-LARGE}"
PY_VER="${PY_VER:-3.12}"
ENV_PREFIX="$CONDA_DIR/envs/$ENV_NAME"

echo "=== DA3 depth setup ==="
echo "  conda   : $CONDA_DIR"
echo "  env     : $ENV_NAME  ->  $ENV_PREFIX"
echo "  DA3 src : $DA3_SRC"
echo "  weights : $DA3_WEIGHTS   (from $DA3_MODEL)"
echo

[[ -d "$CONDA_DIR" ]] || { echo "ERROR: no conda at $CONDA_DIR (set CONDA_DIR=)"; exit 1; }
# The base interpreter has to actually start — a purged/half-deleted conda
# fails here rather than three steps later with a confusing import error.
"$CONDA_DIR/bin/python" -c "import encodings" 2>/dev/null || {
  echo "ERROR: $CONDA_DIR base python is broken. Point CONDA_DIR at a healthy conda."; exit 1; }

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
if [[ -x "$ENV_PREFIX/bin/python" ]]; then
  echo "[1/5] env exists — reusing"
else
  echo "[1/5] creating env (python $PY_VER)"
  conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi
conda activate "$ENV_NAME"
PY="$ENV_PREFIX/bin/python"

echo "[2/5] torch 2.10.0+cu128 / torchvision 0.25.0 / xformers / numpy<2"
"$PY" -m pip install --upgrade pip -q
"$PY" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install xformers==0.0.35 numpy==1.26.4 einops safetensors huggingface_hub

echo "[3/5] ffmpeg 7 (LGPL) into the env"
# The pass shells out to ffmpeg/ffprobe for all video IO. Installing them INTO
# the env means the pinned build is used regardless of what the host has on PATH
# — and gray16le FFV1 encoding is not universally present in distro builds.
conda install -p "$ENV_PREFIX" -c conda-forge 'ffmpeg=7.*=lgpl_*' -y

echo "[4/5] depth_anything_3 (editable clone)"
if [[ ! -e "$DA3_SRC/pyproject.toml" ]]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DA3_SRC"
fi
# open3d is declared as a hard dep but is imported only under bench/ (benchmark
# eval), never on the depth/api/model path we use — and it has no wheel for
# py3.12 on some platforms, which blocks the whole editable install.
if grep -q '"open3d"' "$DA3_SRC/pyproject.toml"; then
  echo "      dropping bench-only open3d dep (backup: pyproject.toml.bak)"
  sed -i.bak '/"open3d"/d' "$DA3_SRC/pyproject.toml"
fi
"$PY" -m pip install -e "$DA3_SRC"
# addict is an UNDECLARED runtime import of model/da3.py (`from addict import Dict`),
# so the editable install does not pull it, but the api import chain needs it.
"$PY" -m pip install addict

echo "[5/5] checkpoint: $DA3_MODEL"
if [[ -f "$DA3_WEIGHTS/config.json" && -f "$DA3_WEIGHTS/model.safetensors" ]]; then
  echo "      already present at $DA3_WEIGHTS — skipping download"
else
  mkdir -p "$DA3_WEIGHTS"
  # snapshot_download, not huggingface-cli: the CLI was renamed to `hf` in
  # recent hub releases, so the old command name errors out.
  "$PY" - "$DA3_MODEL" "$DA3_WEIGHTS" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], local_dir=sys.argv[2])
print("      downloaded ->", p)
PY
fi

echo
echo "=== verify: import, load the checkpoint, run one frame ==="
"$PY" - "$DA3_WEIGHTS" <<'PY'
import shutil, sys, torch, numpy
from depth_anything_3.api import DepthAnything3

w = sys.argv[1]
print("torch    :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy    :", numpy.__version__)
ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
print("ffmpeg   :", ff)
print("ffprobe  :", fp)
assert ff and fp, "ffmpeg/ffprobe missing from the env"

m = DepthAnything3.from_pretrained(w).eval()
name = m.model_name.upper()
is_metric = "DA3METRIC" in name and "NESTED" not in name
print("model    :", m.model_name, "| metric:", is_metric)
assert is_metric or "NESTED" in name, (
    f"{m.model_name} is a RELATIVE-depth model; the labeling pass needs a "
    f"DA3METRIC (or DA3NESTED) checkpoint to emit millimetres.")

if torch.cuda.is_available():
    m = m.cuda()
    # 224x224 (a multiple of the 14px patch) of mid-grey, with a plausible K:
    # exercises the exact forward + metric-rescale path the pass uses.
    x = torch.full((1, 1, 3, 224, 224), 0.5, device="cuda")
    with torch.no_grad():
        raw = m.forward(x, export_feat_layers=[])["depth"]
    focal = 262.2
    depth_m = raw * (focal / 300.0) if is_metric else raw
    print(f"forward  : {tuple(depth_m.shape)} | median {float(depth_m.median())*1000:.0f} mm")
print("=> DA3 READY")
PY

cat > "$HERE/env.sh" <<EOF
# Written by scripts/da3_depth/setup.sh — sourced by run.sh.
# Re-run setup.sh to regenerate; override either value in your shell to win.
DA3_PY="\${DA3_PY:-$ENV_PREFIX/bin/python}"
DA3_WEIGHTS="\${DA3_WEIGHTS:-$DA3_WEIGHTS}"
EOF

echo
echo "wrote $HERE/env.sh:"
sed 's/^/  /' "$HERE/env.sh"
echo
echo "Next:  CHECK=1 bash scripts/da3_depth/run.sh <dataset_dir>"
