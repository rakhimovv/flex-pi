# Training a FlexPi Unified-Flex model

How to train the single checkpoint that serves every inference regime. What the
model is and why it works that way is [`OVERVIEW.md`](OVERVIEW.md); the
environment is [`INSTALL.md`](INSTALL.md); the per-benchmark recipes are
[`ROBOTWIN.md`](ROBOTWIN.md), [`LIBERO.md`](LIBERO.md) and [`YAM.md`](YAM.md).

- [1. Prerequisites](#1-prerequisites)
  - [1.1 Environment](#11-environment)
  - [1.2 Dataset layout](#12-dataset-layout-lerobot-v21-canonical-robotwin-cam-names)
  - [1.3 Precompute text embeddings](#13-precompute-text-embeddings-required)
- [2. Launching a run](#2-launching-a-run)
- [3. The flex knobs](#3-the-flex-knobs)
- [4. Finetune vs Resume](#4-finetune-vs-resume)
- [5. Troubleshooting](#5-troubleshooting)

---

## How a run is configured

Everything is Hydra, and one `task=` picks the entire run. `configs/train.yaml`
is the root and selects nothing by itself:

```
task=libero_unified_flex_2cam224_32d_rotvec_1e-4
 └── configs/task/<that name>.yaml     batch size, lr, epochs, save/eval cadence
      ├── override /data: libero    →  configs/data/libero.yaml
      │                                dataset dirs, shape_meta, action processor,
      │                                text_embedding_cache_dir
      └── override /model: flexpi   →  configs/model/flexpi.yaml
                                       architecture and the flex_joint defaults
```

So *which dataset* and *which architecture* are both consequences of the task
config, and every entry point that needs them takes the same `task=` — training,
the text-embed precompute, evaluation. Anything after it on the command line is a
Hydra override and wins over all three files:

```bash
bash scripts/train_flexpi_libero.sh batch_size=4 num_epochs=10
```

The three shipped task configs are one per setting; each benchmark doc names its
own.

---

## 1. Prerequisites

### 1.1 Environment

Finish [`INSTALL.md`](INSTALL.md) first — environment, Wan2.2 weights, DINOv3.
Everything below runs from the project root:

```bash
conda activate flexpi
cd flex-pi
```

### 1.2 Dataset layout (LeRobot v2.1, canonical RoboTwin cam names)

Every dataset directory must be a LeRobot **v2.1** dataset with **canonical camera
keys** and, for the 3D stream, **depth videos + camera intrinsics**:

```
<dataset_dir>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl                       # language instructions (used by text-embed precompute)
│   └── camera_intrinsics.json            # REQUIRED for the pointmap/3D stream
└── videos/chunk-000/
    ├── observation.images.cam_high/        episode_000000.mp4 ...
    ├── observation.images.cam_left_wrist/  ...
    ├── observation.images.cam_right_wrist/ ...
    ├── observation.depth_ffv1.cam_high/    episode_000000.mkv ...   # FFV1 gray16 depth
    ├── observation.depth_ffv1.cam_left_wrist/ ...
    └── observation.depth_ffv1.cam_right_wrist/ ...
```

`meta/info.json` must declare `"codebase_version": "v2.1"` — that is what the
vendored loader in [`src/flexpi/datasets/lerobot/`](../src/flexpi/datasets/lerobot/)
is pinned to, and what every published dataset and the YAM builder write. v2.0 still
reads, by falling back to `meta/stats.json` for the per-episode statistics; v3.0,
which reorganizes the files on disk, does not.

The loader looks for exactly those keys, and the pointmap stream needs
`meta/camera_intrinsics.json` beside them. How both are produced — the raw layout
in, the LeRobot layout out, and the intrinsics convention — is the dataset builder:
[`YAM.md` step 1](YAM.md#1-data-build).

> **All datasets you combine must share the same fps, camera layout, and action
> dimensionality.** This is enforced at runtime — an fps mismatch raises.

### 1.3 Precompute text embeddings (required)

T5 embeddings are cached to disk; training dies on the first batch without them.
The cache has to cover **every prompt in every dataset dir you train on** —
lookup is by prompt hash, and narrowing `TASK_NAMES` does not narrow what the
cache needs.

```bash
python scripts/precompute_text_embeds.py task=<TASK_CONFIG>

# Your own dataset, on an existing task config — pass the same dirs you will
# train on. Without this it scans the placeholder path in the data config:
python scripts/precompute_text_embeds.py \
  task=yam_unified_flex_3cam_32d_rel_1e-4 \
  data.train.dataset_dirs='[./data/dataset_a]'

# A dataset with a data config but no task config yet:
bash scripts/run_precompute_text_embeds.sh <data_config_name>
```

There is no `--dataset-dir` flag because a dataset directory is not enough to
run the pass. The prompts come from each dir's `meta/tasks.jsonl`, but the output
goes to `text_embedding_cache_dir` — a directory of its own, outside the dataset,
shared by every dir in that data config — and the filenames carry the data
config's `context_len`. Both are properties of the config, which is why `task=`
(or `data=`) is the key on both sides: it is what makes the precompute and the
training run agree on where the cache is.

A re-run only encodes what the cache is missing. `scripts/train_flexpi_yam.sh`
runs this for you; the other two launchers do not.

---

## 2. Launching a run

One launcher per setting — RoboTwin, LIBERO, real-world YAM. They are maintained
in lockstep and expose the same knobs, differing only in defaults; each one's
config block is the reference for what it sets.

```bash
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export DATASET_DIRS='[./data/yam/my_task]'

bash scripts/train_flexpi_yam.sh batch_size=3
```

**Not every knob reads the environment, and which ones do differs per launcher.**
A line written `VAR=value` is in-script only — exporting it does nothing. A line
written `VAR="${VAR:-value}"` honours an export. Every `FLEX_P_*` is in-script in
all three launchers, and `TASK_NAMES` is in-script in the RoboTwin one. Check the
launcher before relying on `export`.

The path that always works is a **Hydra override on the command line** — every
launcher appends `"$@"` last, so a CLI value beats whatever the script set:

```bash
bash scripts/train_flexpi_robotwin.sh model.flex_joint.p_jv=1.0 batch_size=4
```

All three launch through `accelerate` with
`scripts/accelerate_configs/accelerate_zero1_ds.yaml`; `ACCELERATE_CONFIG` points
them at one of the others in that directory (ZeRO-2, DDP, single-GPU, ZeRO-0).

---

## 3. The flex knobs

Every training step, each stream is independently sampled as **present or absent**, and
each present stream is independently sampled as **jointly fused or not**. This is what lets
one checkpoint serve all 8 regimes.

The `FLEX_P_*` knobs are plain config-block assignments in all three launchers — edit
the file; `export` does not reach them. `robotwin.sh` and `yam.sh` ship the `0.5`
defaults below; `libero.sh` ships all six at `1.0`.

| Knob | Default | Meaning |
|------|---------|---------|
| `FLEX_P_PRESENT_VIDEO` | `0.5` | P(video stream provided this sample) |
| `FLEX_P_PRESENT_DINO` | `0.5` | P(dino stream provided this sample) |
| `FLEX_P_PRESENT_POINTMAP` | `0.5` | P(3D/pointmap stream provided this sample) |
| `FLEX_P_JV` / `FLEX_P_JD` / `FLEX_P_JP` | `0.5` | P(video / dino / pointmap is *jointly* fused, i.e. action attends all of its tokens) |
| `model.flex_joint.cross_modal_predict_video` / `_dino` / `_pointmap` (Hydra override, not an env var) | `true` | when a stream is **absent**, still denoise its tokens conditioned on the other streams (the "forecasting" regime) instead of fully masking it |

Tips:

- **Set a probability to `1.0` to disable that dropout** (stream always present). E.g.
  `FLEX_P_PRESENT_DINO=1.0` trains DINO as an always-on conditioning input.
- **Set to `0.0` to switch a stream fully off** for the run. For the pointmap the
  master switch is `model.enable_pointmap=false` (pair it with `data=<name>_nodepth`
  so the loader stops decoding depth nothing will read).
- `model.joint_video` / `_dino` / `_pointmap` (default `true`, Hydra override) set the
  *trained default* joint flags used when `infer_action` is called at eval **without**
  runtime overrides; per-sample sampling still randomizes every step regardless.
- The chosen probabilities are baked into the run name / `REGIME_TAG`
  (e.g. `flex_pv0.5_pd0.5_pp0.5_jv0.5_jd0.5_jp0.5`) so runs are
  self-documenting.

Example — train video always-on, dino/3d dropped 50%, no cross-modal forecasting:

```bash
# in scripts/train_flexpi_robotwin.sh
FLEX_P_PRESENT_VIDEO="1.0"
FLEX_P_PRESENT_DINO="0.5"
FLEX_P_PRESENT_POINTMAP="0.5"
```

Cross-modal forcing has no launcher variable at all — pass it as a Hydra override,
which every launcher forwards verbatim:

```bash
bash scripts/train_flexpi_robotwin.sh \
  model.flex_joint.cross_modal_predict_video=false \
  model.flex_joint.cross_modal_predict_dino=false \
  model.flex_joint.cross_modal_predict_pointmap=false
```

---

## 4. Finetune vs Resume

Mutually exclusive; `RESUME` wins if both are set.

```bash
# Weights only — fresh warmup + cosine, new output directory tagged _ft:
PRETRAINED_CKPT=/path/to/run/checkpoints/weights/step_030000.pt \
  bash scripts/train_flexpi_robotwin.sh

# Weights + optimizer + LR schedule + step counter:
RESUME=/path/to/run/state/step_030000 bash scripts/train_flexpi_robotwin.sh
```

`PRETRAINED_CKPT_STRICT_SHAPE=true` aborts on any shape mismatch; the default
`false` re-initializes whatever differs. `action_dim`, `composite_layout` and
`dino_pixel_unshuffle` all count, so a warm start has to match the donor's
geometry rather than today's default.

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `All dataset_dirs must have the same fps` | one dir has a different fps; rebuild it or drop it from `DATASET_DIRS`. |
| `camera_intrinsics.json missing` | the 3D stream needs intrinsics per dir; write them, or disable 3D with `model.enable_pointmap=false data=<benchmark>_nodepth`. |
| `samples_per_epoch must be a positive int when dataset_weights is set` | set `SAMPLES_PER_EPOCH` whenever you pass `DATASET_WEIGHTS`. |
| `dataset_weights (N) must be parallel to dataset_dirs (M)` | the weight list length must equal the number of dirs. |
| Realized mixture ≠ weights | weights set the **frame** draw ratio; check the logged `frame counts` and confirm `SAMPLES_PER_EPOCH` is large enough for the ratio to converge. |
| Strict-shape load aborts on finetune | you set `PRETRAINED_CKPT_STRICT_SHAPE=true` and `composite_layout` / `dino_pixel_unshuffle` / `action_dim` differ from the source ckpt; match them, or drop back to the default `false` to let the mismatched modules re-initialize. |
| OOM | lower `batch_size`; a larger DINO grid (`dino_pixel_unshuffle=0`) and the pointmap stream both raise memory, and `model.mot_checkpoint_mixed_attn=true` trades compute for activations. If it still OOMs at `batch_size=1`, the card is too small rather than the batch too big — a 32 GB card runs the forward/eval pass but not the backward. |
| `Missing text embedding cache: <hash>.t5_len128...pt` | re-run `precompute_text_embeds.py` with the **same** `data.train.dataset_dirs` you train on. Narrowing `TASK_NAMES` for training does **not** narrow the cache it needs — the prompt is looked up by hash, so every instruction your episodes carry must have been encoded. |
| `Checkpoint was saved with pointmap_mode=...` / `disable_video_stream=True ... refusing to load` | an ablation-only mode that no longer exists; no migration path — retrain |
| `TypeError: __init__() got an unexpected keyword argument 'dino_temporal_stride'` | you pointed a tool at a **saved run** `config.yaml`. Saved configs carry loader keys the merged `RobotVideoDataset` dropped. Eval never instantiates the dataset, so compose a fresh config instead (`task=...`). |
