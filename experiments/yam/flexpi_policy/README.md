# YAM FlexPi deployment

Deployment-time entrypoints for the FlexPi model trained by
`scripts/yam_unified_joint_0509_4xh200_5ep.slurm`. Three layers, each
exercisable independently:

```
YamFlexPiPolicy   ← business logic: load model, build obs tensors,
                     denormalize + invert Yam32DRelativeAction,
                     return absolute 32D action chunk
       ↑
YamFlexPiBridge   ← per-step adapter: raiden-style obs in,
                     replan queue, abs-32 → robot command (14D / 20D)
       ↑
YamFlexPiRaidenBridge   ← raiden ModelBridge subclass:
                           pluggable into `rd infer --bridge ...`
```

Plus two CLIs:

- `smoke_test.py` — load checkpoint, run one inference offline, print
  diagnostics. Supports `--check-against-gt` for parquet MAE comparison
  and `--num-passes N` for steady-state latency measurement.
- `yam_bridge.py` (offline test mode) — exercise the bridge's queue +
  command-mapping logic against recorded data, no robot.

## Files

| File | Purpose |
|---|---|
| `deploy_policy.py` | `YamFlexPiPolicy` + `build_policy_from_checkpoint`. Owns: model load, processor, normalization round-trip, `Yam32DRelativeAction.backward`, `_warmup()`. |
| `yam_bridge.py` | `YamFlexPiBridge`: replan queue + `act() → robot command`. Includes the `_eef32_action_to_joint14_RL` / `_eef32_action_to_ee_pose20` permutations (verbatim from `yam_openpi/openpi_bridge.py`). Standalone CLI for offline testing. |
| `yam_raiden_bridge.py` | `YamFlexPiRaidenBridge(ModelBridge)`. Thin adapter — parses `--bridge-kwargs`, delegates to `YamFlexPiBridge`. Standalone CLI mirrors `openpi_bridge.main()`. |
| `smoke_test.py` | One-shot CLI: dummy / npz / recorded obs sources, GT comparison, latency profiling. |
| `requirements.txt` | Deploy-only extras (opencv-python). The flexpi package itself pulls everything heavy. |

## Quickstart

```bash
# 1. Set up the deploy venv (one time).
cd /path/to/FlexPi
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan22_weights
pip install -e .
pip install -r experiments/yam/flexpi_policy/requirements.txt

# 2. Smoke test against a trained ckpt.
PY=python
CKPT=runs/2026-05-13_*/step_008000.pt
DATA=data/clean_up_table_V2_merged_1_2_4_5

$PY -m experiments.yam.flexpi_policy.smoke_test \
    --ckpt $CKPT --data-dir $DATA \
    --episode-idx 0 --frame-idx 0 \
    --check-against-gt --verbose

# 3. Offline bridge test (queue cycling + command shape).
$PY -m experiments.yam.flexpi_policy.yam_bridge \
    --ckpt $CKPT --data-dir $DATA \
    --num-steps 16 --replan-steps 8 \
    --action-source model_joint --advance-frames --verbose

# 4. On the robot, from the raiden venv:
rd infer \
    --bridge experiments.yam.flexpi_policy.yam_raiden_bridge:YamFlexPiRaidenBridge \
    --ckpt_path /abs/path/to/step_XXXXXX.pt \
    --action_hz 30 --action-type joint \
    --bridge-kwargs prompt="..." \
    --bridge-kwargs action_source=model_joint \
    --bridge-kwargs replan_steps=32 \
    --bridge-kwargs num_inference_steps=4 \
    --bridge-kwargs torch_compile=true \
    --bridge-kwargs offload_text_encoder=true
```

## Action sources

Both pull from the model's 32D absolute action (`yam_eef.STATE_LAYOUT`):

| `action_source` | Output | Pair with raiden |
|---|---|---|
| `model_joint` (default) | `(14,)` float32 in raiden `[R_arm(6), R_grip, L_arm(6), L_grip]` | `--action-type joint` |
| `eef_ik` | `(20,)` EE pose `[l_xyz, r_xyz, l_rot6d, r_rot6d, l_grip, r_grip]` (with documented L↔R swap) | `--action-type ee_pose` |

Both are **absolute** targets. The `Yam32DRelativeAction.backward` inversion (using the current robot state as anchor) happens inside `YamFlexPiPolicy._denormalize_action` before the action_32 → robot-command mapping.

## Performance knobs

All four flow through identically: smoke CLI flag → bridge CLI flag →
`YamFlexPiRaidenBridge.load(**kwargs)` → `YamFlexPiBridge.__init__` →
`build_policy_from_checkpoint` → `YamFlexPiPolicy.__init__`.

| Knob | Default | Effect |
|---|---|---|
| `num_inference_steps` | 10 | Denoise step count. 4 ≈ 2.4× speedup with similar quality. |
| `torch_compile` | False | Compile denoise step + capture CUDA graph. ≈3× speedup, 10–30 s warmup absorbed at startup. |
| `torch_compile_mode` | `reduce-overhead` | `default` / `reduce-overhead` / `max-autotune`. |
| `offload_text_encoder` | False | Keep T5 on CPU, save ~10 GB VRAM. No latency hit. |
| `quantization` | None | `int8` is supported by the underlying model. |

`_warmup()` runs one dummy `infer_action` at the right shapes when
`torch_compile=True`, so the compile cost lands at startup rather than on
the first robot step. Mirrors RoboTwin `deploy_policy._warmup()`.

## What's actually validated

| Layer | Status |
|---|---|
| Module imports + class structure | ✅ |
| Both CLI `--help` parsing | ✅ |
| Mock-policy queue cycling (16 steps, replan=8) | ✅ |
| `Yam32DRelativeAction.backward` round-trip | ✅ (max abs diff 1.4e-4 — float32 rot6d noise) |
| Real-checkpoint smoke test (10 steps, no compile) | ✅ — 1440 ms inference, MAE EEF=0.010, joint=0.031 rad vs GT |
| Real-checkpoint bridge test (8 steps, model_joint) | ✅ — 1 fresh inference + 7 from queue, all (14,) shape, finite, in joint range |
| Perf-knob plumbing (torch_compile/offload) | ✅ via mock — string→bool/int parsing all good |
| Empirical perf measurement with real ckpt | ⏸ blocked: GPU contention (≤5 GB free, model needs ≥12 GB) |
| Real-robot deployment (`rd infer ...`) | ⏸ requires raiden + camera hardware |

## Known issues & workarounds

1. **`config.yaml` references training-time relative path
   `checkpoints/ActionDiT_linear_interp_*.pt`** that doesn't resolve at
   deploy time. `build_policy_from_checkpoint` overrides
   `action_dit_pretrained_path=None` and `skip_dit_load_from_pretrain=True`
   since `model.load_checkpoint(step_*.pt)` provides both DiT weight sets
   anyway. Saves ~2 GB of redundant I/O.

2. **`torch.min/max` not implemented for `uint16` on CUDA** — affected the
   verbose depth-range diagnostic print only. Now casts to int32 for the
   diagnostic; the underlying tensor flowing into the model stays uint16.

3. **Offline obs rebuild fork+thread deadlock.** After the model loads with
   its OMP threads, repeated `subprocess.check_output(ffmpeg)` calls in a
   tight loop can deadlock on glibc locks. Mitigated in the bridge offline
   CLI by gating obs rebuild on `bridge.needs_fresh_observation()` —
   we now only decode video files when the queue is empty. This is
   purely an offline-test artifact; real raiden delivers obs from ZED
   buffers without forking ffmpeg.

4. **VRAM pressure on a shared 32 GB GPU.** FlexPi Joint at full
   precision needs ~22 GB; with `offload_text_encoder=true` it drops to
   ~12 GB. If another tenant holds >20 GB, even the offload path OOMs.

## Deployment timing budget at `action_hz=30`

```
budget per chunk at action_hz=30 = replan_steps × (1/30) seconds
                                 = replan_steps × 33 ms
```

| Config | Inference latency (measured / projected) | replan_steps to keep up | Window |
|---|---|---|---|
| 10-step, no compile | 1440 ms (measured) | impossible (>32) | n/a |
| 4-step, no compile | ~600 ms (projected) | ≥18 | ~600 ms |
| 4-step + torch.compile + offload | ~180–250 ms (projected, RoboTwin-equivalent) | ≥8 | ~265 ms |

The conservative production config is `num_inference_steps=4
torch_compile=true offload_text_encoder=true replan_steps=32`.

## Maintenance notes

- **Don't modify the action mappings** in `yam_bridge.py`. They are
  byte-identical copies of `yam_openpi/deployment/openpi_bridge.py`'s
  `_eef32_action_to_joint14_RL` and `_eef32_action_to_ee_pose20`,
  including the documented L↔R swap rationale. Updating one without
  updating the other will silently desync raiden's motor convention.
- **Don't auto-detect the action source from the trained config.** The
  config doesn't carry deployment intent; some checkpoints will be used
  with both sources. Keep it as an explicit user choice.
- **The trained-config autoload path is the source of truth.** Never
  re-derive the model architecture from sim YAMLs — Hydra config drift
  will silently corrupt the joint flags / scheduler shifts.
