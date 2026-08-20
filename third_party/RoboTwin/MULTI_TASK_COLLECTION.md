# Multi-Task Data Collection

How to generate demonstration data for many RoboTwin tasks at once, parallelized across GPUs.

## Files

- [`collect_data.sh`](collect_data.sh) — single-task entry point (existing).
- [`collect_data_multi.sh`](collect_data_multi.sh) — multi-task wrapper (round-robin across GPUs).

## 1. Configure your task config

Pick (or copy) a config under [`task_config/`](task_config/), e.g. `demo_randomized.yml`.
Set what you need:

```yaml
episode_num: 550          # demos per task
data_type:
  rgb: true
  depth: true             # uint16 mm + gzip-4 (see envs/utils/pkl2hdf5.py)
  pointcloud: false       # leave off if you have depth + intrinsics
  third_view: false       # extra observer cam; off saves storage
  endpose: true
  qpos: true
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
save_freq: 15             # one (obs, qpos) every 15 sim steps
```

## 2. List available tasks

```bash
ls envs/*.py | xargs -n1 basename | sed 's/\.py$//' | grep -vE '^(_|test_)'
```

There are **50 tasks** in [`envs/`](envs/).

## 3. Run

### All tasks across both GPUs

```bash
./collect_data_multi.sh demo_randomized 0,1
```

### A subset across both GPUs

```bash
./collect_data_multi.sh demo_randomized 0,1 \
    click_bell lift_pot place_shoe handover_block
```

Or edit the `TASKS=( ... )` array near the top of `collect_data_multi.sh`,
uncomment/add the task names you want, and run without positional args.
Precedence: **CLI args > in-script `TASKS` > auto-discover all**.

### Single GPU

```bash
./collect_data_multi.sh demo_randomized 0
```

### Multiple workers per GPU (single-GPU parallelism)

Data collection is CPU-bound (physics + motion planning); the renderer is light. You can
fit multiple SAPIEN processes on one GPU and get a real speedup, as long as VRAM holds:

```bash
./collect_data_multi.sh --workers-per-gpu 2 demo_randomized 0       # 2 workers on GPU 0
./collect_data_multi.sh --workers-per-gpu 2 demo_randomized 0,1     # 2×2 = 4 workers
```

**Memory caveat.** Each worker holds its own SAPIEN context (~1-3 GB depending on scene).
Watch `nvidia-smi` after launching, and back off if you hit OOM. Reasonable defaults:
- 12 GB GPU: 1-2 workers
- 24 GB GPU: 2-4 workers
- 48 GB+ GPU: 4-8 workers

### Preview the plan without running

```bash
./collect_data_multi.sh --dry-run demo_randomized 0,1
```

Prints task → GPU assignment and exits.

### Skip tasks that are already complete

```bash
./collect_data_multi.sh --skip-done demo_randomized 0,1
```

A task is considered done when `data/<task>/<config>/seed.txt` already has at least
`episode_num` entries (parsed from `task_config/<config>.yml`). Useful when re-running
after a partial failure to avoid Python startup for finished tasks.

Flags can be combined: `--dry-run --skip-done` shows what *would* run.

### How it parallelizes

- Tasks are assigned **round-robin** to the listed GPUs (`tasks[i] → gpus[i % n_gpus]`).
- One worker per GPU, processing its tasks sequentially. No oversubscription.
- Per-task stdout/stderr → `logs/<task>_<config>_gpu<id>.log`.

## 4. Resume / re-run

Collection is **resumable**: each task's `data/<task>/<config>/seed.txt` records successful seeds.
Re-running the same command picks up where it left off ([script/collect_data.py:118-125](script/collect_data.py#L118-L125)).

To force a clean re-collect for one task:

```bash
rm -rf data/<task>/<config>
```

## 5. Storage planning

Per task at 550 demos, D435 (320×240), 3 cameras with RGB + depth (uint16+gzip):

| Stream | ~Per task | Notes |
|---|---|---|
| RGB (JPEG) | ~3 GB | per camera per frame ~10 KB |
| Depth (uint16+gzip) | ~10 GB | gzip ratio depends on scene clutter |
| State / misc | <100 MB | qpos, endpose |
| **Total** | **~13 GB / task** | range ~9-20 GB by episode length |

For all 50 tasks: **~650 GB**. Confirm `df -h` on `data/` before launching.

## 6. Monitoring

```bash
# Live tail across all logs
tail -f logs/*_${task_config}_gpu*.log

# Quick progress (success counts per task)
for d in data/*/${task_config}; do
    n=$(wc -w < "$d/seed.txt" 2>/dev/null || echo 0)
    printf "%-40s %4d / 550\n" "$(basename "$(dirname "$d")")" "$n"
done | sort -k2 -n
```

## 7. Notes

- The wrapper sets `CUDA_VISIBLE_DEVICES` per worker (via `collect_data.sh`); tasks pinned to a GPU stay there.
- Round-robin balances task **count**, not task **runtime**. Long tasks may stall their GPU while the other finishes early. If this matters, group long-running tasks together and assign them to one GPU explicitly via subset args.
- Depth storage format and config flags: see prior notes / [`envs/utils/pkl2hdf5.py`](envs/utils/pkl2hdf5.py) and [`envs/_base_task.py`](envs/_base_task.py).
