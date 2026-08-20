"""
Batch eval: load FlexPi model ONCE, then run a list of (suite, task_id) pairs.

This avoids the ~3-minute model reload per task (Wan2.2-TI2V-5B is 12GB).
Essential when evaluating LIBERO-Plus (thousands of tasks × 1 trial).

Reads task list from --tasks_file (format: `suite,task_id` per line).
Writes per-task result JSON to:
    {output_dir}/{suite}/gpu{gpu_id}_task{task_id}_results.json
(same format as eval_libero_single.py, so summarize_results_*.py works unchanged).

Usage (single GPU):
    python experiments/libero/eval_libero_batch.py \\
        task=libero_unified_flex_2cam224_32d_rotvec_1e-4 \\
        ckpt=$CKPT \\
        EVALUATION.dataset_stats_path=$STATS \\
        EVALUATION.num_trials=50 \\
        EVALUATION.output_dir=/path/out \\
        EVALUATION.tasks_file=/path/tasks_shard.txt \\
        gpu_id=0
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Resolve project root (FlexPi/) so `from experiments.libero...` works
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Re-use everything from single-task eval
from experiments.libero.eval_libero_single import (
    _is_3d_eval,
    _load_eval_camera_intrinsics,
    _load_model_checkpoint,
    _maybe_swap_eval_cfg_with_saved,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_episode,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_env,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero.task_pool import pop_group, remaining as pool_remaining
from flexpi.datasets.lerobot.processors.flexpi_processor import FlexPiProcessor
from flexpi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from flexpi.utils.pytorch_utils import set_global_seed
from flexpi.utils.libero_setup import assert_mujoco_pin, prepare_libero

# Idempotent; the eval_libero_single/libero_utils imports above already ran it.
# Repeated here so the guarantee survives an import re-sort.
prepare_libero()
# Refuse to produce numbers on a renderer other than the one that made the
# training data. See assert_mujoco_pin's docstring for why this is fatal.
assert_mujoco_pin()

from libero.libero import benchmark


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _read_tasks_file(path: str) -> list[tuple[str, int]]:
    """Parse `suite,task_id` lines from a task file."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            suite, tid = line.split(",", 1)
            tasks.append((suite.strip(), int(tid.strip())))
    return tasks


def _run_one_task(
    *,
    task,
    initial_states,
    env,
    task_description: str,
    model,
    processor,
    cfg,
    video_dir,
    predicted_video_dir,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    """Run all trials for one task on a pre-built env. Mirrors
    run_single_task's trial loop but skips env creation so the caller can
    reuse a single env across multiple tasks sharing a bddl_file.
    """
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video and predicted_future_video_clips:
            all_gt, all_pred = [], []
            for clip in predicted_future_video_clips:
                all_gt.extend(clip["gt_frames"])
                all_pred.extend(clip["pred_frames"])
                save_prediction_video(
                    predicted_video_dir,
                    clip["gt_frames"],
                    clip["pred_frames"],
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    clip["replan_idx"],
                    success=success,
                    task_description=task_description,
                )
            save_prediction_video(
                predicted_video_dir,
                all_gt,
                all_pred,
                f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                "all",
                success=success,
                task_description=task_description,
            )

    if visualize_future_video:
        valid = [x for x in results["episode_future_video_psnr"] if x is not None]
        if valid:
            results["future_video_psnr_mean"] = float(sum(valid) / len(valid))
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_batch(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    cfg = _maybe_swap_eval_cfg_with_saved(cfg)
    partial_state.config = cfg
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Only env_num=1 is supported in eval_libero_batch.py.")

    pool_file = cfg.EVALUATION.get("pool_file", None)
    tasks_file = cfg.EVALUATION.get("tasks_file", None)
    if not pool_file and not tasks_file:
        raise ValueError(
            "EVALUATION.pool_file (shared pool, dynamic) or "
            "EVALUATION.tasks_file (static shard) is required for batch mode."
        )
    if pool_file:
        pool_file = str(pool_file)
        print(f"[batch/pool] Worker will pull from {pool_file} "
              f"({pool_remaining(pool_file)} groups visible at startup)")
    else:
        tasks_file = str(tasks_file)
        task_list = _read_tasks_file(tasks_file)
        if not task_list:
            raise ValueError(f"No tasks loaded from {tasks_file}")
        print(f"[batch] Loaded {len(task_list)} tasks from {tasks_file}")

    # ──────────────── Load model ONCE ────────────────
    print(f"[batch] Loading model (this takes ~3min)...")
    load_start = time.time()
    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    if hasattr(model, "prepare_for_inference"):
        _torch_compile = bool(cfg.EVALUATION.get("torch_compile", False))
        _torch_compile_mode = str(cfg.EVALUATION.get("torch_compile_mode", "reduce-overhead"))
        _quantization_raw = cfg.EVALUATION.get("quantization", None)
        _quantization = None if _quantization_raw is None or str(_quantization_raw).strip().lower() in ("", "none", "null") else str(_quantization_raw)
        model.prepare_for_inference(
            torch_compile=_torch_compile,
            torch_compile_mode=_torch_compile_mode,
            quantization=_quantization,
        )

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FlexPiProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    print(f"[batch] Model loaded in {time.time() - load_start:.1f}s")

    # FlexPi3D: load camera intrinsics ONCE and cache on the model. Mirrors
    # eval_libero_single's eval_single_process — without it the pointmap
    # encoder fails on every task with "No camera intrinsics available".
    # All LIBERO suites share identical K, so a single load suffices for
    # the whole 1680-task batch.
    if _is_3d_eval(cfg):
        K = _load_eval_camera_intrinsics(cfg)
        model.set_camera_intrinsics(K.to(model_device))
        print(f"[batch] FlexPi3D: cached camera intrinsics K={tuple(K.shape)} "
              f"from {cfg.EVALUATION.get('camera_intrinsics_path')}")

    # ──────────────── Loop over tasks ────────────────
    output_root = Path(cfg.EVALUATION.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    # Cache benchmark instances per suite (creating them is cheap but avoid repeat)
    suite_cache: dict = {}

    num_trials = int(cfg.EVALUATION.num_trials)
    completed = 0
    failed = 0
    worker_id = os.environ.get("WORKER_ID", f"gpu{cfg.gpu_id}_pid{os.getpid()}")
    recent_durations: list[float] = []  # rolling window for ETA

    def _log_progress(duration: float):
        recent_durations.append(duration)
        if len(recent_durations) > 10:
            recent_durations.pop(0)
        avg = sum(recent_durations) / len(recent_durations)
        if pool_file:
            left = pool_remaining(pool_file)
        else:
            left = max(0, len(task_list) - completed - failed)  # noqa: F821
        eta_h = (left * avg) / 3600 if avg > 0 else 0.0
        print(
            f"[progress] worker={worker_id} done={completed} failed={failed} "
            f"left~={left} last={duration:.1f}s avg10={avg:.1f}s ETA~{eta_h:.2f}h",
            flush=True,
        )

    def _process_task(suite, task_id, env, task_description, banner):
        """Run one task's trials on a (possibly reused) env. Writes the
        per-task result JSON. Caller handles env lifetime.
        """
        nonlocal completed, failed
        task_start = time.time()
        print(f"\n[batch] {banner} {suite} task_id={task_id} ===")
        try:
            suite_dir = output_root / suite
            video_dir = suite_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            predicted_video_dir = suite_dir / "predicted_videos"
            if bool(cfg.EVALUATION.get("visualize_future_video", False)):
                predicted_video_dir.mkdir(parents=True, exist_ok=True)

            task_suite = suite_cache[suite]
            task = task_suite.get_task(task_id)
            initial_states = list(task_suite.get_task_init_states(task_id))
            while len(initial_states) < num_trials:
                initial_states.extend(initial_states[: (num_trials - len(initial_states))])

            cfg.EVALUATION.task_suite_name = suite
            cfg.EVALUATION.task_id = task_id

            results = {
                "task_suite": suite,
                "task_id": task_id,
                "task_description": None,
                "successes": 0,
                "total_episodes": num_trials,
                "gpu_id": int(cfg.gpu_id),
                "success_episodes": [],
                "failure_episodes": [],
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": 0,
            }
            task_results = _run_one_task(
                task=task,
                initial_states=initial_states,
                env=env,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                video_dir=video_dir,
                predicted_video_dir=predicted_video_dir,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            results.update(task_results)
            results["duration"] = time.time() - task_start

            output_file = suite_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, cls=NumpyEncoder)
            print(
                f"[batch] Task {task_id} done: {results['successes']}/{num_trials} successes "
                f"in {results['duration']:.1f}s"
            )
            completed += 1
            _log_progress(results["duration"])
        except Exception as e:
            logging.exception(f"[batch] Task {suite} task_id={task_id} failed: {e}")
            failed += 1
            _log_progress(time.time() - task_start)

    def _ensure_suite(suite):
        if suite not in suite_cache:
            suite_cache[suite] = benchmark_dict[suite]()
        return suite_cache[suite]

    if pool_file:
        # ──────────── Pool mode: pull groups, reuse env within a group ────────────
        groups_done = 0
        tasks_done = 0
        while True:
            group = pop_group(pool_file)
            if group is None:
                break
            tasks_in_group = group.get("tasks", [])
            if not tasks_in_group:
                continue
            groups_done += 1

            # Build env from the first task; reuse for the rest of the group.
            # All tasks in a group share task.bddl_file by construction, so
            # task_description is also identical (see libero_utils.get_libero_env).
            first_suite, first_tid = tasks_in_group[0]
            _ensure_suite(first_suite)
            first_task_obj = suite_cache[first_suite].get_task(int(first_tid))
            env, task_description = get_libero_env(
                first_task_obj,
                LIBERO_ENV_RESOLUTION,
                cfg.get("seed"),
                camera_depths=_is_3d_eval(cfg),
                prompt_source=str(cfg.EVALUATION.get("prompt_source", "task.language")),
            )
            try:
                for j, (suite, tid) in enumerate(tasks_in_group):
                    _ensure_suite(suite)
                    banner = f"=== Group {groups_done} task {j + 1}/{len(tasks_in_group)}:"
                    _process_task(suite, int(tid), env, task_description, banner)
                    tasks_done += 1
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        total_duration = time.time() - start_time
        per_task = total_duration / max(tasks_done, 1)
        print(
            f"\n[batch/pool] Worker finished: {groups_done} groups, "
            f"{completed} tasks OK, {failed} failed, "
            f"total {total_duration:.1f}s ({per_task:.1f}s/task avg)"
        )
    else:
        # ──────────── Static-shard mode (legacy): build env per task ────────────
        for idx, (suite, task_id) in enumerate(task_list):
            _ensure_suite(suite)
            task_obj = suite_cache[suite].get_task(task_id)
            env, task_description = get_libero_env(
                task_obj,
                LIBERO_ENV_RESOLUTION,
                cfg.get("seed"),
                camera_depths=_is_3d_eval(cfg),
                prompt_source=str(cfg.EVALUATION.get("prompt_source", "task.language")),
            )
            try:
                banner = f"=== Task {idx + 1}/{len(task_list)}:"
                _process_task(suite, task_id, env, task_description, banner)
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        total_duration = time.time() - start_time
        print(
            f"\n[batch] Finished: {completed} OK, {failed} failed, "
            f"total {total_duration:.1f}s ({total_duration / max(len(task_list), 1):.1f}s/task avg)"
        )


if __name__ == "__main__":
    eval_batch()
