"""Client-side step-indexed RTC stitching broker for the FlexPi WS bridge.

Drop-in replacement for ``ActionChunkBroker`` when bridge-kwarg ``use_rtc=true``.

Algorithm
---------
Both bridge and broker track a local step counter. When the broker submits a
WS inference request at step ``S_obs``, it remembers that the server's chunk
row 0 corresponds to step ``S_obs``. The response arrives at a later step
``S_recv``, so ``offset = S_recv - S_obs`` rows of the new chunk are already
in the past.

Let ``useful_new = new[offset:]`` (length ``M = H - offset``) and let the
existing queue have length ``L``. The merge proceeds in three regions:

1. **Cosine blend** — first ``min(merge_steps, overlap)`` rows of the overlap
   are blended with cosine weights ``w_i = 0.5 - 0.5*cos(pi*(i+1)/(blend_n+1))``
   from 0 → 1.
2. **Hard overwrite** — the rest of the overlap commits to the new chunk
   immediately (no smoothing).
3. **Append** — non-overlapping new rows extend the queue past the old tail.

If ``offset >= H``, the entire chunk is stale and gets dropped.

Replanning is **asynchronous**: a single-worker ``ThreadPoolExecutor`` runs
the WS request off the heartbeat thread, so ``infer()`` only blocks on cold
start or when the queue drains (sync fallback).

Public interface mirrors ``ActionChunkBroker``:

    broker = RTCStepBroker(ws_policy, action_horizon, action_dim, ...)
    out = broker.infer(obs)             # {"actions": np.ndarray shape (D,)}
    broker.reset()
    broker.stats                        # diagnostics dict
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Deque, Dict, Optional

import numpy as np


LOG_PREFIX = "[client-rtc-broker]"
logger = logging.getLogger(__name__)


class RTCStepBroker:
    """Step-indexed RTC stitching broker (client-side, no server changes)."""

    def __init__(
        self,
        ws_policy: Any,
        action_horizon: int,
        action_dim: int,
        merge_steps: int = 4,
        replan_steps: int = 8,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be > 0; got {action_horizon}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0; got {action_dim}")
        if merge_steps < 0:
            raise ValueError(f"merge_steps must be >= 0; got {merge_steps}")
        if replan_steps <= 0:
            raise ValueError(f"replan_steps must be > 0; got {replan_steps}")

        self._ws_policy = ws_policy
        self._H = int(action_horizon)
        self._D = int(action_dim)
        self._merge_steps = int(merge_steps)
        self._replan_steps = int(replan_steps)
        self._queue: Deque[np.ndarray] = deque()
        self._step: int = 0
        self._calls_since_last_submit: int = 0

        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)
        self._inflight: Optional[Future] = None
        self._inflight_obs_send_step: Optional[int] = None

        # All queue mutations happen on the heartbeat thread inside infer(),
        # so the lock is defensive (covers a future where someone wires a
        # completion callback on the executor side).
        self._lock = threading.Lock()

        self._stats: Dict[str, int] = {
            "submits": 0,              # async submissions issued
            "cold_starts": 0,          # sync infer fired with NO inflight pending
            "drain_recoveries": 0,     # sync infer BLOCKED on a pending inflight
                                       # → old action tail fully consumed before new arrived
            "stale_drops": 0,          # chunks rejected because offset >= H
            "merges": 0,               # successful merges
            "merges_just_in_time": 0,  # merges where L == 0 (drained at merge time)
                                       # but no sync block was needed
            "last_offset": 0,
            "last_blend_n": 0,
            "last_L": 0,               # queue len at last merge (0 ⇒ drained)
            "queue_len": 0,            # populated by .stats property
        }

    # ------------------------------------------------------------------ API
    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            self._stats["queue_len"] = len(self._queue)
            return dict(self._stats)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """One pop per call. Returns {'actions': ndarray shape (D,)}.

        Blocks only on:
          - cold start (queue empty, no inflight)
          - queue drain (inflight not yet done and queue empty)
        Otherwise returns the next queued row immediately.
        """
        # 1. Pump any completed inflight.
        self._pump_inflight()

        # 2. Maybe trigger a fresh async submission.
        if self._inflight is None and self._calls_since_last_submit >= self._replan_steps:
            self._submit_async(obs)

        # 3. Make sure the queue has something to pop.
        if len(self._queue) == 0:
            # Cold start or genuine drain. Block on a synchronous request.
            self._sync_fallback(obs)

        if len(self._queue) == 0:
            raise RuntimeError(
                f"{LOG_PREFIX} queue is empty after sync fallback; "
                "server must be returning empty 'actions'."
            )

        # 4. Pop and return.
        with self._lock:
            cmd = self._queue.popleft()
        self._step += 1
        self._calls_since_last_submit += 1

        return {"actions": cmd}

    def reset(self) -> None:
        # Cancel any inflight; won't actually stop a running task, but we'll
        # ignore its result via the obs_send_step / inflight-cleared check
        # below.
        if self._inflight is not None:
            self._inflight.cancel()
            self._inflight = None
            self._inflight_obs_send_step = None

        with self._lock:
            self._queue.clear()

        self._step = 0
        self._calls_since_last_submit = 0
        for k in self._stats:
            self._stats[k] = 0

        try:
            self._ws_policy.reset()
        except Exception:  # noqa: BLE001
            logger.warning("%s ws_policy.reset() raised; ignoring", LOG_PREFIX)

    def close(self) -> None:
        """Shut down the background executor. Optional; safe to call once."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---------------------------------------------------------------- internals

    def _pump_inflight(self) -> None:
        if self._inflight is None or not self._inflight.done():
            return
        obs_send_step = self._inflight_obs_send_step
        future = self._inflight
        self._inflight = None
        self._inflight_obs_send_step = None

        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s inflight request failed: %r", LOG_PREFIX, exc)
            return

        new_chunk = self._extract_actions(result)
        if new_chunk is None or obs_send_step is None:
            return
        self._merge_response(obs_send_step, new_chunk)

    def _submit_async(self, obs: Dict[str, Any]) -> None:
        # Don't pin the caller's obs object: copy the ndarray leaves so
        # raiden can keep mutating its buffers without affecting the inflight
        # request payload.
        obs_copy = self._shallow_copy_obs(obs)
        self._inflight_obs_send_step = self._step
        self._inflight = self._executor.submit(self._ws_policy.infer, obs_copy)
        self._calls_since_last_submit = 0
        self._stats["submits"] += 1
        print(
            f"{LOG_PREFIX} submit-async  step={self._step:4d} "
            f"obs_send_step={self._step:4d} queue_len={len(self._queue):3d}",
            flush=True,
        )

    def _sync_fallback(self, obs: Dict[str, Any]) -> None:
        # If there's an inflight, the old action tail has been FULLY CONSUMED
        # before the new chunk arrived — block on the inflight to recover.
        # This is the "bad" drain case; surface it loudly so tuning can fix it
        # (typically: lower replan_steps, or the network is degraded).
        if self._inflight is not None:
            inflight_obs = self._inflight_obs_send_step
            print(
                f"{LOG_PREFIX} DRAIN-RECOVERY  step={self._step:4d} "
                f"inflight_obs_send_step={inflight_obs} "
                f"— old action tail fully consumed before new arrived; "
                f"blocking on inflight",
                flush=True,
            )
            self._stats["drain_recoveries"] += 1
            try:
                self._inflight.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s inflight failed during sync fallback: %r", LOG_PREFIX, exc)
                self._inflight = None
                self._inflight_obs_send_step = None
            else:
                self._pump_inflight()
            if len(self._queue) > 0:
                return
            # If still empty (e.g., the inflight merge was a stale drop),
            # fall through to a fresh synchronous request.
            print(
                f"{LOG_PREFIX} DRAIN-RECOVERY  step={self._step:4d} "
                f"inflight returned but queue still empty (likely stale); "
                f"firing fresh sync infer",
                flush=True,
            )

        # Otherwise: real cold start. Block on a fresh request with the
        # caller's obs and merge it synchronously.
        obs_send_step = self._step
        print(
            f"{LOG_PREFIX} cold-start  step={self._step:4d} "
            f"(queue empty, no inflight) — sync seed inference",
            flush=True,
        )
        obs_copy = self._shallow_copy_obs(obs)
        result = self._ws_policy.infer(obs_copy)
        self._calls_since_last_submit = 0
        self._stats["submits"] += 1
        self._stats["cold_starts"] += 1
        new_chunk = self._extract_actions(result)
        if new_chunk is None:
            return
        self._merge_response(obs_send_step, new_chunk)

    def _merge_response(
        self,
        obs_send_step: int,
        new_chunk: np.ndarray,
    ) -> None:
        if new_chunk.ndim != 2 or new_chunk.shape[1] != self._D:
            raise ValueError(
                f"{LOG_PREFIX} new_chunk shape {new_chunk.shape} != (H, {self._D})"
            )
        if new_chunk.shape[0] != self._H:
            # Server can in principle ship a different H, but our merge math
            # assumes the chunk covers [obs_send_step, obs_send_step + H).
            # Surface this loudly rather than silently truncating.
            raise ValueError(
                f"{LOG_PREFIX} new_chunk rows {new_chunk.shape[0]} != action_horizon {self._H}"
            )

        offset = self._step - obs_send_step
        if offset < 0:
            # Response for a future obs — impossible with single inflight.
            logger.warning(
                "%s ignoring response with negative offset %d (obs_send_step=%d, step=%d)",
                LOG_PREFIX, offset, obs_send_step, self._step,
            )
            return
        if offset >= self._H:
            self._stats["stale_drops"] += 1
            self._stats["last_offset"] = offset
            self._stats["last_blend_n"] = 0
            print(
                f"{LOG_PREFIX} stale-drop    step={self._step:4d} "
                f"obs_send={obs_send_step:4d} offset={offset:3d} >= H={self._H} "
                f"— entire chunk is in the past; dropping",
                flush=True,
            )
            return

        useful_new = new_chunk[offset:]
        M = useful_new.shape[0]

        with self._lock:
            L = len(self._queue)
            overlap = min(L, M)
            blend_n = min(self._merge_steps, overlap)

            # Region 1: cosine blend the first blend_n overlapping rows.
            for i in range(blend_n):
                w = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (blend_n + 1))
                old_row = self._queue[i]
                self._queue[i] = (
                    (1.0 - w) * old_row + w * useful_new[i]
                ).astype(np.float32, copy=False)

            # Region 2: hard-overwrite the rest of the overlap.
            for i in range(blend_n, overlap):
                self._queue[i] = useful_new[i].astype(np.float32, copy=True)

            # Region 3: append non-overlapping new rows.
            for i in range(overlap, M):
                self._queue.append(useful_new[i].astype(np.float32, copy=True))

            # If L > M (old tail extends past where new chunk covers) we keep
            # the old tail. With a constant H and sequential submits, this
            # branch should be unreachable.

        drained = (L == 0)
        self._stats["last_offset"] = offset
        self._stats["last_blend_n"] = blend_n
        self._stats["last_L"] = L
        self._stats["merges"] += 1
        if drained:
            self._stats["merges_just_in_time"] += 1
        print(
            f"{LOG_PREFIX} merge         step={self._step:4d} "
            f"obs_send={obs_send_step:4d} offset={offset:3d} "
            f"L={L:3d} M={M:3d} overlap={overlap:3d} "
            f"blend_n={blend_n:2d} (cosine over {blend_n} rows) "
            f"drained={'YES' if drained else 'no '} "
            f"hard_overwrite={overlap - blend_n:3d} append={max(0, M - overlap):3d}",
            flush=True,
        )

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _extract_actions(result: Any) -> Optional[np.ndarray]:
        if not isinstance(result, dict) or "actions" not in result:
            logger.warning(
                "%s server response missing 'actions' key; got keys=%r",
                LOG_PREFIX, list(result.keys()) if isinstance(result, dict) else type(result),
            )
            return None
        chunk = np.asarray(result["actions"], dtype=np.float32)
        return chunk

    @staticmethod
    def _shallow_copy_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Copy ndarray leaves so the inflight request owns its own buffers."""
        out: Dict[str, Any] = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                out[k] = v.copy()
            else:
                out[k] = v
        return out
