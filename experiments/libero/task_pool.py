"""File-backed shared work queue for LIBERO eval workers.

Multiple eval workers (across GPUs and within a single GPU) pull groups
from one shared `pool_file` so the slowest worker no longer determines
total wall time. Locking via fcntl.flock on a sidecar `.lock` file —
no daemon, no IPC.

Pool format: JSONL, one group per line. Each group is a dict:

    {"bddl_file": "...", "tasks": [["libero_spatial", 12], ["libero_spatial", 47]]}

A group MUST hold tasks that can share an OffScreenRenderEnv (i.e. all
share the same `bddl_file` value passed to MuJoCo). A group with one task
is fine and is the default when env reuse is not exploited.

Pop semantics: take the FIRST line, atomically rewrite the pool file
without it. Workers see consistent state under the lock. If a worker
crashes after popping, that group is dropped — caller is responsible for
detecting missing result files post-hoc and re-queueing.
"""
import fcntl
import json
import os
from typing import Optional


def pop_group(pool_file: str) -> Optional[dict]:
    """Pop the next group from `pool_file`. Returns None if empty.

    Atomic across processes via flock on `<pool_file>.lock`.
    """
    lock_path = pool_file + ".lock"
    # Open lock file in append mode so creation is race-free.
    with open(lock_path, "a") as fl:
        fcntl.flock(fl.fileno(), fcntl.LOCK_EX)
        try:
            try:
                with open(pool_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                return None

            # Skip blank/comment lines while preserving original index.
            head_idx = None
            for i, line in enumerate(lines):
                s = line.strip()
                if s and not s.startswith("#"):
                    head_idx = i
                    break
            if head_idx is None:
                return None

            head = lines[head_idx]
            remaining = lines[:head_idx] + lines[head_idx + 1 :]

            tmp_path = pool_file + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.writelines(remaining)
            os.replace(tmp_path, pool_file)

            return json.loads(head)
        finally:
            fcntl.flock(fl.fileno(), fcntl.LOCK_UN)


def remaining(pool_file: str) -> int:
    """Approximate number of groups left in the pool. Best-effort, no lock."""
    try:
        with open(pool_file, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip() and not line.startswith("#"))
    except FileNotFoundError:
        return 0
