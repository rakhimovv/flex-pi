# Vendored RoboTwin

This directory vendors code from the upstream RoboTwin repository:

- Upstream project: https://github.com/RoboTwin-Platform/RoboTwin
- Upstream commit: `bf44be51cf5717a5595ce59447f2cf5263d2aa95`
- Upstream license: MIT License

License compliance notes:

- The original upstream license is preserved in [`LICENSE`](./LICENSE).
- Files copied from RoboTwin remain subject to the MIT License in this directory.
- The only locally maintained policy implementation is `policy/flexpi_policy`, which
  is a symlink to `experiments/robotwin/flexpi_policy` in this repository.
- `experiments/robotwin/flexpi_policy` contains project-specific code authored for
  this repository and is not copied from the upstream `policy/*` subprojects.
- The unused upstream policy implementations (`ACT`, `DP`, `DP3`, `DexVLA`, `GO1`,
  `LLaVA-VLA`, `RDT`, `TinyVLA`, `openvla-oft`, `pi0`, `pi05`) have been removed for
  redistribution. Several of them carried their own additional license notices; if any
  code is later copied back from an upstream subdirectory with its own license, that
  license file and attribution must be preserved alongside it.

Local modifications:

- RoboTwin is vendored under `third_party/RoboTwin` for easier integration with this
  project. `configs/sim_robotwin.yaml` sets `robotwin_root: third_party/RoboTwin`,
  resolved relative to the repository root.
- `policy/flexpi_policy` is checked in as a *relative* symlink so a fresh clone works
  without manual setup. `experiments/robotwin/eval_robotwin_single.py` also recreates it
  at eval time if missing.
- The following files carry local changes relative to upstream:
  - `envs/_base_task.py`
  - `envs/utils/pkl2hdf5.py` (RGB/JPEG encoding fix)
  - `script/collect_data.py`
  - `script/_install.sh`
  - `description/utils/generate_episode_instructions.py`
  - `README.md`
  - `.gitignore`
- Additional files not present upstream: `collect_data_multi.sh`,
  `collect_data_replay.sh`, `MULTI_TASK_COLLECTION.md`, and the
  `description/_generate_*.txt` prompt templates.
- `task_config/` is checked in (unlike some downstream vendorings) because
  `experiments/robotwin/run_robotwin_manager.py` reads
  `task_config/_eval_step_limit.yml` to enumerate evaluation tasks.

Not included in this repository:

- `assets/` (~16 GB of meshes and embodiment descriptions) and `envs/curobo` are not
  redistributed here. Follow the official RoboTwin installation instructions to install
  the simulator environment and download the assets. See the RoboTwin section of the
  top-level [`README.md`](../../README.md).
