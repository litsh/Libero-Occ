## Requirements

Install upstream LIBERO first:

- LIBERO, for the simulator, BDDL loading, and task environments.
- Emu3 vision tokenizer and VQ model checkpoints required by VIM evaluation.
- FAST action tokenizer used by the action-token training pipeline.

Follow the upstream installation instructions for CUDA, MuJoCo, Python packages, and simulator rendering.

## Python Environment

```bash
conda create -n libero-occ python=3.10
conda activate libero-occ
```

Install a PyTorch build matching your CUDA version, then install this repository's dependencies:

```bash
pip install -r requirements.txt
```

See [univla_vim_code.md](univla_vim_code.md) for the VIM/UniVLA-specific dependency and checkpoint notes.

## Install LIBERO-Occ Assets

From this repository:

```bash
export LIBERO_ROOT=/path/to/LIBERO
bash scripts/setup/install_libero_occ_assets.sh
```

The installer copies:

```text
benchmark_assets/bddl_files/libero_*_occluded -> $LIBERO_ROOT/libero/libero/bddl_files/
benchmark_assets/init_files/libero_*_occluded -> $LIBERO_ROOT/libero/libero/init_files/
```

After installation, LIBERO should be able to resolve the four occluded task-suite names:

```text
libero_spatial_occluded
libero_goal_occluded
libero_object_occluded
libero_10_occluded
```
