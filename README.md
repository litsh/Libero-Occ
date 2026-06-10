# LIBERO-Occ

LIBERO-Occ is an occlusion-oriented extension of LIBERO for evaluating Vision-Language-Action models under scene-induced occlusion. This repository releases the benchmark assets and the scripts used for Viewpoint Imagination (VIM) training and evaluation.


## Contents

```text
LIBERO-Occ/
├── benchmark_assets/
│   ├── bddl_files/          # Occluded LIBERO BDDL task suites
│   └── init_files/          # Occluded LIBERO initial-state files
├── scripts/
│   ├── setup/               # Install/check released assets
│   ├── train/               # VIM Stage-1 and Stage-2 launch scripts
│   └── eval/                # VIM LIBERO-Occ evaluation launch script
├── univla/                  # UniVLA-derived VIM training/evaluation code
├── configs/                 # Reference training/evaluation settings
└── docs/                    # Installation, benchmark, training, and evaluation notes
```

## Benchmark Assets

The release contains four LIBERO-Occ suites:

- `libero_spatial_occluded`
- `libero_goal_occluded`
- `libero_object_occluded`
- `libero_10_occluded`

Each suite has matching files under `benchmark_assets/bddl_files/` and `benchmark_assets/init_files/`.

## Installation

Install upstream LIBERO first, then install this repository's Python dependencies and benchmark assets:

```bash
git clone https://github.com/TODO/LIBERO-Occ.git
cd LIBERO-Occ

conda create -n libero-occ python=3.10
conda activate libero-occ
# Install the CUDA-matched PyTorch build first, then:
pip install -r requirements.txt

export LIBERO_ROOT=/path/to/LIBERO
bash scripts/setup/install_libero_occ_assets.sh
```

See [docs/installation.md](docs/installation.md) and [docs/univla_vim_code.md](docs/univla_vim_code.md) for details.

## Train VIM

Run the scripts from this repository. By default, they use the bundled `univla/` code. You can also set `UNIVLA_ROOT=/path/to/UniVLA` to run against another checkout.

```bash
export STAGE1_DATA_PATH=/path/to/stage1_multiview_meta.pkl
export STAGE2_DATA_PATH=/path/to/stage2_multiview_meta.pkl
export WORLD_MODEL_CKPT=/path/to/WORLD_MODEL_POSTTRAIN
export ACTION_TOKENIZER_PATH=/path/to/fast

bash scripts/train/train_vim_stage1.sh

export STAGE1_CKPT=/path/to/UniVLA/logs/vim_stage1_gripper/checkpoint-4000
bash scripts/train/train_vim_stage2.sh
```

See [docs/training.md](docs/training.md).

## Evaluate VIM on LIBERO-Occ

```bash
export LIBERO_ROOT=/path/to/LIBERO
export VIM_CKPT=/path/to/vim/checkpoint
export VISION_HUB=/path/to/Emu3-VisionTokenizer
export VQ_HUB=/path/to/Emu3-Stage1
export ACTION_TOKENIZER_PATH=/path/to/fast

TASK_SUITE_NAME=libero_goal_occluded bash scripts/eval/eval_vim_libero_occ.sh
```

See [docs/evaluation.md](docs/evaluation.md).

## License

This repository is released under the MIT License. LIBERO-Occ is built on top of LIBERO; please also follow the license and citation requirements of upstream LIBERO and UniVLA.

## Citation

```bibtex
@misc{li2026liberooccevaluatingimprovingvisionlanguageaction,
      title={LIBERO-Occ: Evaluating and Improving Vision-Language-Action Models under Scene-Induced Occlusion via Viewpoint Imagination}, 
      author={Taishan Li and Jiwen Zhang and Siyuan Wang and Xuanjing Huang and Zhongyu Wei},
      year={2026},
      eprint={2606.10862},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.10862}, 
}
```
