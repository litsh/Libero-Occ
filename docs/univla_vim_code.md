# Included VIM / UniVLA-Derived Code

This repository includes the UniVLA-derived source code needed for VIM training and LIBERO-Occ evaluation.

```text
univla/
├── train/
│   ├── train_moe.py
│   └── datasets.py
├── models/
│   ├── policy_head/
│   └── tokenizer/action_tokenizer.py
├── configs/
│   └── moe_fast_video.json
├── scripts/sft/
│   └── zero3_H200.json
└── reference/
    ├── Emu3/
    └── RoboVLMs/eval/libero/
```

## Environment Setup

Create a Python environment:

```bash
conda create -n libero-occ python=3.10
conda activate libero-occ
```

Install PyTorch for your CUDA version first. For example:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining Python dependencies:

```bash
pip install -r requirements.txt
```

`flash-attn` is useful for training throughput but can be environment-specific. Install it separately if your CUDA/PyTorch setup supports it:

```bash
pip install flash-attn==2.5.7 --no-build-isolation
```

## Required External Checkpoints

Set these paths before training or evaluation:

```bash
export WORLD_MODEL_CKPT=/path/to/WORLD_MODEL_POSTTRAIN
export ACTION_TOKENIZER_PATH=/path/to/fast
export VISION_HUB=/path/to/Emu3-VisionTokenizer
export VQ_HUB=/path/to/Emu3-Stage1
```

For evaluation, set:

```bash
export VIM_CKPT=/path/to/vim/checkpoint
```

For training, set:

```bash
export STAGE1_DATA_PATH=/path/to/stage1_multiview_meta.pkl
export STAGE2_DATA_PATH=/path/to/stage2_multiview_meta.pkl
```

## Running from the Bundled Code

The launch scripts default to the bundled `univla/` directory:

```bash
bash scripts/train/train_vim_stage1.sh
bash scripts/train/train_vim_stage2.sh
TASK_SUITE_NAME=libero_goal_occluded bash scripts/eval/eval_vim_libero_occ.sh
```

If you want to use a separate UniVLA checkout instead, override:

```bash
export UNIVLA_ROOT=/path/to/your/UniVLA
```

