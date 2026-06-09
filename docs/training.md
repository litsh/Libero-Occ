# VIM Training

The training scripts use the bundled UniVLA-derived code under `univla/` by default. Run them from this repository. Set `UNIVLA_ROOT=/path/to/UniVLA` only if you want to use a separate checkout.

## Stage 1: Viewpoint Imagination

Stage 1 starts from a UniVLA world-model checkpoint and trains the model to generate the complementary viewpoint.

```bash
export WORLD_MODEL_CKPT=/path/to/WORLD_MODEL_POSTTRAIN
export STAGE1_DATA_PATH=/path/to/stage1_multiview_meta.pkl
export ACTION_TOKENIZER_PATH=/path/to/fast

bash scripts/train/train_vim_stage1.sh
```

Important defaults:

- `PERSPECTIVE_IMAGE_KEY=gripper_image`
- `PERSPECTIVE_VIEW_NAME=gripper`
- `MAX_STEPS=4000`
- `LEARNING_RATE=8e-5`
- visual loss weight is enabled and action loss weight is disabled.

## Stage 2: Joint Viewpoint Imagination and Action Prediction

Stage 2 starts from the Stage-1 checkpoint and jointly trains the view-generation and action objectives.

```bash
export STAGE1_CKPT=/path/to/UniVLA/logs/vim_stage1_gripper/checkpoint-4000
export STAGE2_DATA_PATH=/path/to/stage2_multiview_meta.pkl

bash scripts/train/train_vim_stage2.sh
```

Important defaults:

- `MAX_STEPS=6000`
- `LEARNING_RATE=4e-5`
- `VISUAL_CONTENT_LOSS_WEIGHT=0.5`
- `ACTION_CONTENT_LOSS_WEIGHT=1.0`
- `ACTION_SPECIAL_LOSS_WEIGHT=0.2`

Adjust paths and batch settings through environment variables rather than editing the scripts.
