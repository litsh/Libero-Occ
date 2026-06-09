# VIM Evaluation

The evaluation script launches the bundled UniVLA/RoboVLMs LIBERO evaluation code with perspective generation enabled.

## Example

```bash
export LIBERO_ROOT=/path/to/LIBERO
export VIM_CKPT=/path/to/vim/checkpoint
export VISION_HUB=/path/to/Emu3-VisionTokenizer
export VQ_HUB=/path/to/Emu3-Stage1
export ACTION_TOKENIZER_PATH=/path/to/fast

TASK_SUITE_NAME=libero_spatial_occluded bash scripts/eval/eval_vim_libero_occ.sh
TASK_SUITE_NAME=libero_goal_occluded bash scripts/eval/eval_vim_libero_occ.sh
TASK_SUITE_NAME=libero_object_occluded bash scripts/eval/eval_vim_libero_occ.sh
TASK_SUITE_NAME=libero_10_occluded bash scripts/eval/eval_vim_libero_occ.sh
```

## Useful Variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `TASK_SUITE_NAME` | `libero_goal_occluded` | LIBERO-Occ suite to evaluate |
| `GPUS_PER_NODE` | `8` | Number of GPUs for `torchrun` |
| `NUM_TRIALS_PER_TASK` | `1` | Trials per task instance |
| `CAMERA_RESOLUTION` | `200` | LIBERO camera resolution |
| `PERSPECTIVE_OBS_KEY` | `robot0_eye_in_hand_image` | Complementary view used for debug/reference |
| `CACHE_ROOT` | `logs/libero_occ_eval/...` | Output/cache directory |
