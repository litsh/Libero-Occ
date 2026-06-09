#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIVLA_ROOT="${UNIVLA_ROOT:-${REPO_ROOT}/univla}"

WORLD_SIZE="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-23456}"
NGPUS="${NGPUS:-8}"

STAGE1_CKPT="${STAGE1_CKPT:-}"
MODEL_CONFIG_PATH="${MODEL_CONFIG_PATH:-${UNIVLA_ROOT}/configs/moe_fast_video.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${UNIVLA_ROOT}/scripts/sft/zero3_H200.json}"
STAGE2_DATA_PATH="${STAGE2_DATA_PATH:-}"
ACTION_TOKENIZER_PATH="${ACTION_TOKENIZER_PATH:-}"

PERSPECTIVE_IMAGE_KEY="${PERSPECTIVE_IMAGE_KEY:-gripper_image}"
PERSPECTIVE_VIEW_NAME="${PERSPECTIVE_VIEW_NAME:-gripper}"
PERSPECTIVE_USE_VANILLA_PREFIX="${PERSPECTIVE_USE_VANILLA_PREFIX:-False}"
EXP_NAME="${EXP_NAME:-vim_stage2_${PERSPECTIVE_VIEW_NAME}_weighted}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-192}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-3}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-$((GLOBAL_BATCH_SIZE / NGPUS / PER_GPU_BATCH_SIZE))}"
MAX_STEPS="${MAX_STEPS:-6000}"
LEARNING_RATE="${LEARNING_RATE:-4e-5}"

for required in STAGE1_CKPT STAGE2_DATA_PATH ACTION_TOKENIZER_PATH; do
  if [[ -z "${!required}" ]]; then
    echo "Set ${required} before running this script." >&2
    exit 1
  fi
done

cd "${UNIVLA_ROOT}"
export PYTHONPATH="${UNIVLA_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_WEIGHTS_ONLY_LOAD:-0}"

torchrun \
  --nproc_per_node="${NGPUS}" \
  --nnodes="${WORLD_SIZE}" \
  --node_rank="${RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train/train_moe.py \
  --model_name_or_path "${STAGE1_CKPT}" \
  --model_config_path "${MODEL_CONFIG_PATH}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --output_dir "logs/${EXP_NAME}" \
  --learning_rate "${LEARNING_RATE}" \
  --null_prompt_prob 0.15 \
  --weight_decay 0.1 \
  --min_learning_rate 5e-6 \
  --max_grad_norm 5.0 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-6 \
  --bf16 True \
  --tf32 True \
  --data_path "${STAGE2_DATA_PATH}" \
  --max_steps "${MAX_STEPS}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-12}" \
  --lr_scheduler_type "cosine_with_min_lr" \
  --warmup_steps "${WARMUP_STEPS:-50}" \
  --frames 1 \
  --action_frames "${ACTION_FRAMES:-10}" \
  --max_position_embeddings "${MAX_POSITION_EMBEDDINGS:-3200}" \
  --seed "${SEED:-42}" \
  --logging_steps "${LOGGING_STEPS:-20}" \
  --gradient_checkpointing True \
  --apply_loss_on_only_action False \
  --actions True \
  --actions_format "fast" \
  --action_tokenizer_path "${ACTION_TOKENIZER_PATH}" \
  --per_device_train_batch_size "${PER_GPU_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS:-2000}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --eval_strategy no \
  --use_gripper False \
  --with_perspective True \
  --perspective_image_key "${PERSPECTIVE_IMAGE_KEY}" \
  --perspective_use_vanilla_prefix "${PERSPECTIVE_USE_VANILLA_PREFIX}" \
  --group_loss_mode four_group \
  --use_group_loss_weighting True \
  --visual_content_loss_weight "${VISUAL_CONTENT_LOSS_WEIGHT:-0.5}" \
  --visual_special_loss_weight "${VISUAL_SPECIAL_LOSS_WEIGHT:-0.5}" \
  --action_content_loss_weight "${ACTION_CONTENT_LOSS_WEIGHT:-1.0}" \
  --action_special_loss_weight "${ACTION_SPECIAL_LOSS_WEIGHT:-0.2}"
