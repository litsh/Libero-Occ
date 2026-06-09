#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIVLA_ROOT="${UNIVLA_ROOT:-${REPO_ROOT}/univla}"

if [[ -n "${LIBERO_ROOT:-}" ]]; then
  export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
fi

VIM_CKPT="${VIM_CKPT:-}"
VISION_HUB="${VISION_HUB:-}"
VQ_HUB="${VQ_HUB:-}"
ACTION_TOKENIZER_PATH="${ACTION_TOKENIZER_PATH:-}"

for required in VIM_CKPT VISION_HUB VQ_HUB ACTION_TOKENIZER_PATH; do
  if [[ -z "${!required}" ]]; then
    echo "Set ${required} before running this script." >&2
    exit 1
  fi
done

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal_occluded}"
PERSPECTIVE_OBS_KEY="${PERSPECTIVE_OBS_KEY:-robot0_eye_in_hand_image}"
CACHE_ROOT="${CACHE_ROOT:-${UNIVLA_ROOT}/logs/libero_occ_eval/${TASK_SUITE_NAME}/$(basename "${VIM_CKPT}")}"
MASTER_PORT="${MASTER_PORT:-30002}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
CAMERA_RESOLUTION="${CAMERA_RESOLUTION:-200}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
NORMALIZER_PATH="${NORMALIZER_PATH:-}"
NORMALIZER_KEY="${NORMALIZER_KEY:-libero}"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"

NORMALIZER_ARGS=()
if [[ -n "${NORMALIZER_PATH}" ]]; then
  NORMALIZER_ARGS+=(--normalizer_path "${NORMALIZER_PATH}")
fi

cd "${UNIVLA_ROOT}/reference/RoboVLMs"
export PYTHONPATH="${UNIVLA_ROOT}:${UNIVLA_ROOT}/reference/RoboVLMs:${PYTHONPATH:-}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  eval/libero/evaluate_libero_emu_multi_gpu.py \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --num_steps_wait "${NUM_STEPS_WAIT}" \
  --emu_hub "${VIM_CKPT}" \
  --cache_root "${CACHE_ROOT}" \
  --vision_hub "${VISION_HUB}" \
  --vq_hub "${VQ_HUB}" \
  --with_perspective_gen \
  --cot_max_new_tokens "${COT_MAX_NEW_TOKENS:-1024}" \
  --no_gripper \
  --perspective_obs_key "${PERSPECTIVE_OBS_KEY}" \
  --camera_resolution "${CAMERA_RESOLUTION}" \
  --normalizer_key "${NORMALIZER_KEY}" \
  "${NORMALIZER_ARGS[@]}" \
  ${DEBUG:+--debug} \
  --perspective_eval
