"""
Evaluate UniVLA / RoboVLMs-CoT via LIBERO with multi-GPU episode-level sharding.
"""
import os
import faulthandler
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
if os.environ.get("MUJOCO_GL", "").lower() == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
faulthandler.enable()
import argparse
import json
import logging
import re
import sys
import time
import traceback
from pathlib import Path
from PIL import Image
import tqdm
import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from datetime import timedelta
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from model_wrapper_emu import EmuVLAModel
from libero_utils import (
    get_libero_camera_image,
    get_episode_length,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_gif,
)

LIBERO_ROOT = os.environ.get("LIBERO_ROOT")
if LIBERO_ROOT:
    sys.path.insert(0, LIBERO_ROOT)
from libero.libero import benchmark

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
)
logger = logging.getLogger(__name__)


def _to_uint8_image(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    return image


def _save_png(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(_to_uint8_image(image)).save(path)


class PerspectiveDatasetWriter:
    def __init__(self, dataset_root, rank):
        self.dataset_root = dataset_root
        self.samples_root = os.path.join(dataset_root, "samples")
        os.makedirs(self.samples_root, exist_ok=True)
        self.manifest_path = os.path.join(dataset_root, f"samples_rank{rank}.jsonl")
        self.manifest_file = open(self.manifest_path, "w")

    def save_sample(
        self,
        task_id,
        task_description,
        episode_idx,
        global_episode_idx,
        decision_step_idx,
        env_step_idx,
        agentview_image,
        gt_perspective_image,
        wrist_image=None,
    ):
        sample_dir = os.path.join(
            self.samples_root,
            f"task{task_id:02d}",
            f"episode{episode_idx + 1:03d}",
            f"step{decision_step_idx:04d}",
        )
        agentview_path = os.path.join(sample_dir, "input_agentview.png")
        gt_perspective_path = os.path.join(sample_dir, "gt_perspective.png")
        _save_png(agentview_image, agentview_path)
        _save_png(gt_perspective_image, gt_perspective_path)

        wrist_rel_path = None
        if wrist_image is not None:
            wrist_path = os.path.join(sample_dir, "input_wrist.png")
            _save_png(wrist_image, wrist_path)
            wrist_rel_path = os.path.relpath(wrist_path, self.dataset_root)

        return {
            "task_id": task_id,
            "task_description": task_description,
            "episode_idx": episode_idx,
            "global_episode_idx": global_episode_idx,
            "decision_step_idx": decision_step_idx,
            "env_step_idx": env_step_idx,
            "input_agentview_path": os.path.relpath(agentview_path, self.dataset_root),
            "gt_perspective_path": os.path.relpath(gt_perspective_path, self.dataset_root),
            "input_wrist_path": wrist_rel_path,
        }

    def finalize_episode(self, records, episode_success, episode_error=None):
        for record in records:
            item = dict(record)
            item["episode_success"] = bool(episode_success)
            item["episode_error"] = episode_error
            self.manifest_file.write(json.dumps(item) + "\n")
        self.manifest_file.flush()

    def close(self):
        self.manifest_file.close()


def _decode_subgoal_images_from_text(model, thought_text):
    if not thought_text:
        return []

    tokenizer = model.processor.tokenizer
    boi = re.escape(tokenizer.boi_token)
    eoi = re.escape(tokenizer.eoi_token)
    img_token = re.escape(tokenizer.img_token)
    visual_pattern = re.compile(model.processor.visual_template[1])

    pattern = re.compile(
        rf"{boi}(\d+)\*(\d+)\*(\d+){img_token}(.*?){eoi}",
        re.DOTALL,
    )
    images = []
    for match in pattern.finditer(thought_text):
        frames = int(match.group(1))
        height = int(match.group(2))
        width = int(match.group(3))
        content = match.group(4)
        token_ids = [int(x) for x in visual_pattern.findall(content)]
        if height <= 0 or width <= 0:
            continue
        expected = height * width * max(frames, 1)
        if len(token_ids) < expected:
            continue
        token_ids = token_ids[:expected]
        frame_tokens = [
            token_ids[i * height * width:(i + 1) * height * width]
            for i in range(max(frames, 1))
        ]
        for tokens in frame_tokens:
            token_tensor = torch.tensor(
                tokens,
                dtype=torch.long,
                device=model.processor.vision_tokenizer.device,
            )
            token_tensor = token_tensor.reshape(height, width)
            decoded = model.processor.vision_decode(token_tensor[None]).float()
            img = model.processor.image_processor.postprocess(decoded)["pixel_values"][0]
            images.append(img)
    return images


def save_subgoal_images(model, thought_text, out_dir, prefix):
    try:
        if not thought_text:
            return 0, []

        images = _decode_subgoal_images_from_text(model, thought_text)
        if not images:
            os.makedirs(out_dir, exist_ok=True)
            text_path = os.path.join(out_dir, f"{prefix}.txt")
            with open(text_path, "w") as f:
                f.write(thought_text)
            return 0, []

        os.makedirs(out_dir, exist_ok=True)
        gif_frames = []
        for image in images:
            img = np.asarray(image)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = img * 255.0
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[-1] == 3:
                gif_frames.append(img)
        return 0, gif_frames
    except Exception as exc:
        traceback.print_exc()
        os.makedirs(out_dir, exist_ok=True)
        text_path = os.path.join(out_dir, f"{prefix}.txt")
        with open(text_path, "w") as f:
            f.write(thought_text)
        return 0, []


def save_generated_perspective_images(generation_info, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    thought_text = generation_info.get("thought_text", "")
    images = generation_info.get("perspective_images", []) or []

    if thought_text:
        text_path = os.path.join(out_dir, f"{prefix}.txt")
        with open(text_path, "w") as f:
            f.write(thought_text)

    frames = []
    for image_idx, image in enumerate(images):
        image = _to_uint8_image(image)
        if len(images) == 1:
            image_path = os.path.join(out_dir, f"{prefix}.png")
        else:
            image_path = os.path.join(out_dir, f"{prefix}_frame{image_idx:02d}.png")
        Image.fromarray(image).save(image_path)
        frames.append(image)
    return frames


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def resolve_physical_cuda_device(visible_index: int) -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible:
        return visible_index
    if cuda_visible.isdigit():
        if visible_index != 0:
            raise RuntimeError(
                f"Requested visible device index {visible_index} but "
                f"CUDA_VISIBLE_DEVICES={cuda_visible} exposes only one device."
            )
        return int(cuda_visible)
    device_ids = [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
    if visible_index < 0 or visible_index >= len(device_ids):
        raise RuntimeError(
            f"Requested visible device index {visible_index} but "
            f"CUDA_VISIBLE_DEVICES={cuda_visible} provides {len(device_ids)} device(s)."
        )
    return device_ids[visible_index]


def setup_distributed():
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240) # 4 hours
        )
    local_rank, rank, world_size = world_info_from_env()
    # Bind EGL to the physical device that matches this process's visible index.
    render_gpu_device_id = resolve_physical_cuda_device(local_rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_device_id)
    
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size, render_gpu_device_id


def prepare_observation(obs):
    img = get_libero_image(obs)
    observation = {
        "full_image": img,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }
    if "robot0_eye_in_hand_image" in obs:
        observation["wrist_image"] = get_libero_wrist_image(obs)
    return observation, img


def obs_key_to_camera_name(obs_key):
    if obs_key == "gripper_image":
        return "robot0_eye_in_hand"
    if not obs_key.endswith("_image"):
        raise ValueError(f"Unexpected observation key format: {obs_key}")
    return obs_key[: -len("_image")]


def episode_is_assigned(task_id, episode_idx, num_trials_per_task, rank, world_size):
    global_episode_idx = task_id * num_trials_per_task + episode_idx
    return global_episode_idx % world_size == rank, global_episode_idx


def evaluate(
    model,
    task_suite_name,
    local_log_dir,
    rank,
    world_size,
    render_gpu_device_id,
    num_trials_per_task,
    num_steps_wait,
    debug=False,
    perspective_eval=False,
    perspective_obs_key="robot0_eye_in_hand_image",
    camera_resolution=256,
    dataset_writer=None,
):
    os.makedirs(local_log_dir, exist_ok=True)
    log_path = os.path.join(local_log_dir, f"eval_rank{rank}.txt")
    log_file = open(log_path, "w")
    logger.info("Logging to %s", log_path)

    results_path = os.path.join(local_log_dir, f"episodes_rank{rank}.jsonl")
    results_file = open(results_path, "w")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    EP_LEN = get_episode_length(task_suite_name)

    logger.info("Task suite: %s", task_suite_name)
    log_file.write(f"Task suite: {task_suite_name}\n")
    log_file.write(f"World size: {world_size}, rank: {rank}\n")
    log_file.write(f"Num trials per task: {num_trials_per_task}\n")
    log_file.flush()

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"Task Pregress", total=num_tasks_in_suite, disable=(rank != 0 )):
        assigned_episode_indices = []
        for episode_idx in range(num_trials_per_task):
            assigned, _ = episode_is_assigned(
                task_id, episode_idx, num_trials_per_task, rank, world_size
            )
            if assigned:
                assigned_episode_indices.append(episode_idx)

        if not assigned_episode_indices:
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        if len(initial_states) < num_trials_per_task:
            raise ValueError(
                f"Task {task_id} has only {len(initial_states)} initial states, "
                f"but num_trials_per_task={num_trials_per_task}"
            )

        camera_names = ["agentview"]
        if getattr(model, "use_gripper", False):
            camera_names.append("robot0_eye_in_hand")
        if perspective_eval:
            target_camera = obs_key_to_camera_name(perspective_obs_key)
            if target_camera not in camera_names:
                camera_names.append(target_camera)

        env, task_description = get_libero_env(
            task,
            resolution=camera_resolution,
            render_gpu_device_id=render_gpu_device_id,
            camera_names=camera_names,
        )
        task_episodes = 0
        task_successes = 0

        logger.info("Task %s: %s", task_id, task_description)
        log_file.write(f"\nTask {task_id}: {task_description}\n")
        log_file.write(
            f"Assigned episodes: {assigned_episode_indices}\n"
        )
        log_file.flush()

        for episode_idx in tqdm.tqdm(assigned_episode_indices, desc=f"Episode Pregress", total=len(assigned_episode_indices), disable=(rank != 0 )):
            env.reset()
            model.reset()

            obs = env.set_init_state(initial_states[episode_idx])
            _, global_episode_idx = episode_is_assigned(
                task_id, episode_idx, num_trials_per_task, rank, world_size
            )
            t = 0
            replay_images = []
            subgoal_gif_frames = []
            generated_perspective_frames = []
            reference_images = []
            episode_dataset_records = []
            decision_step_idx = 0
            done = False
            error = None

            if model.use_cot:
                thought = [""]

            logger.info("Starting episode %s (task %s)", episode_idx + 1, task_id)
            log_file.write(
                f"Starting episode {episode_idx + 1} (task {task_id})\n"
            )
            log_file.flush()

            action_counter = 0
            while t < EP_LEN + num_steps_wait:
                try:
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue
                    # if rank == 0:
                    #     logger.info("Episode %s (Task %s): Step %s", episode_idx + 1, task_id, t)
                    
                    observation, img = prepare_observation(obs)
                    if debug:
                        replay_images.append(img)

                    if action_counter == 0:
                        target_perspective_image = None
                        if perspective_eval or dataset_writer is not None:
                            target_perspective_image = get_libero_camera_image(obs, perspective_obs_key)
                        if debug and (model.use_cot or model.use_perspective_gen or model.use_multiview_gen):
                            if perspective_eval:
                                reference_images.append(target_perspective_image)
                            else:
                                reference_images.append(get_libero_image(obs))
                        if dataset_writer is not None:
                            episode_dataset_records.append(
                                dataset_writer.save_sample(
                                    task_id=task_id,
                                    task_description=task_description,
                                    episode_idx=episode_idx,
                                    global_episode_idx=global_episode_idx,
                                    decision_step_idx=decision_step_idx,
                                    env_step_idx=t,
                                    agentview_image=img,
                                    gt_perspective_image=target_perspective_image,
                                    wrist_image=observation.get("wrist_image"),
                                )
                            )
                            decision_step_idx += 1
                        step_output = model.step(observation, task_description)
                        if isinstance(step_output, tuple):
                            action = step_output[0]
                            if len(step_output) > 1:
                                thought = step_output[1]
                        else:
                            action = step_output
                        if model.use_cot:
                            if debug:
                                pred_dir = os.path.join(
                                    local_log_dir,
                                    "pred_perspective_images" if perspective_eval else "pred_images",
                                    f"task{task_id}/episode{episode_idx + 1}",
                                )
                                _, gif_frames = save_subgoal_images(
                                    model,
                                    thought[0],
                                    pred_dir,
                                    prefix=f"step{t}",
                                )
                                if gif_frames:
                                    subgoal_gif_frames.extend(gif_frames)
                        elif debug and (model.use_perspective_gen or model.use_multiview_gen):
                            generation_info = getattr(model, "last_generation_info", {}) or {}
                            pred_dir = os.path.join(
                                local_log_dir,
                                "pred_perspective_images" if perspective_eval else "pred_images",
                                f"task{task_id}/episode{episode_idx + 1}",
                            )
                            gif_frames = save_generated_perspective_images(
                                generation_info,
                                pred_dir,
                                prefix=f"step{t}",
                            )
                            if gif_frames:
                                generated_perspective_frames.extend(gif_frames)
                        action_counter = action.shape[0]

                    step_action = action[-action_counter]
                    obs, reward, done, info = env.step(step_action.tolist())
                    action_counter -= 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as exc:
                    error = str(exc)
                    logger.exception("Episode error")
                    log_file.write(f"Caught exception: {error}\n")
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            if debug and replay_images:
                gif_dir = os.path.join(local_log_dir, "videos")
                os.makedirs(gif_dir, exist_ok=True)
                gif_path = os.path.join(
                    gif_dir, f"task{task_id}_episode{episode_idx + 1}_{done}.gif"
                )
                save_rollout_gif(replay_images, gif_path, fps=15)
            if debug and subgoal_gif_frames:
                subgoal_gif_dir = os.path.join(
                    local_log_dir,
                    "pred_perspective_images" if perspective_eval else "pred_images",
                )
                os.makedirs(subgoal_gif_dir, exist_ok=True)
                subgoal_gif_path = os.path.join(
                    subgoal_gif_dir, f"task{task_id}_episode{episode_idx + 1}_{done}.gif"
                )
                save_rollout_gif(subgoal_gif_frames, subgoal_gif_path, fps=15)
            if debug and generated_perspective_frames:
                generated_gif_dir = os.path.join(
                    local_log_dir,
                    "pred_perspective_images" if perspective_eval else "pred_images",
                )
                os.makedirs(generated_gif_dir, exist_ok=True)
                generated_gif_path = os.path.join(
                    generated_gif_dir, f"task{task_id}_episode{episode_idx + 1}_{done}.gif"
                )
                save_rollout_gif(generated_perspective_frames, generated_gif_path, fps=15)
            if debug and (model.use_cot or model.use_perspective_gen or model.use_multiview_gen) and reference_images:
                real_state_dir = os.path.join(
                    local_log_dir,
                    "real_states_perspective" if perspective_eval else "real_states",
                )
                os.makedirs(real_state_dir, exist_ok=True)
                real_state_path = os.path.join(
                    real_state_dir, f"task{task_id}_episode{episode_idx + 1}_{done}.gif"
                )
                save_rollout_gif(reference_images, real_state_path, fps=15)
            if dataset_writer is not None:
                dataset_writer.finalize_episode(
                    episode_dataset_records,
                    episode_success=done,
                    episode_error=error,
                )

            result = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "global_episode_idx": global_episode_idx,
                "success": bool(done),
                "steps": t,
                "error": error,
            }
            results_file.write(json.dumps(result) + "\n")
            results_file.flush()

            logger.info("Success: %s", done)
            logger.info("# episodes completed so far: %s", total_episodes)
            if total_episodes > 0:
                logger.info(
                    "# successes: %s (%.1f%%)",
                    total_successes,
                    total_successes / total_episodes * 100,
                )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            if total_episodes > 0:
                log_file.write(
                    f"# successes: {total_successes} "
                    f"({total_successes / total_episodes * 100:.1f}%)\n"
                )
            log_file.flush()

        if task_episodes > 0:
            logger.info(
                "Task %s success rate: %.3f",
                task_id,
                float(task_successes) / float(task_episodes),
            )
            log_file.write(
                f"Task {task_id} success rate: "
                f"{float(task_successes) / float(task_episodes):.3f}\n"
            )
            log_file.flush()
        env.close()
        
    log_file.close()
    results_file.close()

    return total_episodes, total_successes


def parse_args():
    seed_everything(0, workers=True)  # type: ignore
    parser = argparse.ArgumentParser(
        description="Evaluate UniVLA on LIBERO with multi-GPU episode-level sharding."
    )
    parser.add_argument("--debug", action="store_true", help="Save rollout GIFs.")
    parser.add_argument(
        "--perspective_eval",
        action="store_true",
        help="Save predicted images and real states from a selected perspective view.",
    )
    parser.add_argument(
        "--dump_perspective_eval_dataset",
        action="store_true",
        help="Save input agentview and GT perspective images at each model query for offline image-generation evaluation.",
    )
    parser.add_argument(
        "--dataset_dirname",
        type=str,
        default="perspective_eval_dataset",
        help="Subdirectory name under the run log dir for dumped offline perspective-eval samples.",
    )
    parser.add_argument(
        "--perspective_obs_key",
        type=str,
        default="robot0_eye_in_hand_image",
        help=(
            "Observation key used as the real target view during perspective evaluation, "
            "e.g. robot0_eye_in_hand_image, birdview_image, sideview_image."
        ),
    )
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--is_pt_config", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, nargs="+", default="")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_idx", type=int, default=-1)
    parser.add_argument("--emu_hub", type=str, default="")
    parser.add_argument(
        "--vq_hub",
        type=str,
        default=os.environ.get("VQ_HUB", ""),
    )
    parser.add_argument(
        "--vision_hub",
        type=str,
        default=os.environ.get("VISION_HUB", ""),
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
            "libero_spatial_occluded",
            "libero_goal_occluded",
            "libero_10_occluded",
            "libero_object_occluded",
            "libero_spatial_occluded_auto",
            "libero_goal_occluded_auto",
            "libero_10_occluded_auto",
            "libero_object_occluded_auto",
        ],
        required=True,
    )
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--no_nccl", action="store_true")
    parser.add_argument("--no_action_ensemble", action="store_true")
    parser.add_argument(
        "--cache_root",
        type=str,
        default=os.environ.get("CACHE_ROOT", "logs/libero"),
        help="Root directory to store cache/logs.",
    )
    parser.add_argument(
        "--with_cot",
        action="store_true",
        help="Enable CoT-style evaluation (subgoal reasoning).",
    )
    parser.add_argument(
        "--with_perspective_gen",
        action="store_true",
        help="Generate perspective-view image tokens autoregressively before generating actions.",
    )
    parser.add_argument(
        "--with_multiview_gen",
        action="store_true",
        help="Generate a selected target-view image using the multiview training prompt before generating actions.",
    )
    parser.add_argument(
        "--no_gripper",
        action="store_true",
        help="Not to use gripper image"
    )
    parser.add_argument(
        "--cot_max_new_tokens",
        type=int,
        default=256,
        help="Max tokens when generating CoT reasoning and goal image.",
    )
    parser.add_argument(
        "--num_trials_per_task", type=int, default=50, help="Episodes per task."
    )
    parser.add_argument(
        "--num_steps_wait", type=int, default=10, help="Initial wait steps."
    )
    parser.add_argument(
        "--camera_resolution",
        type=int,
        default=256,
        help="LIBERO offscreen render resolution per camera.",
    )
    parser.add_argument(
        "--normalizer_path",
        type=str,
        default=None,
        help="Path to a LIBERO normalizer directory or directly to norm_stats.json.",
    )
    parser.add_argument(
        "--normalizer_key",
        type=str,
        default="libero",
        help="Dataset key inside the normalizer stats file.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run id to share across ranks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.with_cot and args.with_perspective_gen:
        parser.error("--with_cot and --with_perspective_gen are mutually exclusive.")
    if args.with_cot and args.with_multiview_gen:
        parser.error("--with_cot and --with_multiview_gen are mutually exclusive.")
    if args.with_perspective_gen and args.with_multiview_gen:
        parser.error("--with_perspective_gen and --with_multiview_gen are mutually exclusive.")
    if args.dump_perspective_eval_dataset and not args.perspective_eval:
        parser.error("--dump_perspective_eval_dataset requires --perspective_eval.")
    return args


def get_run_id(args):
    if args.run_id:
        return args.run_id
    run_id = f"{args.task_suite_name}-{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    if dist.is_available() and dist.is_initialized():
        obj = [run_id if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]
    return run_id


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)  # type: ignore

    local_rank, rank, world_size = world_info_from_env()

    if not args.no_nccl and world_size > 1:
        local_rank, rank, world_size, render_gpu_device_id = setup_distributed()
    else:
        torch.cuda.set_device(args.device_id)
        render_gpu_device_id = resolve_physical_cuda_device(args.device_id)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_device_id)

    cache_root = args.cache_root
    os.makedirs(cache_root, exist_ok=True)

    run_id = get_run_id(args)
    eval_log_dir = os.path.join(cache_root, run_id)
    os.makedirs(eval_log_dir, exist_ok=True)

    if rank == 0:
        meta_path = os.path.join(eval_log_dir, "meta_info.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "task_suite_name": args.task_suite_name,
                    "num_trials_per_task": args.num_trials_per_task,
                    "world_size": world_size,
                    "results_pattern": "episodes_rank{rank}.jsonl",
                    "log_pattern": "eval_rank{rank}.txt",
                    "perspective_eval": args.perspective_eval,
                    "dump_perspective_eval_dataset": args.dump_perspective_eval_dataset,
                    "with_perspective_gen": args.with_perspective_gen,
                    "with_multiview_gen": args.with_multiview_gen,
                    "perspective_obs_key": args.perspective_obs_key,
                },
                f,
                indent=2,
            )

    dataset_writer = None
    if args.dump_perspective_eval_dataset:
        dataset_root = os.path.join(eval_log_dir, args.dataset_dirname)
        os.makedirs(dataset_root, exist_ok=True)
        if rank == 0:
            dataset_meta_path = os.path.join(dataset_root, "meta_info.json")
            with open(dataset_meta_path, "w") as f:
                json.dump(
                    {
                        "run_id": run_id,
                        "task_suite_name": args.task_suite_name,
                        "perspective_obs_key": args.perspective_obs_key,
                        "camera_resolution": args.camera_resolution,
                        "num_trials_per_task": args.num_trials_per_task,
                        "manifest_pattern": "samples_rank{rank}.jsonl",
                    },
                    f,
                    indent=2,
                )
        dataset_writer = PerspectiveDatasetWriter(dataset_root=dataset_root, rank=rank)

    model = EmuVLAModel(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vision_hub,
        device=torch.device("cuda"),
        use_cot=args.with_cot,
        cot_max_new_tokens=args.cot_max_new_tokens,
        use_perspective_gen=args.with_perspective_gen,
        use_multiview_gen=args.with_multiview_gen,
        perspective_obs_key=args.perspective_obs_key,
        use_gripper=not args.no_gripper,
        normalizer_path=args.normalizer_path,
        normalizer_key=args.normalizer_key,
    )

    try:
        total_episodes, total_successes = evaluate(
            model=model,
            task_suite_name=args.task_suite_name,
            local_log_dir=eval_log_dir,
            rank=rank,
            world_size=world_size,
            render_gpu_device_id=render_gpu_device_id,
            num_trials_per_task=args.num_trials_per_task,
            num_steps_wait=args.num_steps_wait,
            debug=args.debug,
            perspective_eval=args.perspective_eval,
            perspective_obs_key=args.perspective_obs_key,
            camera_resolution=args.camera_resolution,
            dataset_writer=dataset_writer,
        )
    finally:
        if dataset_writer is not None:
            dataset_writer.close()

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor([total_episodes, total_successes], device="cuda")
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            global_episodes = int(totals[0].item())
            global_successes = int(totals[1].item())
            if global_episodes > 0:
                global_sr = global_successes / global_episodes * 100
                logger.info(
                    "Global success rate: %s/%s (%.1f%%)",
                    global_successes,
                    global_episodes,
                    global_sr,
                )
                with open(os.path.join(eval_log_dir, "global_success_rate.txt"), "w") as f:
                    f.write(f"Global success rate: {global_successes}/{global_episodes} ({global_sr}%)")
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    main()
