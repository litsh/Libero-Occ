"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import sys
import numpy as np
LIBERO_ROOT = os.environ.get("LIBERO_ROOT")
if LIBERO_ROOT:
    sys.path.insert(0, LIBERO_ROOT)
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from moviepy.editor import ImageSequenceClip

def get_episode_length(task_suite_name="libero_object") -> int:
    if "libero_spatial" in task_suite_name:
        max_steps = 220  # longest training demo has 193 steps
    elif  "libero_object" in task_suite_name:
        max_steps = 280  # longest training demo has 254 steps
    elif "libero_goal" in task_suite_name:
        max_steps = 300  # longest training demo has 270 steps
    elif "libero_10" in task_suite_name:
        max_steps = 520  # longest training demo has 505 steps
    elif task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        print(f"ERROR: Invalid task suite name {task_suite_name}")
        sys.exit(-1)
    return max_steps


def get_libero_env(task, resolution=256, render_gpu_device_id=-1, camera_names=None):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "render_gpu_device_id": render_gpu_device_id,
    }
    if camera_names is not None:
        env_args["camera_names"] = camera_names
    env = OffScreenRenderEnv(**env_args)
    env.seed(
        0
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(
        img, expand_animations=False, dtype=tf.uint8
    )  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_camera_image(obs, camera_key):
    """Extract and preprocess an arbitrary LIBERO camera observation."""
    if camera_key == "gripper_image":
        camera_key = "robot0_eye_in_hand_image"
    if camera_key == "robot0_eye_in_hand_image":
        return get_libero_wrist_image(obs)
    if camera_key == "agentview_image":
        return get_libero_image(obs)
    if camera_key not in obs:
        raise KeyError(
            f"Camera key '{camera_key}' not found in observation. "
            f"Available keys: {sorted(obs.keys())}"
        )
    img = obs[camera_key]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_gif(img_list, path, fps=15):
    # print(len(img_list))
    # print(img_list[0].shape)
    clip = ImageSequenceClip(img_list, fps=fps)
    clip.write_gif(path, fps=fps)


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
