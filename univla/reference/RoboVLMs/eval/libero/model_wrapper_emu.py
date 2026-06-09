import json
import os
import re
import torch
import numpy as np
from queue import Queue
from PIL import Image
from pathlib import Path

from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import sys
UNIVLA_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(UNIVLA_ROOT))
sys.path.insert(0, str(UNIVLA_ROOT / "reference" / "Emu3"))
from emu3.mllm import Emu3Tokenizer, Emu3ForCausalLM, Emu3Processor
from emu3.mllm import Emu3MoE
from transformers import LogitsProcessor

class ActionIDConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        """
        :param allowed_token_ids: 允许的token ID列表
        """
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        # 创建掩码：允许的token位置为True，其他为False
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if mask.ndim == 1:
            mask[self.allowed_token_ids] = True
        else:
            mask[:, self.allowed_token_ids] = True
        
        # 将不允许的token概率设为负无穷
        scores[~mask] = -float("inf")
        return scores

class EmuVLAModel:
    HARDCODED_ACTION_HIGH = np.array([
        0.93712500009996,
        0.86775000009256,
        0.93712500009996,
        0.13175314309916836,
        0.19275000005139997,
        0.3353504997073735,
        0.9996000000999599,
    ])
    HARDCODED_ACTION_LOW = np.array([
        -0.7046250000751599,
        -0.80100000008544,
        -0.9375000001,
        -0.11467779149968735,
        -0.16395000004372,
        -0.2240490058320433,
        -1.0000000001,
    ])

    # model option
    def __init__(
        self,
        emu_hub,
        vq_hub,
        vision_hub,
        device,
        use_cot: bool = False,
        cot_max_new_tokens: int = 1024,
        use_perspective_gen: bool = False,
        use_multiview_gen: bool = False,
        perspective_obs_key: str = "robot0_eye_in_hand_image",
        use_gripper = True,
        normalizer_path: str | None = None,
        normalizer_key: str = "libero",
    ):

        self.emu_hub = emu_hub
        self.vq_hub = vq_hub
        self.vision_hub = vision_hub
        self.device = device

        ## hard code here
        self.window_size = 2
        self.predict_action_frames = 10
        self.context_frames = 1
        self.predict_frames = 1
        self.action_dim = 7
        self.use_fast = True
        self.use_one_step = False
        self.eoa_token_id = 151845
        self.use_cot = use_cot
        self.cot_max_new_tokens = cot_max_new_tokens
        self.use_perspective_gen = use_perspective_gen
        self.use_multiview_gen = use_multiview_gen
        self.perspective_obs_key = perspective_obs_key
        self.use_gripper = use_gripper
        self.normalizer_path = normalizer_path
        self.normalizer_key = normalizer_key
        # if self.use_cot:
        #     self.use_gripper = False
        self.video_mode = False
        self.action_low, self.action_high = self._load_action_normalizer()
    
        # load model and tokenizer
        self.init_config(device=device)
        self.image_processor.min_pixels = 80 * 80

        self.kwargs = dict(
            mode='VLA_COT' if self.use_cot else 'VLA',
            padding="longest",
        )
        if self.use_fast:
            self.GENERATION_CONFIG = GenerationConfig(
                    pad_token_id=self.model.config.pad_token_id,
                    bos_token_id=self.model.config.bos_token_id,
                    eos_token_id=self.eoa_token_id,
                    do_sample=False,
                )
        else:
            self.GENERATION_CONFIG = GenerationConfig(
                use_cache=True,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
                max_new_tokens=800,
                do_sample=True,
                top_k=2048,
                temperature=0.8,
            )

    def _load_action_normalizer(self):
        if not self.normalizer_path:
            print("[LIBERO normalizer] Using hardcoded action q01/q99 values.")
            return (
                self.HARDCODED_ACTION_LOW.copy(),
                self.HARDCODED_ACTION_HIGH.copy(),
            )

        norm_path = self.normalizer_path
        if os.path.isdir(norm_path):
            norm_path = os.path.join(norm_path, "norm_stats.json")

        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalizer file not found: {norm_path}")

        with open(norm_path, "r") as f:
            norm_payload = json.load(f)

        norm_stats = norm_payload.get("norm_stats", norm_payload)
        if self.normalizer_key not in norm_stats:
            raise KeyError(
                f"Normalizer key '{self.normalizer_key}' not found in {norm_path}. "
                f"Available keys: {list(norm_stats.keys())}"
            )

        stats = norm_stats[self.normalizer_key]
        if "q01" not in stats or "q99" not in stats:
            raise KeyError(
                f"Normalizer at {norm_path} key '{self.normalizer_key}' must contain 'q01' and 'q99'."
            )

        action_low = np.array(stats["q01"], dtype=np.float64)
        action_high = np.array(stats["q99"], dtype=np.float64)
        if action_low.shape != action_high.shape:
            raise ValueError(
                f"Normalizer q01/q99 shape mismatch: {action_low.shape} vs {action_high.shape}"
            )

        print(f"[LIBERO normalizer] Using {norm_path} (key={self.normalizer_key}).")
        return action_low, action_high

    def _perspective_obs_key_to_view_name(self, obs_key: str) -> str:
        alias_map = {
            "gripper_image": "gripper view",
            "robot0_eye_in_hand_image": "gripper view",
            "birdview_image": "bird view",
            "sideview_image": "side view",
            "agentview_image": "agent view",
        }
        if obs_key in alias_map:
            return alias_map[obs_key]
        cleaned = obs_key
        if cleaned.endswith("_image"):
            cleaned = cleaned[: -len("_image")]
        cleaned = cleaned.replace("robot0_", "").replace("_", " ").strip()
        if cleaned.endswith("view"):
            cleaned = cleaned[: -len("view")].strip() + " view"
        elif "view" not in cleaned:
            cleaned = cleaned + " view"
        return cleaned

    def _build_multiview_prompt(self, goal: str) -> str:
        target_view = self._perspective_obs_key_to_view_name(self.perspective_obs_key)
        return (
            f"Given the image of the current state from the agent view, generate the current state from the "
            f"{target_view}, then output the low-level action(s) the robot should take to {goal}."
        )

    def init_config(self, device):
        attn_impl = os.environ.get("EMU_ATTENTION_IMPL", "flash_attention_2")
        self.model = Emu3MoE.from_pretrained(
            self.emu_hub,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.model.to(device).eval()

        self.tokenizer = Emu3Tokenizer.from_pretrained(
            self.vq_hub,
            model_max_length=self.model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True).to(device).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)

        self.boa_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.boa_token)
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eot_token)
        if self.use_cot or self.use_perspective_gen or self.use_multiview_gen:
            self.COT_CONFIG = GenerationConfig(
                pad_token_id=self.model.config.pad_token_id,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.eot_token_id,
                do_sample=False,
            )

        # fast tokenization
        fast_path = os.environ.get("ACTION_TOKENIZER_PATH", str(UNIVLA_ROOT / "pretrain" / "fast"))
        self.action_tokenizer = AutoProcessor.from_pretrained(fast_path, trust_remote_code=True)

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0
        self.last_generation_info = {}

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    
    def add_image(self, image):
        if self.vision_queue.full():
            self.vision_queue.get()
        self.vision_queue.put(image)
    
    def get_history(self):
        return list(self.vision_queue.queue) 

    def add_action(self, action):
        if self.action_queue.full():
            self.action_queue.get()
        self.action_queue.put(action)
    
    def get_action_history(self):
        return list(self.action_queue.queue)


    def reset(self):

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []
        self.last_generation_info = {}

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def preprocess(self, image):
        # preprocess image
        agent_view = image['full_image']
        agent_view = Image.fromarray(agent_view)
        agent_view = agent_view.resize((200, 200))
        image_x = self.image_processor(agent_view, return_tensors="pt")["pixel_values"].to(self.device)
        image_code = self.image_tokenizer.encode(image_x)

        gripper_code = None
        if "wrist_image" in image:
            gripper_view = image['wrist_image']
            gripper_view = Image.fromarray(gripper_view)
            gripper_view = gripper_view.resize((200, 200))
            gripper_x = self.image_processor(gripper_view, return_tensors="pt")["pixel_values"].to(self.device)
            gripper_code = self.image_tokenizer.encode(gripper_x)

        return (
            image_code,
            gripper_code,
        )

    def _normalize_goal_tokens(self, goal_tokens):
        if isinstance(goal_tokens, np.ndarray):
            tokens = torch.from_numpy(goal_tokens)
        elif torch.is_tensor(goal_tokens):
            tokens = goal_tokens.detach().cpu()
        else:
            raise TypeError(f"Unsupported goal_tokens type: {type(goal_tokens)}")
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3:
            raise ValueError(f"Expected goal_tokens with 3 dims [T,H,W], got shape {tokens.shape}")
        return tokens

    def _append_teacher_goal_tokens(self, context_input_ids, context_attention_mask, goal_tokens):
        goal_tokens = self._normalize_goal_tokens(goal_tokens)
        goal_prompt = self.processor.format_video_prompt(goal_tokens)
        goal_inputs = self.tokenizer(
            goal_prompt,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        goal_input_ids = goal_inputs["input_ids"].to(self.device)
        goal_attention_mask = goal_inputs["attention_mask"].to(self.device)
        context_input_ids = torch.cat([context_input_ids, goal_input_ids], dim=1)
        context_attention_mask = torch.cat([context_attention_mask, goal_attention_mask], dim=1)
        eot_tensor = torch.full(
            (context_input_ids.size(0), 1),
            self.eot_token_id,
            dtype=context_input_ids.dtype,
            device=context_input_ids.device,
        )
        context_input_ids = torch.cat([context_input_ids, eot_tensor], dim=1)
        context_attention_mask = torch.cat([context_attention_mask, torch.ones_like(eot_tensor)], dim=1)
        return context_input_ids, context_attention_mask

    def _encode_visual_condition_image(self, image):
        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((200, 200))
        image_x = self.image_processor(pil_image, return_tensors="pt")["pixel_values"].to(self.device)
        image_code = self.image_tokenizer.encode(image_x)
        if image_code.ndim == 4:
            image_code = image_code.squeeze(1)
        return image_code

    def _normalize_visual_condition_tokens(self, token_grid):
        if isinstance(token_grid, np.ndarray):
            token_tensor = torch.from_numpy(token_grid)
        elif torch.is_tensor(token_grid):
            token_tensor = token_grid.detach()
        else:
            raise TypeError(f"Unsupported visual token grid type: {type(token_grid)}")

        token_tensor = token_tensor.to(self.device, dtype=torch.long)
        if token_tensor.ndim == 2:
            token_tensor = token_tensor.unsqueeze(0)
        elif token_tensor.ndim == 3 and token_tensor.shape[0] == 1:
            pass
        else:
            raise ValueError(
                f"Expected generated visual tokens with shape [H, W] or [1, H, W], got {tuple(token_tensor.shape)}"
            )
        return token_tensor

    def encode_visual_condition_image_tokens(self, image):
        image_code = self._encode_visual_condition_image(image)
        if image_code.ndim == 3 and image_code.shape[0] == 1:
            image_code = image_code[0]
        return image_code.detach().cpu().long()

    def extract_visual_token_grids(self, thought_text):
        if not thought_text:
            return []

        boi = re.escape(self.tokenizer.boi_token)
        eoi = re.escape(self.tokenizer.eoi_token)
        img_token = re.escape(self.tokenizer.img_token)
        visual_pattern = re.compile(self.processor.visual_template[1])
        pattern = re.compile(
            rf"{boi}(\d+)\*(\d+)\*(\d+){img_token}(.*?){eoi}",
            re.DOTALL,
        )

        token_grids = []
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
            for i in range(max(frames, 1)):
                frame_tokens = token_ids[i * height * width:(i + 1) * height * width]
                token_grid = torch.tensor(frame_tokens, dtype=torch.long).reshape(height, width)
                token_grids.append(token_grid)
        return token_grids

    def decode_visual_tokens(self, token_grid):
        if isinstance(token_grid, np.ndarray):
            token_tensor = torch.from_numpy(token_grid)
        else:
            token_tensor = token_grid.detach().cpu()
        if token_tensor.ndim != 2:
            raise ValueError(f"Expected [H, W] token grid, got shape {tuple(token_tensor.shape)}")

        token_tensor = token_tensor.to(self.processor.vision_tokenizer.device, dtype=torch.long)
        decoded = self.processor.vision_decode(token_tensor[None]).float()
        image = self.processor.image_processor.postprocess(decoded)["pixel_values"][0]
        image = np.asarray(image)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    def get_last_generation_info(self):
        return self.last_generation_info

    def _append_teacher_perspective_tokens(self, context_input_ids, context_attention_mask, perspective_image):
        perspective_tokens = self._encode_visual_condition_image(perspective_image)
        bot_tensor = torch.full(
            (context_input_ids.size(0), 1),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.bot_token),
            dtype=context_input_ids.dtype,
            device=context_input_ids.device,
        )
        context_input_ids = torch.cat([context_input_ids, bot_tensor], dim=1)
        context_attention_mask = torch.cat([context_attention_mask, torch.ones_like(bot_tensor)], dim=1)

        perspective_prompt = self.tokenizer(
            self.processor.format_video_prompt(perspective_tokens),
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        perspective_input_ids = perspective_prompt["input_ids"].to(self.device)
        perspective_attention_mask = perspective_prompt["attention_mask"].to(self.device)
        context_input_ids = torch.cat([context_input_ids, perspective_input_ids], dim=1)
        context_attention_mask = torch.cat([context_attention_mask, perspective_attention_mask], dim=1)
        eot_tensor = torch.full(
            (context_input_ids.size(0), 1),
            self.eot_token_id,
            dtype=context_input_ids.dtype,
            device=context_input_ids.device,
        )
        context_input_ids = torch.cat([context_input_ids, eot_tensor], dim=1)
        context_attention_mask = torch.cat([context_attention_mask, torch.ones_like(eot_tensor)], dim=1)
        return context_input_ids, context_attention_mask

    def _build_perspective_context(self, prompt, video_code, gripper_code):
        text_prompt = self.tokenizer(
            self.tokenizer.bos_token + prompt,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        context_input_ids = text_prompt["input_ids"].to(self.device)
        context_attention_mask = text_prompt["attention_mask"].to(self.device)

        obs_prompt = self.tokenizer(
            self.processor.format_video_prompt(video_code[0]),
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        context_input_ids = torch.cat([context_input_ids, obs_prompt["input_ids"].to(self.device)], dim=1)
        context_attention_mask = torch.cat([context_attention_mask, obs_prompt["attention_mask"].to(self.device)], dim=1)

        if gripper_code is not None:
            gripper_prompt = self.tokenizer(
                self.processor.format_video_prompt(gripper_code[0]),
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            context_input_ids = torch.cat([context_input_ids, gripper_prompt["input_ids"].to(self.device)], dim=1)
            context_attention_mask = torch.cat([context_attention_mask, gripper_prompt["attention_mask"].to(self.device)], dim=1)

        return context_input_ids, context_attention_mask

    def generate_perspective_tokens(self, image, goal):
        self.last_generation_info = {}
        image_code, gripper_code = self.preprocess(image)

        prompt = self._build_multiview_prompt(goal) if self.use_multiview_gen else goal
        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper and gripper_code is not None else None
        context_input_ids, context_attention_mask = self._build_perspective_context(
            prompt, video_code, gripper_code
        )
        context_length = context_input_ids.shape[-1]
        bot_tensor = torch.full(
            (context_input_ids.size(0), 1),
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.bot_token),
            dtype=context_input_ids.dtype,
            device=context_input_ids.device,
        )
        context_input_ids = torch.cat([context_input_ids, bot_tensor], dim=1)
        context_attention_mask = torch.cat([context_attention_mask, torch.ones_like(bot_tensor)], dim=1)

        with torch.no_grad():
            perspective_outputs = self.model.generate(
                context_input_ids,
                self.COT_CONFIG,
                max_new_tokens=self.cot_max_new_tokens,
                attention_mask=context_attention_mask,
            )

        perspective_tokens = perspective_outputs[:, context_length + 1:]
        thought_text = self.tokenizer.decode(perspective_tokens[0], skip_special_tokens=False)
        perspective_token_grids = self.extract_visual_token_grids(thought_text)
        perspective_images = []
        for token_grid in perspective_token_grids:
            try:
                perspective_images.append(self.decode_visual_tokens(token_grid))
            except Exception:
                perspective_images = []
                break
        self.last_generation_info = {
            "mode": "multiview_gen" if self.use_multiview_gen else "perspective_gen",
            "thought_text": thought_text,
            "perspective_token_grids": perspective_token_grids,
            "perspective_images": perspective_images,
            "parse_success": len(perspective_token_grids) > 0,
        }
        if not perspective_token_grids:
            return None, thought_text
        return perspective_token_grids[0], thought_text

    def step(self, image, goal, goal_tokens=None, perspective_image=None, gripper_token_grid=None):
        input_dict = dict()
        self.last_generation_info = {}
        step_idx = self.rollout_step_counter
        self.rollout_step_counter += 1
        
        image_code, gripper_code = self.preprocess(image)
        if gripper_token_grid is not None:
            gripper_code = self._normalize_visual_condition_tokens(gripper_token_grid)

        prompt_text = goal
        if self.use_multiview_gen:
            prompt_text = self._build_multiview_prompt(goal)
        elif self.use_cot or self.use_perspective_gen or perspective_image is not None:
            # prompt_text = f"Given the image of the current state, what actions should the robot take to {goal}? Output the low-level action(s) to take."
            prompt_text = goal
            
        prompt,neg_prompt = prompt_text,""

        video_code = image_code.unsqueeze(1)
        gripper_code = gripper_code.unsqueeze(1) if self.use_gripper else None

        text_prompt = [self.tokenizer.bos_token + prompt]
        text_tokens = self.processor.tokenizer(text_prompt)
        
        text_tokens = BatchFeature(data={**text_tokens}, tensor_type='pt')

        if self.video_mode:
            kwargs = dict(
                    mode='VLA_Video',
                    padding="longest",
                )
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **kwargs)
        elif self.use_perspective_gen or self.use_multiview_gen or perspective_image is not None:
            pos_inputs = None
        else:
            pos_inputs = self.processor.video_process(text=prompt, video_tokens=video_code, gripper_tokens=gripper_code ,context_frames=self.context_frames, frames = self.predict_frames, return_tensors="pt", **self.kwargs)
        
        if self.video_mode:
            self.add_image(pos_inputs)
            
            # 获取历史图像和动作
            history = self.get_history()
            action_history = self.get_action_history()

            # 初始化输入ID、token类型ID和attention mask
            all_input_ids = []
            all_token_type_ids = []
            all_attention_mask = []

            # Add text
            all_input_ids.append(text_tokens['input_ids'])
            all_token_type_ids.append(text_tokens['token_type_ids'])
            all_attention_mask.append(text_tokens['attention_mask'])

            # 遍历历史图像
            for i in range(len(history)):
                img_input_ids = history[i]['input_ids']
                img_token_type_ids = history[i]['token_type_ids']
                img_attention_mask = history[i]['attention_mask']
                
                # 对应的动作
                if i < len(action_history):
                    act_input_ids = action_history[i]
                    
                    # 动作的token_type_ids和attention_mask分别填充为全0和全1
                    act_token_type_ids = torch.zeros_like(act_input_ids)
                    act_attention_mask = torch.ones_like(act_input_ids)
                    
                    # 交替添加图像和动作数据
                    all_input_ids.extend([img_input_ids, act_input_ids])
                    all_token_type_ids.extend([img_token_type_ids, act_token_type_ids])
                    all_attention_mask.extend([img_attention_mask, act_attention_mask])
                else:
                    # 若没有对应的动作，添加图像数据
                    all_input_ids.append(img_input_ids)
                    all_token_type_ids.append(img_token_type_ids)
                    all_attention_mask.append(img_attention_mask)
            # 拼接所有的input_ids、token_type_ids和attention_mask
            concatenated_input_ids = torch.cat(all_input_ids, dim=1)
            concatenated_token_type_ids = torch.cat(all_token_type_ids, dim=1)
            concatenated_attention_mask = torch.cat(all_attention_mask, dim=1)
            
            # 更新pos_inputs
            final_inputs = pos_inputs.copy()
            final_inputs['input_ids'] = concatenated_input_ids
            final_inputs['token_type_ids'] = concatenated_token_type_ids
            final_inputs['attention_mask'] = concatenated_attention_mask
        elif self.use_perspective_gen or self.use_multiview_gen or perspective_image is not None:
            context_input_ids, context_attention_mask = self._build_perspective_context(
                prompt, video_code, gripper_code
            )
            context_length = context_input_ids.shape[-1]
        else:
            final_inputs = pos_inputs
            context_input_ids = final_inputs.input_ids.to(self.device)
            context_attention_mask = final_inputs.attention_mask.to(self.device)
            context_length = context_input_ids.shape[-1]
        thought_text = ""

        if self.use_cot:
            if goal_tokens is None:
                with torch.no_grad():
                    cot_outputs = self.model.generate(
                        context_input_ids,
                        self.COT_CONFIG,
                        max_new_tokens=self.cot_max_new_tokens,
                        attention_mask=context_attention_mask,
                    )
                cot_tokens = cot_outputs[:, context_length:]
                thought_text = self.tokenizer.decode(cot_tokens[0], skip_special_tokens=False)
                context_input_ids = cot_outputs
                context_attention_mask = torch.ones_like(context_input_ids)
            else:
                context_input_ids, context_attention_mask = self._append_teacher_goal_tokens(
                    context_input_ids, context_attention_mask, goal_tokens
                )
                thought_text = "teacher_forced_goal"
            boa_tensor = torch.full(
                (context_input_ids.size(0), 1),
                self.boa_token_id,
                dtype=context_input_ids.dtype,
                device=context_input_ids.device,
            )
            context_input_ids = torch.cat([context_input_ids, boa_tensor], dim=1)
            context_attention_mask = torch.ones_like(context_input_ids)
            context_length = context_input_ids.shape[-1]
        elif self.use_perspective_gen or self.use_multiview_gen:
            bot_tensor = torch.full(
                (context_input_ids.size(0), 1),
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.bot_token),
                dtype=context_input_ids.dtype,
                device=context_input_ids.device,
            )
            context_input_ids = torch.cat([context_input_ids, bot_tensor], dim=1)
            context_attention_mask = torch.cat([context_attention_mask, torch.ones_like(bot_tensor)], dim=1)
            before_generation_text = self.tokenizer.decode(
                context_input_ids[0], skip_special_tokens=False
            )
            with torch.no_grad():
                perspective_outputs = self.model.generate(
                    context_input_ids,
                    self.COT_CONFIG,
                    max_new_tokens=self.cot_max_new_tokens,
                    attention_mask=context_attention_mask,
                )
            after_generation_text = self.tokenizer.decode(
                perspective_outputs[0], skip_special_tokens=False
            )
            # with open("cot_debug/perspective_gen_step00000_before.txt", "w") as f:
            #     print(
            #         before_generation_text,
            #         file=f
            #     )
            # with open("cot_debug/perspective_gen_step00000_after.txt", "w") as f:
            #     print(
            #         after_generation_text,
            #         file=f
            #     )
            #     exit(0)
            perspective_tokens = perspective_outputs[:, context_length + 1:]
            thought_text = self.tokenizer.decode(perspective_tokens[0], skip_special_tokens=False)
            perspective_token_grids = self.extract_visual_token_grids(thought_text)
            perspective_images = []
            for token_grid in perspective_token_grids:
                try:
                    perspective_images.append(self.decode_visual_tokens(token_grid))
                except Exception:
                    perspective_images = []
                    break
            self.last_generation_info = {
                "mode": "multiview_gen" if self.use_multiview_gen else "perspective_gen",
                "thought_text": thought_text,
                "perspective_token_grids": perspective_token_grids,
                "perspective_images": perspective_images,
                "parse_success": len(perspective_token_grids) > 0,
            }
            context_input_ids = perspective_outputs
            context_attention_mask = torch.ones_like(context_input_ids)
            boa_tensor = torch.full(
                (context_input_ids.size(0), 1),
                self.boa_token_id,
                dtype=context_input_ids.dtype,
                device=context_input_ids.device,
            )
            context_input_ids = torch.cat([context_input_ids, boa_tensor], dim=1)
            context_attention_mask = torch.ones_like(context_input_ids)
            context_length = context_input_ids.shape[-1]
        elif perspective_image is not None:
            context_input_ids, context_attention_mask = self._append_teacher_perspective_tokens(
                context_input_ids, context_attention_mask, perspective_image
            )
            boa_tensor = torch.full(
                (context_input_ids.size(0), 1),
                self.boa_token_id,
                dtype=context_input_ids.dtype,
                device=context_input_ids.device,
            )
            context_input_ids = torch.cat([context_input_ids, boa_tensor], dim=1)
            context_attention_mask = torch.ones_like(context_input_ids)
            context_length = context_input_ids.shape[-1]

        if self.use_fast: 
            last_token_id = self.tokenizer.pad_token_id - 1
            allowed_token_ids = list(range(last_token_id - self.action_tokenizer.vocab_size, last_token_id + 1)) + [self.eoa_token_id]
            action_id_processor = ActionIDConstraintLogitsProcessor(allowed_token_ids)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    context_input_ids,
                    self.GENERATION_CONFIG,
                    max_new_tokens=80,
                    logits_processor=[action_id_processor],
                    attention_mask=context_attention_mask,
                )
            # omit the eoa token
            orig_outputs = outputs[:, context_length:]
            # with open("cot_debug/eval_prompt.txt", "w") as f:
            #     print(
            #         self.tokenizer.decode(outputs[0], skip_special_tokens=False),
            #         file=f
            #     )
            #     exit(0)
            outputs = outputs[:, context_length:-1]
            last_token_id_tensor = torch.tensor(last_token_id, dtype=outputs.dtype, device=outputs.device)
            processed_outputs = last_token_id_tensor - outputs
            action_outputs = self.action_tokenizer.decode(
                processed_outputs, time_horizon=self.predict_action_frames, action_dim=self.action_dim
            )
            action = action_outputs[0]
            if self.video_mode:
                self.add_action(orig_outputs.detach().cpu())

        else:
            pass
        
        # unnormalize action
        action = self.unormalize_action(action)

        # NOTE(zbzhu): Flip the gripper action here
        # refer to https://github.com/openvla/openvla/blob/1b024f242eda833dc8e321953f25cfd5f3d2f76d/experiments/robot/libero/run_libero_eval.py#L225
        action[..., -1] = np.where(action[..., -1] > 0, 1, -1)

        
        if self.use_one_step:
            # only one step
            action_pred = action[0:1]
        else:
            # action chunk
            action_pred = action
        
        if self.use_cot or self.use_perspective_gen or self.use_multiview_gen:
            return action_pred, [thought_text.replace("\n", "@")]
        else:
            return action_pred
    
    def unormalize_action(self, action):
        action = 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low
        return action
