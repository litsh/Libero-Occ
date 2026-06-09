# -*- coding: utf-8 -*-
import json
import os
import os.path as osp
import random
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union
from torch.utils.data import Dataset
from PIL import Image
import sys
from pathlib import Path
UNIVLA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UNIVLA_ROOT))
from models.tokenizer.action_tokenizer import ActionTokenizer
from transformers import AutoModel, AutoImageProcessor, GenerationConfig, AutoProcessor
    
class Emu3SFTDataset(Dataset):

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__()

        self.args = args
        # data args
        self.random_frame_sampling = args.random_frame_sampling
        self.raw_image = args.raw_image
        
        with open(args.data_path,'rb') as f:
            self.data = pickle.load(f)
        
        if not self.random_frame_sampling:
            self.data = list(self.sliding_window_sampling(self.data, interval=args.action_frames*args.frames))
        
        self.tokenizer = tokenizer
        self.bov = tokenizer.encode(args.visual_token_pattern.format(token_id=0))[0]
        self.eov = tokenizer.encode(args.visual_token_pattern.format(token_id=args.codebook_size - 1))[0]
        self.chat_template="You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:"
        self.gen_template="You are a powerful painter. USER: {text} ASSISTANT:{image}"
        self.act_template="Action: {action_prompt}"
        self.VL = args.VL
        self.cfg = False
        self.post_training = args.post_training

        # pretrain use
        if self.post_training:
            # v2
            # self.dataset_fps = {'rt1':3, 'bridgev2':5, 'droid':15, '1x':1, 'kuka':3, 'calvin':5, 'libero':5} 
            # v3
            self.dataset_fps = {'1x':1, 'SSv2':1,'rt1':3, 'kuka':3, \
                                'bridgev2':5, 'taco_play':5, \
                                'calvin':10, 'libero':10,'maniskill':10,'cmu_play_fusion':10,'utaustin_mutex':10, \
                                'droid':15, 'viola':15, \
                                'toto':20}
        else:
            self.dataset_fps = {}
        self.T = args.frames
        self.action_frames = args.action_frames
        
        self.actions = args.actions
        self.actions_format = args.actions_format

        self.use_gripper = args.use_gripper  

        self.video_format = args.video_format

        if self.raw_image:
            self.vision_hub = getattr(args, "vision_hub", None) or os.environ.get("VISION_HUB")
            if not self.vision_hub:
                raise ValueError("raw_image=True requires args.vision_hub or VISION_HUB.")
            self.image_processor = AutoImageProcessor.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_tokenizer = AutoModel.from_pretrained(self.vision_hub, trust_remote_code=True)
            self.image_processor.min_pixels = 80 * 80
        if self.actions_format == "openvla":
            self.action_tokenizer = ActionTokenizer(tokenizer, bins=256, min_action=-1.0, max_action=1.0)
        elif self.actions_format == "fast":
            self.fast_path = args.action_tokenizer_path
            self.action_tokenizer = AutoProcessor.from_pretrained(self.fast_path, trust_remote_code=True)

    def __len__(self):
        return len(self.data)
    
    def sliding_window_sampling(self, data, interval=5):
        """
        Implement sliding window sampling using a generator.
        """
        for item in data:
            T = len(item['image'])
            if T <= interval:
                raise ValueError("Length of 'image', 'action', and 'gripper' must be greater than 'interval'.")
            for start_idx in range(0, T - interval + 1, 1):
                yield {
                    'text': item['text'],
                    'image': item['image'][start_idx:start_idx+interval],
                    'action': item['action'][start_idx:start_idx+interval],
                    'gripper_image': item['gripper_image'][start_idx:start_idx+interval],
                }

    def random_frames_to_tensor(self, img_list, T, action_prompt=None, gripper=None):
        
        start_idx = random.randint(0, len(img_list) - T)

        if hasattr(self, 'raw_image') and self.raw_image:
            self.image_tokenizer.eval()
            # Process raw images with VQ encoding
            selected_frames = [Image.open(img_path) for img_path in img_list[start_idx:start_idx + T]]
            selected_frames = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_frames]

            tensor_frames = torch.stack(selected_frames, dim=0)
            with torch.no_grad():
                image_code = self.image_tokenizer.encode(tensor_frames)
            
            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [Image.open(img_path) for img_path in gripper[start_idx:start_idx + T]]
                selected_gripper = [self.image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0) for img in selected_gripper]
                tensor_gripper = torch.stack(selected_gripper, dim=0)
                with torch.no_grad():
                    gripper_code = self.image_tokenizer.encode(tensor_gripper)
                return image_code, selected_actions, gripper_code
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                return image_code, selected_actions
        else:
            selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
            tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
            tensor = torch.stack(tensor_frames, dim=1)

            if gripper is not None and action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                return tensor.squeeze(0), selected_actions, torch.stack(tensor_gripper, dim=1).squeeze(0)
            elif action_prompt is not None:
                selected_actions = action_prompt[start_idx:start_idx + T]
                return tensor.squeeze(0), selected_actions
            elif gripper is not None:
                selected_gripper = [np.load(img_path) for img_path in gripper[start_idx:start_idx + T]]
                tensor_gripper = [torch.from_numpy(frame) for frame in selected_gripper]
                return tensor.squeeze(0), torch.stack(tensor_gripper, dim=1).squeeze(0)
        return tensor.squeeze(0)
    
    def get_fps_for_path(self, image_tokens_path):
        for key in self.dataset_fps.keys():
            if key in image_tokens_path[0]:
                return self.dataset_fps[key]
        # Default return value if no key matches
        return None  # or some default FPS value
    
    def pad_tensor(self, tensor, max_length, pad_value):
        """Pads a tensor to a specified maximum length."""
        current_length = tensor.shape[-1]
        if current_length < max_length:
            pad_length = max_length - current_length
            padding = torch.full((pad_length,), fill_value=pad_value, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=-1)
        return tensor

    def __getitem__(self, index: int):

        scene = self.data[index]

        if self.cfg:
            p_prob = random.random()
            if p_prob < self.args.null_prompt_prob:
                prompt = ""
            else:
                prompt = scene["text"]
        else:
            prompt = scene["text"]

        image_tokens_path = scene["image"]

        # handle different dataset fps for post training
        fps = self.get_fps_for_path(image_tokens_path)
        if fps is not None:
            self.action_frames = fps
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"] 
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, action_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, gripper=gripper)
            else:
                image_tokens, action_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)

                    # Debugging
                    # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                    # error = action_tokens - action_debug
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            if self.use_gripper:
                gripper = scene["gripper_image"]
                image_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, gripper=gripper)
            else:
                image_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num) 
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            if self.use_gripper:
                gripper_tokens = gripper_tokens[0:self.T,...]
                gripper_prompt = self.format_video_prompt(gripper_tokens)
                image_prompt = image_prompt + gripper_prompt  

            if self.VL:
                p_prob_order = random.random()
                if p_prob_order < 0.5:
                    input = self.tokenizer.bos_token + prompt + image_prompt + self.tokenizer.eos_token
                else:
                    # input = self.tokenizer.bos_token + image_prompt + prompt
                    input = self.tokenizer.bos_token + self.chat_template.format(image_prompt=image_prompt, text_prompt=prompt) + self.tokenizer.eos_token
            else:
                input = self.tokenizer.bos_token + prompt + image_prompt 
            # 先不进行padding，后面统一padding
            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # only use vision loss
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.actions_format == "openvla":
                    action_sample = self.wrap_action_sequence(action_ids)
                    sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                    # Update attention_mask
                    action_mask = torch.ones_like(action_sample, dtype=torch.long)
                    sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                    action_labels = action_sample.clone()  # Clone action_sample for labels
                    sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                
                # FAST
                elif self.actions_format == "fast":
                    if self.args.apply_loss_on_only_action:
                        sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    sample = self.append_action_to_sample(sample, action_ids)
                
                # Flow Matching
                elif self.actions_format == "continuous":
                    boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                    sample = self.append_boa_to_sample(sample, [boa_token_id])
                    sample["action"] = action_continuous
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample

    def append_action_to_sample(self, sample, action_ids):
        """
        将 action_ids 处理后，追加到 sample 中，包括 input_ids, attention_mask 和 labels。
        """
        action_sample = self.wrap_action_sequence(action_ids)
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample
    
    def append_boa_to_sample(self, sample, action_ids):

        action_sample = torch.tensor(action_ids, dtype=torch.long)
        action_mask = torch.ones_like(action_sample, dtype=torch.long)

        for key, value in zip(["input_ids", "attention_mask", "labels"], [action_sample, action_mask, action_sample.clone()]):
            sample[key] = torch.cat([sample[key], value], dim=-1)

        return sample

    def wrap_action_sequence(self, action_ids: List[int]) -> torch.Tensor:
        """
        Wraps a sequence of action token IDs with special tokens (beginning and end).

        Args:
            action_ids (List[int]): The sequence of action token IDs.

        Returns:
            torch.Tensor: A tensor containing the wrapped sequence.
        """
        # Encode the beginning and end action tokens
        action_begin = self.tokenizer.encode(self.tokenizer.boa_token)[0]
        action_end = self.tokenizer.encode(self.tokenizer.eoa_token)[0]
        eos = self.tokenizer.encode(self.tokenizer.eos_token)[0]

        # Wrap the action sequence
        # wrapped_action = [action_begin] + action_ids + [action_end] + [eos]
        wrapped_action = [action_begin] + action_ids + [action_end]
        
        # Convert to a PyTorch tensor
        return torch.tensor(wrapped_action, dtype=torch.long)

    def format_video_prompt(self, video_tokens):
        # 假设video_tokens是一个形状为[frames, height, width]的张量
        frames, h, w = video_tokens.shape
        videostr = self.to_videostr(video_tokens)

        video_prompt = (
            self.tokenizer.boi_token +
            f"{frames}*{h}*{w}" +  # 视频的帧数、高度和宽度
            self.tokenizer.img_token +  # 视频开始标记
            videostr +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return video_prompt

    def to_videostr(self, video_tokens):
        frame_str_list = []
        for frame in video_tokens:
            frame_token_str = [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in frame.flatten()
            ]
            frame_str = "".join(frame_token_str)
            frame_str_list.append(frame_str)
        videostr = self.tokenizer.eof_token.join(frame_str_list)
        return videostr


    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +
            f"{h}*{w}" +
            self.tokenizer.img_token +
            imgstr +
            self.tokenizer.eol_token +
            self.tokenizer.eof_token +
            self.tokenizer.eoi_token
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        image_token_str = [
            [
                self.args.visual_token_pattern.format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr


class Emu3WorldModelDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        # weights
        dataset_weights = {
            'rt1': 0.3,
            'droid_fast': 0.2,
            'oxembodiment/bridge': 1.0,
            'oxembodiment/toto': 1.0,
            'oxembodiment/taco_play': 1.0,
            'oxembodiment/fmb': 1.0,
            'oxembodiment/maniskill': 0.5,
            'oxembodiment/kuka': 0.1,
            'oxembodiment/berkeley_autolab_ur5': 1.0,
            'calvin': 0.8,
            'libero': 1.0,
        }
        self.datasets_weight = args.datasets_weight
        if self.datasets_weight:
            self.sample_weights = [dataset_weights.get(d["dataset"], 1.0) for d in self.data]
        self.without_text = args.without_text

    def __getitem__(self, index: int):

        scene = self.data[index]

        if self.without_text:
            prompt = ""
        else:
            prompt = scene["text"]

        image_tokens_path = scene["image"]

        # handle different dataset fps for post training
        fps = self.get_fps_for_path(image_tokens_path)
        if fps is not None:
            self.action_frames = fps
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        if self.use_gripper and "gripper_image" in scene:
            gripper = scene["gripper_image"]
            image_tokens, gripper_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num, gripper=gripper)
        else:
            image_tokens = self.random_frames_to_tensor(image_tokens_path, frames_num) 
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper and "gripper_image" in scene:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper and "gripper_image" in scene:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                
                sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        
        else:
            raise NotImplementedError("Only interleave video format is supported for world model dataset.")
        return sample
    
class Emu3RealRobotDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer: "Emu3Tokenizer"):
        super().__init__(args, tokenizer=tokenizer)
        self.use_views = ['cam_high','cam_left_wrist','cam_right_wrist']
    
    def random_frames_to_tensor(self, img_list, T, action_prompt=None, wrist=None):
        
        start_idx = random.randint(0, len(img_list) - T)

        selected_frames = [np.load(img_path) for img_path in img_list[start_idx:start_idx + T]]
        tensor_frames = [torch.from_numpy(frame) for frame in selected_frames]
        tensor = torch.stack(tensor_frames, dim=1)

        wrist_left = wrist["cam_left_wrist"]
        wrist_right = wrist["cam_right_wrist"]

        select_wrist_left = [torch.from_numpy(np.load(img_path)) for img_path in wrist_left[start_idx:start_idx + T]]
        select_wrist_right = [torch.from_numpy(np.load(img_path)) for img_path in wrist_right[start_idx:start_idx + T]]

        tensor_wrist_left = torch.stack(select_wrist_left, dim=1)
        tensor_wrist_right = torch.stack(select_wrist_right, dim=1)

        if action_prompt is None:
            return tensor.squeeze(0), tensor_wrist_left.squeeze(0), tensor_wrist_right.squeeze(0)

        selected_actions = action_prompt[start_idx:start_idx + T]
        return tensor.squeeze(0), tensor_wrist_left.squeeze(0), tensor_wrist_right.squeeze(0), selected_actions
    
    def __getitem__(self, index: int):

        scene = self.data[index]

        prompt = scene["text"]

        image_tokens_path = scene["cam_high"]
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        
        # use action information
        if self.actions:
            action = scene["action"] 
            image_tokens, wrist_left_token, wrist_right_token, action_tokens= self.random_frames_to_tensor(image_tokens_path, frames_num, action_prompt=action, wrist=scene)
            
            if self.video_format == "interleave":
                if self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                    action_ids = self.action_tokenizer(action_tokens)
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                    
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
            else:
                if self.actions_format == "openvla":
                    action_tokens = action_tokens.flatten()
                    action_ids = self.action_tokenizer(action_tokens)
                elif self.actions_format == "text":
                    action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                    action_prompt = self.act_template.format(action_prompt=action_str)
                elif self.actions_format == "continuous":
                    action_continuous = action_tokens
                elif self.actions_format == "fast":
                    if isinstance(action_tokens, list):
                        tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                        # Concatenate tensors along the first dimension
                        action_tokens = torch.cat(tensor_list, dim=0)
                    action_ids = self.action_tokenizer(action_tokens)[0]
                    # action_decode = self.action_tokenizer.decode([action_ids])
                    self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                    action_ids = [self.last_vocab_idx - id for id in action_ids]
                else:
                    raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            image_tokens, wrist_left_token, wrist_right_token = self.random_frames_to_tensor(image_tokens_path, frames_num, wrist=scene)
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            wrist_left_token = wrist_left_token[0::self.action_frames,...]
            wrist_right_token = wrist_right_token[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                wrist_left_prompt = self.format_video_prompt(wrist_left_token[i:i+1])
                wrist_right_prompt = self.format_video_prompt(wrist_right_token[i:i+1])
                image_prompt += wrist_left_prompt + wrist_right_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)

            wrist_left_tokens = wrist_left_token[0:self.T,...]
            wrist_right_tokens = wrist_right_token[0:self.T,...]
            wrist_left_prompt = self.format_video_prompt(wrist_left_tokens)
            wrist_right_prompt = self.format_video_prompt(wrist_right_tokens)
            image_prompt = image_prompt + wrist_left_prompt + wrist_right_prompt
            
            input = self.tokenizer.bos_token + prompt + image_prompt 

            sample = self.tokenizer(
                input,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            labels = sample["input_ids"]

            # only use vision loss
            if self.args.apply_loss_on_only_vision:
                labels = torch.where(torch.logical_and(labels >= self.bov, labels <= self.eov), labels, self.args.ignore_index)

            sample["labels"] = labels
            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.actions_format == "openvla":
                    action_sample = self.wrap_action_sequence(action_ids)
                    sample["input_ids"] = torch.cat([sample["input_ids"], action_sample], dim=-1)

                    # Update attention_mask
                    action_mask = torch.ones_like(action_sample, dtype=torch.long)
                    sample["attention_mask"] = torch.cat([sample["attention_mask"], action_mask], dim=-1)

                    action_labels = action_sample.clone()  # Clone action_sample for labels
                    sample["labels"] = torch.cat([sample["labels"], action_labels], dim=-1)
                
                # FAST
                elif self.actions_format == "fast":
                    if 'state' in scene.keys():
                        state = scene['state'].reshape(1, 1, -1)
                        state_tokens = self.action_tokenizer(state)[0]
                        state_ids = [self.last_vocab_idx - id for id in state_tokens]
                        state_tensor = torch.tensor(state_ids, dtype=sample["input_ids"].dtype, device=sample["input_ids"].device)

                        sample["input_ids"] = torch.cat([sample["input_ids"], state_tensor], dim=-1)

                        state_label_tensor = torch.full_like(state_tensor, fill_value=-100)  # -100 means ignored in loss
                        sample["labels"] = torch.cat([sample["labels"], state_label_tensor], dim=-1)

                        state_mask = torch.ones_like(state_tensor)
                        sample["attention_mask"] = torch.cat([sample["attention_mask"], state_mask], dim=-1)
                    
                    if self.args.apply_loss_on_only_action:
                        sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                    sample = self.append_action_to_sample(sample, action_ids)
                # Flow Matching
                elif self.actions_format == "continuous":
                    boa_token_id = self.tokenizer.encode(self.tokenizer.boa_token)[0]
                    sample = self.append_boa_to_sample(sample, [boa_token_id])
                    sample["action"] = action_continuous
            
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample

class Emu3CoTDataset(Emu3SFTDataset):    

    def __init__(self, args: "DataArguments", tokenizer):
        super().__init__(args, tokenizer=tokenizer)

    def load_token_clip(self, paths):
        frames = [torch.from_numpy(np.load(path)) for path in paths]
        if not frames:
            raise ValueError("No token paths provided for clip loading.")
        tensor = torch.stack(frames, dim=1)
        return tensor.squeeze(0)
    
    def __getitem__(self, index: int):

        scene = self.data[index]
        prompt = scene["text"]
        image_tokens_path = scene["image"]
        
        if self.T > 1 and self.video_format == "interleave":
            if len(image_tokens_path) > self.T * self.action_frames:
                frames_num = self.T * self.action_frames
            else:
                frames_num = (len(image_tokens_path) // self.action_frames) * self.action_frames
        else:
            frames_num = self.action_frames if len(image_tokens_path) >= self.action_frames else len(image_tokens_path)
        reason_entry = random.choice(scene["reasoning"])
        action = scene["action"]

        start_idx = min(max(reason_entry.get("obs_idx", 0), 0), max(len(image_tokens_path) - frames_num, 0))
        end_idx = start_idx + frames_num

        selected_image_paths = image_tokens_path[start_idx:end_idx]
        image_tokens = self.load_token_clip(selected_image_paths)

        action_tokens = action[start_idx:end_idx]
            
        if self.use_gripper:
            gripper_paths = scene["gripper_image"][start_idx:end_idx]
            gripper_tokens = self.load_token_clip(gripper_paths)
        
        if self.video_format == "interleave":
            if self.actions_format == "fast":
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_tokens = action_tokens.reshape(-1, self.action_frames, action_tokens.shape[-1])
                action_ids = self.action_tokenizer(action_tokens)
                self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - torch.tensor(id) for id in action_ids]
                
            else:
                raise ValueError(f"Invalid actions_format: {self.actions_format}")
        else:
            if self.actions_format == "openvla":
                action_tokens = action_tokens.flatten()
                action_ids = self.action_tokenizer(action_tokens)

                # Debugging
                # action_debug = self.action_tokenizer.decode_token_ids_to_actions(action_ids)
                # error = action_tokens - action_debug
            elif self.actions_format == "text":
                action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
                action_prompt = self.act_template.format(action_prompt=action_str)
            elif self.actions_format == "continuous":
                action_continuous = action_tokens
            elif self.actions_format == "fast":
                if isinstance(action_tokens, list):
                    tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                    # Concatenate tensors along the first dimension
                    action_tokens = torch.cat(tensor_list, dim=0)
                action_ids = self.action_tokenizer(action_tokens)[0]
                # action_decode = self.action_tokenizer.decode([action_ids])
                self.last_vocab_idx = self.tokenizer.pad_token_id - 1
                action_ids = [self.last_vocab_idx - id for id in action_ids]
            else:
                raise ValueError(f"Invalid actions_format: {self.actions_format}")
        
        # video VLA
        if self.video_format == "interleave":
            text_prompt = self.tokenizer.bos_token + prompt
            image_tokens = image_tokens[0::self.action_frames,...]
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0::self.action_frames,...]
            
            sample_text = self.tokenizer(text_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
            sample_input_ids = sample_text["input_ids"][0]
            sample_attention_mask = sample_text["attention_mask"][0]

            labels = torch.full((self.tokenizer.model_max_length,), fill_value=-100, dtype=torch.long)
            for i in range(len(image_tokens)):
                image_prompt = self.format_video_prompt(image_tokens[i:i+1])
                if self.use_gripper:
                    gripper_prompt = self.format_video_prompt(gripper_tokens[i:i+1])
                    image_prompt += gripper_prompt
                sample_img = self.tokenizer(image_prompt, padding=False, return_token_type_ids=False, return_tensors="pt")
                image_input_ids = sample_img["input_ids"][0]
                image_attention_mask = sample_img["attention_mask"][0]
                if self.actions:
                    if self.actions_format == "fast":
                        action_sample = self.wrap_action_sequence(action_ids[i].tolist()) 
                        sample_input_ids = torch.cat([sample_input_ids, image_input_ids, action_sample], dim=-1)  
                        sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask, torch.ones_like(action_sample, dtype=torch.long)], dim=-1) 
                        action_start = len(sample_input_ids) - len(action_sample)
                        action_end = len(sample_input_ids)
                        if self.args.apply_loss_on_only_action:  
                            labels[action_start:action_end] = action_sample
                        else:  # Otherwise, fill both vision and action parts in the labels
                            labels[action_start-len(image_input_ids):action_start] = image_input_ids  
                            labels[action_start:action_end] = action_sample 
                else:
                    sample_input_ids = torch.cat([sample_input_ids, image_input_ids], dim=-1)
                    sample_attention_mask = torch.cat([sample_attention_mask, image_attention_mask], dim=-1)
                    labels[len(sample_input_ids)-len(image_input_ids):len(sample_input_ids)] = image_input_ids
            
            sample = self.tokenizer.pad(
                    {
                        "input_ids": sample_input_ids,
                        "attention_mask": sample_attention_mask,
                        "labels": labels
                    },
                    padding="max_length",
                    return_tensors="pt"
                )
            for k, v in sample.items():
                sample[k] = v.squeeze(0)
                
        # VLA Baseline (Img)
        else:
            image_tokens = image_tokens[0:self.T,...]
            image_prompt = self.format_video_prompt(image_tokens)
            if self.use_gripper:
                gripper_tokens = gripper_tokens[0:self.T,...]
                gripper_prompt = self.format_video_prompt(gripper_tokens)
                image_prompt += gripper_prompt
                
            prompt = f"Given the image of the current state, what actions should the robot take to {prompt}? Output the low-level action(s) to take."
            text_prompt = self.tokenizer(
                self.tokenizer.bos_token + prompt,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            sample_input_ids = text_prompt["input_ids"][0]
            sample_attention_mask = text_prompt["attention_mask"][0]
            labels = torch.full_like(sample_input_ids, self.args.ignore_index)

            def append_segment(segment, supervise=True):
                nonlocal sample_input_ids, sample_attention_mask, labels
                seg_ids = segment["input_ids"][0]
                seg_mask = segment["attention_mask"][0]
                sample_input_ids = torch.cat([sample_input_ids, seg_ids], dim=-1)
                sample_attention_mask = torch.cat([sample_attention_mask, seg_mask], dim=-1)
                if supervise:
                    seg_labels = seg_ids.clone()
                else:
                    seg_labels = torch.full_like(seg_ids, self.args.ignore_index)
                labels = torch.cat([labels, seg_labels], dim=-1)

            obs_prompt = self.tokenizer(
                image_prompt,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            append_segment(obs_prompt, supervise=False)
            # reason_text = reason_entry.get('reasoning', '').strip()
            reason_text = "To complete the task, we can get to the next state like this: "
            reason_prompt = self.tokenizer(
                self.tokenizer.bot_token + reason_text,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            append_segment(reason_prompt, supervise=True)

            goal_paths = reason_entry.get("goal_tokens", [])
            if goal_paths:
                goal_tokens = self.load_token_clip(goal_paths)
                if goal_tokens.dim() == 2:
                    goal_tokens = goal_tokens.unsqueeze(0)
                goal_prompt = self.tokenizer(
                    self.format_video_prompt(goal_tokens),
                    padding=False,
                    return_token_type_ids=False,
                    return_tensors="pt",
                )
                append_segment(goal_prompt, supervise=True)

            end_prompt = self.tokenizer(
                self.tokenizer.eot_token,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            append_segment(end_prompt, supervise=True)

            sample = {
                "input_ids": sample_input_ids,
                "attention_mask": sample_attention_mask,
                "labels": labels,
            }

            # based on the actions_format, append the action information to the sample
            if self.actions:
                if self.args.apply_loss_on_only_action and not getattr(self.args, "with_cot", False):
                    sample['labels'] = torch.full_like(sample['labels'], self.args.ignore_index)
                sample = self.append_action_to_sample(sample, action_ids)
            # finally, do padding
            sample = self.tokenizer.pad(
                sample,
                padding="max_length",
                return_tensors="pt"
            )

            for k, v in sample.items():
                sample[k] = v.squeeze(0)

            if "labels" in sample:
                sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        # with open("cot_debug/debug_training_cot_vla_prompt.txt", "w") as f:
        #     out = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        #     print(out, file=f)
        #     exit(0)
        return sample

class Emu3PerspectiveDataset(Emu3SFTDataset):

    def __init__(self, args: "DataArguments", tokenizer):
        super().__init__(args, tokenizer=tokenizer)
        self.perspective_image_key = getattr(args, "perspective_image_key", "gripper_image")
        self.perspective_use_vanilla_prefix = getattr(args, "perspective_use_vanilla_prefix", False)

    def _perspective_view_name(self):
        view_name = self.perspective_image_key
        if view_name.endswith("_image"):
            view_name = view_name[: -len("_image")]
        return view_name.replace("_", " ")

    def load_token_clip(self, paths):
        frames = [torch.from_numpy(np.load(path)) for path in paths]
        if not frames:
            print(paths)
            raise ValueError("No token paths provided for clip loading.")
        tensor = torch.stack(frames, dim=1)
        return tensor.squeeze(0)

    def __getitem__(self, index: int):
        scene = self.data[index]
        prompt = scene["text"]
        image_tokens_path = scene["image"]
        if self.perspective_image_key not in scene:
            available_views = sorted(
                key for key, value in scene.items()
                if key.endswith("_image") and key != "image" and isinstance(value, list)
            )
            raise KeyError(
                f"Perspective view '{self.perspective_image_key}' not found in sample. "
                f"Available views: {available_views}"
            )
        other_view_paths = scene[self.perspective_image_key]
        action = scene["action"]

        if self.T > 1 and self.video_format == "interleave":
            raise NotImplementedError("Perspective dataset does not support interleave video format yet.")

        available_frames = min(len(image_tokens_path), len(other_view_paths), len(action))
        if available_frames <= 0:
            raise ValueError(
                f"No aligned frames available for perspective training. "
                f"image={len(image_tokens_path)}, {self.perspective_image_key}={len(other_view_paths)}, action={len(action)}"
            )
        frames_num = self.action_frames if available_frames >= self.action_frames else available_frames

        if self.random_frame_sampling:
            max_start = max(available_frames - frames_num, 0)
            start_idx = random.randint(0, max_start)
        else:
            start_idx = 0
        end_idx = start_idx + frames_num

        selected_image_paths = image_tokens_path[start_idx:end_idx]
        selected_other_paths = other_view_paths[start_idx:end_idx]

        image_tokens = self.load_token_clip(selected_image_paths)
        other_view_tokens = self.load_token_clip(selected_other_paths)

        action_tokens = action[start_idx:end_idx]

        if self.actions_format == "openvla":
            action_tokens = action_tokens.flatten()
            action_ids = self.action_tokenizer(action_tokens)
        elif self.actions_format == "text":
            action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
            action_prompt = self.act_template.format(action_prompt=action_str)
        elif self.actions_format == "continuous":
            action_continuous = action_tokens
        elif self.actions_format == "fast":
            if isinstance(action_tokens, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                action_tokens = torch.cat(tensor_list, dim=0)
            action_ids = self.action_tokenizer(action_tokens)[0]
            self.last_vocab_idx = self.tokenizer.pad_token_id - 1
            action_ids = [self.last_vocab_idx - id for id in action_ids]
        else:
            raise ValueError(f"Invalid actions_format: {self.actions_format}")

        image_tokens = image_tokens[0:self.T, ...]
        image_prompt = self.format_video_prompt(image_tokens)

        if self.perspective_use_vanilla_prefix:
            prefix_prompt = self.tokenizer(
                self.tokenizer.bos_token + prompt + image_prompt,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            sample_input_ids = prefix_prompt["input_ids"][0]
            sample_attention_mask = prefix_prompt["attention_mask"][0]
            labels = torch.full_like(sample_input_ids, self.args.ignore_index)
        else:
            prompt = (
                f"Given the image of the current state, what actions should the robot take to {prompt}? "
                "Output the low-level action(s) to take."
            )
            text_prompt = self.tokenizer(
                self.tokenizer.bos_token + prompt,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            sample_input_ids = text_prompt["input_ids"][0]
            sample_attention_mask = text_prompt["attention_mask"][0]
            labels = torch.full_like(sample_input_ids, self.args.ignore_index)

        def append_segment(segment, supervise=True):
            nonlocal sample_input_ids, sample_attention_mask, labels
            seg_ids = segment["input_ids"][0]
            seg_mask = segment["attention_mask"][0]
            sample_input_ids = torch.cat([sample_input_ids, seg_ids], dim=-1)
            sample_attention_mask = torch.cat([sample_attention_mask, seg_mask], dim=-1)
            if supervise:
                seg_labels = seg_ids.clone()
            else:
                seg_labels = torch.full_like(seg_ids, self.args.ignore_index)
            labels = torch.cat([labels, seg_labels], dim=-1)

        if not self.perspective_use_vanilla_prefix:
            obs_prompt = self.tokenizer(
                image_prompt,
                padding=False,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            append_segment(obs_prompt, supervise=False)

        view_name = self._perspective_view_name()
        # perspective_text = f"The current state from {view_name} is: "
        perspective_text = ""
        perspective_prompt = self.tokenizer(
            self.tokenizer.bot_token + perspective_text,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        append_segment(perspective_prompt, supervise=True)

        other_view_tokens = other_view_tokens[0:self.T, ...]
        other_view_prompt = self.tokenizer(
            self.format_video_prompt(other_view_tokens),
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        append_segment(other_view_prompt, supervise=True)

        end_prompt = self.tokenizer(
            self.tokenizer.eot_token,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        append_segment(end_prompt, supervise=True)

        sample = {
            "input_ids": sample_input_ids,
            "attention_mask": sample_attention_mask,
            "labels": labels,
        }

        if self.actions:
            if self.args.apply_loss_on_only_action and not getattr(self.args, "with_perspective", False):
                sample["labels"] = torch.full_like(sample["labels"], self.args.ignore_index)
            sample = self.append_action_to_sample(sample, action_ids)

        sample = self.tokenizer.pad(
            sample,
            padding="max_length",
            return_tensors="pt",
        )

        for k, v in sample.items():
            sample[k] = v.squeeze(0)

        if "labels" in sample:
            sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        # with open("cot_debug/train_prompt.txt", "w") as f:
        #     out = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        #     print(out, file=f)
        #     exit(0)
        return sample

class Emu3MultiViewDataset(Emu3SFTDataset):

    def __init__(self, args: "DataArguments", tokenizer):
        super().__init__(args, tokenizer=tokenizer)

    def load_token_clip(self, paths):
        frames = [torch.from_numpy(np.load(path)) for path in paths]
        if not frames:
            raise ValueError("No token paths provided for clip loading.")
        tensor = torch.stack(frames, dim=1)
        return tensor.squeeze(0)

    def _describe_view(self, view_name):
        normalized = str(view_name).replace("_", " ").strip()
        if not normalized:
            return "unknown view"
        if "view" in normalized:
            return normalized
        return f"{normalized} view"

    def __getitem__(self, index: int):
        scene = self.data[index]
        task_text = scene["text"].strip()
        source_view_paths = scene["image"]
        target_view_paths = scene["target_image"]
        action = scene["action"]
        source_view = self._describe_view(scene.get("source_view", "agent"))
        target_view = self._describe_view(scene["target_view"])

        if self.T > 1 and self.video_format == "interleave":
            raise NotImplementedError("Multi-view dataset does not support interleave video format yet.")

        available_frames = min(len(source_view_paths), len(target_view_paths), len(action))
        if available_frames <= 0:
            raise ValueError(
                "No aligned frames available for multi-view training. "
                f"image={len(source_view_paths)}, target_image={len(target_view_paths)}, action={len(action)}"
            )
        frames_num = self.action_frames if available_frames >= self.action_frames else available_frames

        if self.random_frame_sampling:
            max_start = max(available_frames - frames_num, 0)
            start_idx = random.randint(0, max_start)
        else:
            start_idx = 0
        end_idx = start_idx + frames_num

        source_tokens = self.load_token_clip(source_view_paths[start_idx:end_idx])
        target_tokens = self.load_token_clip(target_view_paths[start_idx:end_idx])
        action_tokens = action[start_idx:end_idx]

        if self.actions_format == "openvla":
            action_tokens = action_tokens.flatten()
            action_ids = self.action_tokenizer(action_tokens)
        elif self.actions_format == "text":
            action_str = "\n".join(",".join(f"{num:.2f}" for num in row) for row in action_tokens)
            action_prompt = self.act_template.format(action_prompt=action_str)
        elif self.actions_format == "continuous":
            action_continuous = action_tokens
        elif self.actions_format == "fast":
            if isinstance(action_tokens, list):
                tensor_list = [torch.tensor(item).unsqueeze(0) for item in action_tokens]
                action_tokens = torch.cat(tensor_list, dim=0)
            action_ids = self.action_tokenizer(action_tokens)[0]
            self.last_vocab_idx = self.tokenizer.pad_token_id - 1
            action_ids = [self.last_vocab_idx - id for id in action_ids]
        else:
            raise ValueError(f"Invalid actions_format: {self.actions_format}")

        source_tokens = source_tokens[0:self.T, ...]
        target_tokens = target_tokens[0:self.T, ...]
        source_prompt = self.format_video_prompt(source_tokens)
        target_prompt = self.format_video_prompt(target_tokens)

        prompt = (
            f"Given the image of the current state from the {source_view}, generate the current state from the "
            f"{target_view}, then output the low-level action(s) the robot should take to {task_text}."
        )
        text_prompt = self.tokenizer(
            self.tokenizer.bos_token + prompt,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        sample_input_ids = text_prompt["input_ids"][0]
        sample_attention_mask = text_prompt["attention_mask"][0]
        labels = torch.full_like(sample_input_ids, self.args.ignore_index)

        def append_segment(segment, supervise=True):
            nonlocal sample_input_ids, sample_attention_mask, labels
            seg_ids = segment["input_ids"][0]
            seg_mask = segment["attention_mask"][0]
            sample_input_ids = torch.cat([sample_input_ids, seg_ids], dim=-1)
            sample_attention_mask = torch.cat([sample_attention_mask, seg_mask], dim=-1)
            if supervise:
                seg_labels = seg_ids.clone()
            else:
                seg_labels = torch.full_like(seg_ids, self.args.ignore_index)
            labels = torch.cat([labels, seg_labels], dim=-1)

        obs_prompt = self.tokenizer(
            source_prompt,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        append_segment(obs_prompt, supervise=False)

        target_prefix = self.tokenizer(
            self.tokenizer.bot_token,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        append_segment(target_prefix, supervise=True)

        target_prompt_tokens = self.tokenizer(
            target_prompt,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        append_segment(target_prompt_tokens, supervise=True)

        end_prompt = self.tokenizer(
            self.tokenizer.eot_token,
            padding=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        append_segment(end_prompt, supervise=True)

        sample = {
            "input_ids": sample_input_ids,
            "attention_mask": sample_attention_mask,
            "labels": labels,
        }

        if self.actions:
            if self.args.apply_loss_on_only_action and not getattr(self.args, "with_multiview", False):
                sample["labels"] = torch.full_like(sample["labels"], self.args.ignore_index)
            sample = self.append_action_to_sample(sample, action_ids)

        sample = self.tokenizer.pad(
            sample,
            padding="max_length",
            return_tensors="pt",
        )

        for k, v in sample.items():
            sample[k] = v.squeeze(0)

        if "labels" in sample:
            sample["labels"] = self.pad_tensor(sample["labels"], self.tokenizer.model_max_length, self.args.ignore_index)
        return sample
