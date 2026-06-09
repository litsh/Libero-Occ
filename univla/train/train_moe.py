import os
import os.path as osp
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List
import pathlib
import transformers as tf
from datasets import Emu3SFTDataset
import sys
UNIVLA_ROOT = pathlib.Path(__file__).resolve().parents[1]
EMU3_ROOT = UNIVLA_ROOT / "reference" / "Emu3"
sys.path.insert(0, str(EMU3_ROOT))
from emu3.mllm import Emu3Config, Emu3Tokenizer, Emu3ForCausalLM, Emu3MoE, Emu3MoEConfig
from transformers import AutoModel,Trainer
from datasets import Emu3WorldModelDataset,Emu3RealRobotDataset,Emu3CoTDataset,Emu3PerspectiveDataset,Emu3MultiViewDataset
from torch.utils.data import WeightedRandomSampler, DataLoader
import torch
import numpy
# Allowlist the specific numpy functions that are causing the crash
torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype, numpy.dtypes.UInt32DType])
class TokenLossLoggingTrainer(Trainer):
    def __init__(
        self,
        *args,
        visual_token_range=None,
        action_token_range=None,
        reasoning_phrase_ids_list=None,
        bot_token_id=None,
        eot_token_id=None,
        eoi_token_id=None,
        boa_token_id=None,
        eoa_token_id=None,
        perspective_strict: bool = False,
        ignore_index: int = -100,
        use_group_loss_weighting: bool = False,
        group_loss_mode: str = "four_group",
        visual_content_loss_weight: float = 1.0,
        visual_special_loss_weight: float = 1.0,
        action_content_loss_weight: float = 1.0,
        action_special_loss_weight: float = 1.0,
        visual_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.visual_token_range = visual_token_range
        self.action_token_range = action_token_range
        self.reasoning_phrase_ids_list = reasoning_phrase_ids_list or []
        self.bot_token_id = bot_token_id
        self.eot_token_id = eot_token_id
        self.eoi_token_id = eoi_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.perspective_strict = perspective_strict
        self.ignore_index = ignore_index
        self.use_group_loss_weighting = use_group_loss_weighting
        if group_loss_mode not in {"four_group", "two_group"}:
            raise ValueError(
                f"Unsupported group_loss_mode: {group_loss_mode}. "
                "Expected one of {'four_group', 'two_group'}."
            )
        self.group_loss_mode = group_loss_mode
        self.visual_content_loss_weight = visual_content_loss_weight
        self.visual_special_loss_weight = visual_special_loss_weight
        self.action_content_loss_weight = action_content_loss_weight
        self.action_special_loss_weight = action_special_loss_weight
        self.visual_loss_weight = visual_loss_weight
        self.action_loss_weight = action_loss_weight
        self._last_token_losses = {}

    def _masked_mean(self, loss_per_token, mask):
        count = mask.sum()
        if count.item() == 0:
            return torch.tensor(0.0, device=loss_per_token.device), count
        loss = (loss_per_token * mask).sum() / count
        return loss, count

    def _id_mask(self, labels, ids):
        if not ids:
            return torch.zeros_like(labels, dtype=torch.bool)
        mask = labels.eq(ids[0])
        for token_id in ids[1:]:
            mask = mask | labels.eq(token_id)
        return mask

    def _find_first_subsequence(self, seq, subseq):
        if not subseq:
            return None
        max_start = len(seq) - len(subseq) + 1
        if max_start < 1:
            return None
        for i in range(max_start):
            if seq[i:i + len(subseq)] == subseq:
                return i
        return None

    def _find_phrase_start(self, seq):
        if self.bot_token_id is None:
            return None
        best_start = None
        for phrase_ids in self.reasoning_phrase_ids_list:
            combined = [self.bot_token_id] + phrase_ids
            start = self._find_first_subsequence(seq, combined)
            if start is not None and (best_start is None or start < best_start):
                best_start = start
        if best_start is not None:
            return best_start
        # Fallback: allow empty or changed prompt by anchoring on the first bot token.
        try:
            return seq.index(self.bot_token_id)
        except ValueError:
            if self.perspective_strict:
                raise ValueError("bot_token not found while perspective_strict is enabled")
            return None

    def _build_visual_with_special_mask(self, shift_input_ids):
        mask = torch.zeros_like(shift_input_ids, dtype=torch.bool)
        batch_size, seq_len = shift_input_ids.shape
        for b in range(batch_size):
            seq = shift_input_ids[b].tolist()
            start = self._find_phrase_start(seq)
            if start is None:
                # No visual-with-special span found; keep mask empty for this sample.
                continue
            end = None
            if self.eot_token_id is not None:
                try:
                    end = seq.index(self.eot_token_id, start)
                except ValueError:
                    end = None
            if end is None:
                # No end token; leave mask empty for this sample.
                raise ValueError("eot_token not found.")
            mask[b, start:end + 1] = True
        return mask

    def _build_action_with_special_mask(self, shift_input_ids):
        mask = torch.zeros_like(shift_input_ids, dtype=torch.bool)
        batch_size, seq_len = shift_input_ids.shape
        for b in range(batch_size):
            seq = shift_input_ids[b].tolist()
            start = None
            if self.boa_token_id is not None:
                try:
                    start = seq.index(self.boa_token_id)
                except ValueError:
                    start = None
            if start is None:
                raise ValueError(f"action_with_special start not found for sample {b}")
            end = None
            if self.eoa_token_id is not None:
                try:
                    end = seq.index(self.eoa_token_id, start)
                except ValueError:
                    end = None
            if end is None:
                raise ValueError(f"action_with_special end not found for sample {b}")
            mask[b, start:end + 1] = True
        return mask

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        self._last_token_losses = {}
        if labels is not None and hasattr(outputs, "logits"):
            if self.visual_token_range or self.action_token_range:
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                vocab_size = shift_logits.size(-1)
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    reduction="none",
                    ignore_index=self.ignore_index,
                ).view(shift_labels.size())

                valid_mask = shift_labels.ne(self.ignore_index)
                visual_content_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                action_content_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                if self.visual_token_range:
                    vmin, vmax = self.visual_token_range
                    visual_content_mask = valid_mask & shift_labels.ge(vmin) & shift_labels.le(vmax)
                if self.action_token_range:
                    amin, amax = self.action_token_range
                    action_content_mask = valid_mask & shift_labels.ge(amin) & shift_labels.le(amax)

                input_ids = inputs.get("input_ids")
                attention_mask = inputs.get("attention_mask")
                visual_with_special_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                action_with_special_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                visual_special_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                action_special_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
                if input_ids is not None and attention_mask is not None:
                    shift_input_ids = input_ids[..., 1:].contiguous()
                    shift_attention_mask = attention_mask[..., 1:].contiguous().bool()

                    visual_with_special_mask = self._build_visual_with_special_mask(shift_input_ids)
                    visual_with_special_mask = visual_with_special_mask & shift_attention_mask & valid_mask
                    visual_special_mask = visual_with_special_mask & (~visual_content_mask)

                    action_with_special_mask = self._build_action_with_special_mask(shift_input_ids)
                    action_with_special_mask = action_with_special_mask & shift_attention_mask & valid_mask
                    action_special_mask = action_with_special_mask & (~action_content_mask)

                if self.group_loss_mode == "four_group":
                    group_losses = {}
                    visual_content_loss, _ = self._masked_mean(loss_per_token, visual_content_mask)
                    action_content_loss, _ = self._masked_mean(loss_per_token, action_content_mask)
                    visual_special_loss, _ = self._masked_mean(loss_per_token, visual_special_mask)
                    action_special_loss, _ = self._masked_mean(loss_per_token, action_special_mask)

                    group_losses["loss/visual_content"] = visual_content_loss
                    group_losses["loss/action_content"] = action_content_loss
                    group_losses["loss/visual_special"] = visual_special_loss
                    group_losses["loss/action_special"] = action_special_loss
                    group_losses["loss/groups_unweighted_total"] = (
                        visual_content_loss
                        + action_content_loss
                        + visual_special_loss
                        + action_special_loss
                    )

                    self._last_token_losses = {
                        key: value.detach() for key, value in group_losses.items()
                    }

                    if self.use_group_loss_weighting:
                        weighted_loss = (
                            self.visual_content_loss_weight * visual_content_loss
                            + self.visual_special_loss_weight * visual_special_loss
                            + self.action_content_loss_weight * action_content_loss
                            + self.action_special_loss_weight * action_special_loss
                        )
                        self._last_token_losses["loss/groups_weighted_total"] = weighted_loss.detach()
                        loss = weighted_loss
                else:
                    group_losses = {}
                    visual_mask = visual_content_mask | visual_special_mask
                    action_mask = action_content_mask | action_special_mask
                    visual_loss, _ = self._masked_mean(loss_per_token, visual_mask)
                    action_loss, _ = self._masked_mean(loss_per_token, action_mask)

                    group_losses["loss/visual"] = visual_loss
                    group_losses["loss/action"] = action_loss
                    group_losses["loss/groups_unweighted_total"] = (
                        visual_loss + action_loss
                    )

                    self._last_token_losses = {
                        key: value.detach() for key, value in group_losses.items()
                    }

                    if self.use_group_loss_weighting:
                        weighted_loss = (
                            self.visual_loss_weight * visual_loss
                            + self.action_loss_weight * action_loss
                        )
                        self._last_token_losses["loss/groups_weighted_total"] = weighted_loss.detach()
                        loss = weighted_loss

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs):
        if self._last_token_losses:
            logs = dict(logs)
            for key, value in self._last_token_losses.items():
                logs[key] = float(value)
        return super().log(logs)

class WeightedSamplerTrainer(TokenLossLoggingTrainer):
    def get_train_dataloader(self):
        # Assuming train_dataset has a sample_weights attribute
        sample_weights = torch.tensor(
            self.train_dataset.sample_weights, dtype=torch.double
        )

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    model_config_path: Optional[str] = field(default="pretrain/Emu3-Base")

@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    null_prompt_prob: float = field(default=0.05)
    apply_loss_on_only_vision: bool = field(default=True)
    apply_loss_on_only_text: bool = field(default=False)
    apply_loss_on_only_action: bool = field(default=False) 
    ignore_index: int = field(default=-100)
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")
    codebook_size: Optional[int] = field(default=32768)
    frames: int = field(default=4)
    VL: bool = field(default=False)
    actions: bool = field(default=False)
    actions_format: str = field(default="openvla")
    action_frames: int = field(default=8)
    use_gripper: bool = field(default=False)
    action_tokenizer_path: Optional[str] = field(default=None)
    video_format: str = field(default=None)
    random_frame_sampling: bool = field(default=True)
    raw_image: bool = field(default=False)
    post_training: bool = field(default=False)
    datasets_weight: bool = field(default=False)
    without_text: bool = field(default=False)
    real_robot: bool = field(default=False)
    with_cot: bool = field(default=False)
    with_perspective: bool = field(default=False)
    with_multiview: bool = field(default=False)
    perspective_image_key: str = field(default="gripper_image")
    perspective_use_vanilla_prefix: bool = field(default=False)
    use_group_loss_weighting: bool = field(default=False)
    group_loss_mode: str = field(default="four_group")
    visual_content_loss_weight: float = field(default=1.0)
    visual_special_loss_weight: float = field(default=1.0)
    action_content_loss_weight: float = field(default=1.0)
    action_special_loss_weight: float = field(default=1.0)
    visual_loss_weight: float = field(default=1.0)
    action_loss_weight: float = field(default=1.0)

@dataclass
class TrainingArguments(tf.TrainingArguments):
    report_to: List[str] = field(default_factory=list)
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)
    from_scratch: bool = field(default=False)
    dataloader_num_workers: Optional[int] = field(default=0)

def load_model(model_args, model_config, training_args):
    """
    Load model based on whether to train from scratch or fine-tune from a pre-trained model.
    """
    if training_args.from_scratch:
        model_config.torch_dtype = torch.bfloat16 if training_args.bf16 else None
        model_config.attn_implementation = "flash_attention_2" if training_args.attn_type == "fa2" else None
        return Emu3MoE(config=model_config)
    else:
        return Emu3MoE.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else None,
            torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        )

def get_dataset(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        return Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
        # return Emu3SFTDataset(data_args, tokenizer=tokenizer)
    elif data_args.real_robot:
        return Emu3RealRobotDataset(data_args, tokenizer=tokenizer)
    elif data_args.with_cot:
        return Emu3CoTDataset(data_args, tokenizer=tokenizer)
    elif data_args.with_multiview:
        return Emu3MultiViewDataset(data_args, tokenizer=tokenizer)
    elif data_args.with_perspective:
        return Emu3PerspectiveDataset(data_args, tokenizer=tokenizer)
    return Emu3SFTDataset(data_args, tokenizer=tokenizer)

def get_dataset_split(data_args, tokenizer):
    """
    Initialize and return the training dataset.
    """
    if data_args.post_training:
        full_dataset = Emu3WorldModelDataset(data_args, tokenizer=tokenizer)
    elif data_args.with_multiview:
        full_dataset = Emu3MultiViewDataset(data_args, tokenizer=tokenizer)
    else:
        full_dataset = Emu3SFTDataset(data_args, tokenizer=tokenizer)
    # 自动划分 90% train, 10% val
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    return split["train"], split["test"]

def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

def get_token_ranges(data_args, tokenizer, train_dataset):
    bov = tokenizer.encode(data_args.visual_token_pattern.format(token_id=0))[0]
    eov = tokenizer.encode(data_args.visual_token_pattern.format(token_id=data_args.codebook_size - 1))[0]
    visual_range = (bov, eov)
    reasoning_phrases = [
        "To complete the task, we can get to the next state like this: ",
        ""
    ]
    reasoning_phrase_ids_list = [
        tokenizer.encode(p, add_special_tokens=False) for p in reasoning_phrases
    ]
    reasoning_phrase_ids_list = [p for p in reasoning_phrase_ids_list if p]
    bot_token_id = tokenizer.encode(tokenizer.bot_token)[0]
    eot_token_id = tokenizer.encode(tokenizer.eot_token)[0]
    eoi_token_id = tokenizer.encode(tokenizer.eoi_token)[0]

    action_range = None
    boa_token_id = tokenizer.encode(tokenizer.boa_token)[0]
    eoa_token_id = tokenizer.encode(tokenizer.eoa_token)[0]
    if getattr(data_args, "actions", False) and hasattr(train_dataset, "action_tokenizer"):
        last_vocab_idx = tokenizer.pad_token_id - 1
        action_tokenizer = train_dataset.action_tokenizer
        if hasattr(action_tokenizer, "action_token_begin_idx"):
            action_min = action_tokenizer.action_token_begin_idx + 1
            action_max = last_vocab_idx - 1
            action_range = (action_min, action_max)
        elif hasattr(action_tokenizer, "vocab_size"):
            action_min = last_vocab_idx - (action_tokenizer.vocab_size - 1)
            action_max = last_vocab_idx
            action_range = (action_min, action_max)

    return (
        visual_range,
        reasoning_phrase_ids_list,
        bot_token_id,
        eot_token_id,
        eoi_token_id,
        action_range,
        boa_token_id,
        eoa_token_id,
    )

def train():
    """
    Main function to train the model.
    """
    # Parse arguments
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set environment variable for WANDB logging
    os.environ["WANDB_DIR"] = osp.join(training_args.output_dir, "wandb")

    # Load model configuration and tokenizer
    model_config = Emu3MoEConfig.from_pretrained(model_args.model_config_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False,
    )

    # Initialize model
    model = load_model(model_args, model_config, training_args)

    # Initialize dataset
    train_dataset = get_dataset(data_args, tokenizer)
    (
        visual_range,
        reasoning_phrase_ids_list,
        bot_token_id,
        eot_token_id,
        eoi_token_id,
        action_range,
        boa_token_id,
        eoa_token_id,
    ) = get_token_ranges(
        data_args, tokenizer, train_dataset
    )

    if data_args.datasets_weight:
        trainer = WeightedSamplerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset, 
            tokenizer=tokenizer,
            visual_token_range=visual_range,
            action_token_range=action_range,
            reasoning_phrase_ids_list=reasoning_phrase_ids_list,
            bot_token_id=bot_token_id,
            eot_token_id=eot_token_id,
            eoi_token_id=eoi_token_id,
            boa_token_id=boa_token_id,
            eoa_token_id=eoa_token_id,
            perspective_strict=(data_args.with_perspective or data_args.with_multiview),
            ignore_index=data_args.ignore_index,
            use_group_loss_weighting=data_args.use_group_loss_weighting,
            group_loss_mode=data_args.group_loss_mode,
            visual_content_loss_weight=data_args.visual_content_loss_weight,
            visual_special_loss_weight=data_args.visual_special_loss_weight,
            action_content_loss_weight=data_args.action_content_loss_weight,
            action_special_loss_weight=data_args.action_special_loss_weight,
            visual_loss_weight=data_args.visual_loss_weight,
            action_loss_weight=data_args.action_loss_weight,
        )
    else:
        # Setup Trainer
        trainer = TokenLossLoggingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,  # Pass tokenizer to trainer
            visual_token_range=visual_range,
            action_token_range=action_range,
            reasoning_phrase_ids_list=reasoning_phrase_ids_list,
            bot_token_id=bot_token_id,
            eot_token_id=eot_token_id,
            eoi_token_id=eoi_token_id,
            boa_token_id=boa_token_id,
            eoa_token_id=eoa_token_id,
            perspective_strict=(data_args.with_perspective or data_args.with_multiview),
            ignore_index=data_args.ignore_index,
            use_group_loss_weighting=data_args.use_group_loss_weighting,
            group_loss_mode=data_args.group_loss_mode,
            visual_content_loss_weight=data_args.visual_content_loss_weight,
            visual_special_loss_weight=data_args.visual_special_loss_weight,
            action_content_loss_weight=data_args.action_content_loss_weight,
            action_special_loss_weight=data_args.action_special_loss_weight,
            visual_loss_weight=data_args.visual_loss_weight,
            action_loss_weight=data_args.action_loss_weight,
        )

    # Check if resuming from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model and training state
    trainer.save_state()
    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    train()
