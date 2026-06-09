# UniVLA-Derived VIM Code

This directory contains the UniVLA-derived code needed for LIBERO-Occ VIM training and evaluation.

Included:

- `train/train_moe.py` and `train/datasets.py`
- VIM-required model utilities under `models/`
- Emu3 model code under `reference/Emu3/`
- LIBERO evaluation code under `reference/RoboVLMs/eval/libero/`
- model and DeepSpeed configs under `configs/` and `scripts/sft/`

Not included:

- model checkpoints,
- datasets or metadata pickle files,
- image-token caches,
- FAST action tokenizer checkpoint,
- experiment logs.

Use the top-level scripts in `../scripts/` to launch training and evaluation.

