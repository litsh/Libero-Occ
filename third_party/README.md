# Third-Party Code

This repository vendors the UniVLA-derived code needed for VIM training/evaluation under `univla/`.

Install upstream LIBERO separately:

- LIBERO: simulator and benchmark environment.

The bundled VIM code is developed from UniVLA and includes these paths:

```text
train/train_moe.py
configs/moe_fast_video.json
scripts/sft/zero3_H200.json
reference/RoboVLMs/eval/libero/evaluate_libero_emu_multi_gpu.py
```

Please also follow the license and citation terms of upstream UniVLA, Emu3, RoboVLMs, and LIBERO.
