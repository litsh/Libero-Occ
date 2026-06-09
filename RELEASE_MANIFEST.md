# Release Manifest

Publishable contents:

- `benchmark_assets/`: final LIBERO-Occ BDDL and init files.
- `scripts/setup/`: asset installation and validation scripts.
- `scripts/train/`: VIM Stage-1 and Stage-2 UniVLA launch scripts.
- `scripts/eval/`: LIBERO-Occ VIM evaluation launcher.
- `univla/`: UniVLA-derived VIM training/evaluation source code.
- `configs/`: reference training and evaluation settings.
- `docs/`: installation, benchmark, training, and evaluation notes.
- `README.md`, `LICENSE`, `CITATION.cff`, `.gitignore`.

Intentionally excluded:

- benchmark-generation code,
- benchmark-selection code,
- paper table/figure reproduction scripts,
- local logs, checkpoints, datasets, caches, and Hugging Face downloads,
- full upstream LIBERO source tree,
- unrelated UniVLA datasets, logs, checkpoints, caches, and auxiliary inference/preprocessing scripts.

Before publishing:

- replace `TODO` entries in `README.md` and `CITATION.cff`,
- initialize git outside the current sandbox if needed,
- copy or archive only the publishable contents listed above.
