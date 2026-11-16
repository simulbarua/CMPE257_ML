# Experiment record: ADRec on Beauty

- Model: ADRec, full mode.
- Completed seeds: 42-46. Complete five-seed set.
- Dataset SHA-256: `a2f6ca22b3efe2bc5030ef9a7c4f6c928db2086771d58d9a40147e660f574bc1`. Released train/validation/test splits retained.
- Training: separate embedding pretraining and ADRec training for each seed; the same seed is set in both stages.
- Training notebook: `notebooks/adrec_kaggle.ipynb`.
- Configuration: `adrec_seed<N>_config.json`; pretraining metadata is under `pretrain/`.
- Selection: improvement in any tracked validation metric; evaluation interval 5, patience 4.
- Evaluation: `adrec_upstream`, batch size 512, HR/NDCG percentages, all catalog columns including padding; no seen-item filtering.
- Aggregation: arithmetic mean and sample standard deviation (n-1) of the per-seed test percentages. A single run has no reported standard deviation.
- Source/runtime: per-seed environment files contain source hashes, Python, PyTorch, CUDA, and hardware. Per-seed requirements files record installed packages.
- Checkpoints: hashes and storage references are in `adrec_seed<N>_checkpoints.json`. Weights are stored separately from these result files.
- Timing: training and inference benchmarks were not recorded by this export.
- Deviations: project seed settings, resumable training, enforced embedding freeze during alignment, and evaluation-mode checkpoint inference are documented in `docs/KAGGLE_TRAINING.md`.

`environment.json` and `requirements-lock.txt` are copies from the lowest listed seed for compatibility with the baseline result layout. Per-seed files retain each run's own environment.
