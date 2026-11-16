# ADRec Baby warm-up ablation

New runs set warmup_epochs=0; historical baseline runs used five epochs. Both use 32 diffusion steps and the same per-seed pretrained embeddings. Only warmup_epochs changes in the full-run configuration. New ADRec backbones start fresh; pretraining is not repeated.

accuracy_runs.csv and accuracy_summary.csv compare the same completed seeds in both conditions. Summaries report mean and sample standard deviation. Partial batches are labeled in completion.json; missing seeds are not filled. paired_differences.csv reports warmup0 minus warmup5 in percentage points.

Baseline results are reused as recorded, not re-evaluated or retrained. The exact audited Baby source migration is recorded in source_migrations.json. Runtime and package versions may differ between historical and new runs. HR/NDCG use the existing upstream-style evaluator. Checkpoints are selected by validation with the same early-stopping settings. Our baseline strictly freezes embeddings during warm-up; this differs from the released optimizer's possible weight-decay updates. These limitations remain applicable to comparisons with the paper.

paper_reference.csv contains published Baby point estimates from Table 4, https://arxiv.org/html/2505.19544v1#S4.SS4. They are separate from project measurements and have no invented per-seed values or standard deviations. No inference-speed benchmark is part of this experiment.
