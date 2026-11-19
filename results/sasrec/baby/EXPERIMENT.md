# Experiment record: SASRec on Amazon Baby

- Owner / date: Anusha Subramanian; runs in November 2026
- Model / run ID: SASRec+ on Amazon Baby, full mode, 5 runs with seeds 42-46.
  Run names: anusha_sasrec_full_01 (seed 42), anusha_sasrec_full_seed43,
  anusha_sasrec_full_seed44, anusha_sasrec_full_seed45, anusha_sasrec_full_seed46.
- Team commit / upstream commit: Team Kaggle bundle base commit
  44bb5beb5aeadcc404410181886842de23d0056c (branch codex/kaggle-training-notebooks;
  working-copy snapshot, including local changes). SASRec code is vendored from the ADRec
  authors' repo (Nemo-1024/ADRec), commit 0a30ef764cd85efa44f884a387198e9bf28c7d4c,
  src/sasrec.py.
- Dataset checksum / split definition: dataset.pkl SHA-256
  bf8f3485f9e36763e07f0234e0ddb479cb023358797615388a36f0d7238cea99.
  11,761 users, 4,731 items. Each user's last item is held out for test and the
  second-to-last for validation (69,301 train / 11,761 val / 11,761 test interactions).
- Config file / random seed: configs/shared.json via make_config('sasrec', 'full'), with only
  the seed changed per run (42, 43, 44, 45, 46). Seed 42 is the shared-config default and was
  not set explicitly; seeds 43-46 were set with cfg['seed']. Per-run configs are in
  results/sasrec/baby/ (sasrec_seed42_config.json ... sasrec_seed46_config.json).
  Key settings: max_len 50, hidden size 128, learning rate 0.001, weight decay 1e-5,
  train batch size 512, evaluation batch size 512, up to 500 epochs.
- Python and package versions: Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 (from
  environment.json of one run; the other runs used the same Kaggle setup but were not
  individually checked). Full package list: requirements-lock.txt from the same run.
- Hardware / accelerator: Kaggle, NVIDIA Tesla T4 GPU (cuda:0).
- Training command or notebook: Kaggle notebook mahalakshmi_sasrec_kaggle.ipynb
  (SHA-256 36fe6e5a47c545209e40d4f03afdded7b21d54b8f60cf1c13d3892cbab9ec1d2), full mode,
  FINALIZE = True, no resume.
- Evaluation command: final evaluation cell of the same notebook, protocol "adrec_upstream",
  evaluation batch size 512.
- Training stages completed / pretrained weight source: Single stage, trained from scratch.
  No pretrained weights.
- Validation selection rule: Validation every 5 epochs (zero-based epoch index, so epochs
  5, 10, 15, ...). A checkpoint becomes best.pt when any tracked validation metric (HR or NDCG
  at 5, 10, 20) reaches a new best; training stops after 4 consecutive validation checks with
  no such improvement. Reported best_epoch and epochs_completed use the same zero-based
  numbering. The test split is evaluated once, on best.pt, at the end.
- Metrics and candidate-item rules: HR@5/10/20 and NDCG@5/10/20, reported in percent. Scores
  cover item IDs 0..4731 with no extra filtering of padding or previously seen items, matching
  the upstream metric path (see evaluation/PROTOCOL.md).
- Results / variability: Test, arithmetic mean ± sample standard deviation (n-1) of the
  per-seed test percentages over 5 seeds:
  HR@5 2.3872 ± 0.1046, HR@10 3.6180 ± 0.1738, HR@20 5.5902 ± 0.2868,
  NDCG@5 1.6331 ± 0.0509, NDCG@10 2.0279 ± 0.0568, NDCG@20 2.5217 ± 0.0766.
  Per-seed values are in results/sasrec/baby/sasrec_baby_runs.csv and the summary in
  results/sasrec/baby/sasrec_baby_summary.csv. Best epochs were 30-45; runs early-stopped at
  epochs 51-66.
- Training and inference timing methodology: Each full run took under 5 minutes (approximate;
  wall-clock time was not measured precisely). Inference timing was not measured.
- Checkpoint location and checksum: Weights are not committed to Git. The SHA-256
  hash of each best.pt, as downloaded from Kaggle, is listed here:
  - seed 42: E573DA8643851F425D5899042424B715DA60EBD5560933D043C19AB6FC66FF3E
  - seed 43: D52AFFE8B0C2B6047ACEF15D085DDFC3B725378FA8FA30BE01D23D09C8BC7254
  - seed 44: D31FFC463C06C813D9559CB2316A49E7BE3CF253AC323F74149293C7DF185C3B
  - seed 45: 3D21B57B7C348DBCC6043D9D66CFC8FFB4D26D9D0A6991887143F04361FFEE21
  - seed 46: A73EC4B46AEBE29052BFEEC5981CE9869C22158364D97F2D35FBCC952FE990B7
- Known limitations / deviations from paper:
  - Results are close to the paper's Table 2 for SASRec+ on Baby (HR@20 5.5268, NDCG@20 2.5197).
  - The interaction total here is 92,823 (69,301 + 11,761 + 11,761); the paper's Table 7 lists
    92,613. Users and items match.
  - The repo's loader holds out each user's last items, rather than using the paper's 8:1:1
    user split.
