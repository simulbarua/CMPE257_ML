# Experiment record: SASRec+ maximum sequence length on ml-100k

- **Owner / date:** Anusha Subramanian; runs in November 2026.
- **Model / run ID:** SASRec+, full mode. Maximum history lengths: [10, 25]. Seeds 42-46. Runs: `sasrec_full_len{10,25}_seed{42..46}`.
- **What changes:** the maximum sequence length, in BOTH training and testing (the model is retrained at each length; SASRec+'s learned position table is sized to max_len). This is not an evaluation-only truncation of a trained model and not a cold-start test. Everything else matches the shared settings.
- **Team commit / upstream commit:** Kaggle code bundle base commit `61de4e8e7b6f6e527e9664faecbb9e3e8458afcb`; upstream ADRec commit `0a30ef764cd85efa44f884a387198e9bf28c7d4c`.
- **Dataset checksum / split definition:** `ml-100k`, SHA-256 `dfc5dfaa7950d1a85af4fd6bb168c38efc8213092a9f72563c6d75ab00cef8d0`. 938 users, 1,008 items; train 52,537 / val 938 / test 938 interactions. Released per-user holdout split retained.
- **Config file / random seed:** `make_config('sasrec', 'full', 'ml-100k')` with `max_len` set to 10 or 25; per-run configs in `len10/` and `len25/`. Seeds 42-46. Hidden size 128, learning rate 0.001, weight decay 1e-5, batch size 512, up to 500 epochs, eval every 5 epochs, patience 4.
- **Python and package versions:** Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 (seed 42, length 10 `environment.json`).
- **Hardware / accelerator:** Kaggle, Tesla T4.
- **Training command or notebook:** `notebooks/sasrec_maximum_sequence_length_kaggle.ipynb` (full mode).
- **Evaluation command:** built into `train(..., finalize=True)`; `adrec_upstream` protocol, test evaluated once at the end.
- **Training stages completed / pretrained weight source:** single stage, trained from scratch at each length.
- **Validation selection rule:** Validation every 5 epochs; a checkpoint becomes best.pt when any tracked validation metric (HR or NDCG at 5, 10, 20) reaches a new best; training stops after 4 consecutive validation checks with no improvement (up to 500 epochs). Test is evaluated once, on best.pt, at the end. Best epochs, length 10 (seeds 42-46): 135, 85, 80, 100, 120; length 25: 85, 120, 110, 150, 105.
- **Metrics and candidate-item rules:** HR@5/10/20 and NDCG@5/10/20, in percent, protocol `adrec_upstream` (see `evaluation/PROTOCOL.md`): all item columns, no filtering of padding or previously seen items, evaluation batch size 512.
- **Results / variability:** mean ± sample standard deviation (n-1) over 5 seeds, percent:
  - max_len 10: HR@10 8.991 ± 1.2554, NDCG@10 4.3228 ± 0.6732, HR@20 15.7856 ± 1.203, NDCG@20 6.0174 ± 0.6168
  - max_len 25: HR@10 11.2321 ± 0.8933, NDCG@10 5.2091 ± 0.4803, HR@20 18.2024 ± 0.8969, NDCG@20 6.9458 ± 0.4409
  - max_len 50 (default baseline, `results/sasrec/ml-100k/`): HR@10 11.4468 ± 0.4821, NDCG@10 5.6648 ± 0.2159, HR@20 19.7153 ± 1.1902, NDCG@20 7.7396 ± 0.1994
  Per-seed values: `sasrec_ml-100k_history_length_runs.csv`; summary: `sasrec_ml-100k_history_length_summary.csv`.
- **Training and inference timing methodology:** not measured.
- **Checkpoint location and checksum:** Weights are not committed to Git. Per-seed `*_checkpoints.json` records the SHA-256 hash of each weight file.
- **Known limitations / deviations from paper:** The paper fixes max_len at 50 and does not study history length, so there is no paper number to compare with. Only lengths 10, 25 (this experiment) and 50 (baseline run) are available; the 50 baseline was produced in a separate run set with the same seeds. Single dataset (ML-100K, 938 users in the test set), so differences between lengths may be within noise for some metrics.
