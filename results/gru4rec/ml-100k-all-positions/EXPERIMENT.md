# Experiment record: GRU4Rec on MovieLens-100K, all-positions training (extra experiment)

- **Owner / date:** Mahalakshmi Rajkumar; runs in November 2026.
- **Model / run ID:** GRU4Rec (RecBole v1.2.0 CE adapter), full mode, `gru4rec_targets = "all"`, 5 runs with seeds 42-46. Run IDs: `mahalakshmi_gru4rec_full_01` (seed 42), `mahalakshmi_gru4rec_full_seed43` ... `mahalakshmi_gru4rec_full_seed46`.
- **Purpose:** The main GRU4Rec runs (`results/gru4rec/ml-100k/`) train on one target per user (the last training item). This experiment trains on every position in each history, like SASRec+ and ADRec. Model, data, evaluation and all other settings are unchanged. Training-target count: at most 50 training targets per user (last 50 positions) versus one per user.
- **Team commit / upstream commit:** Kaggle code bundle base commit `433125167975d884cc7e5fd10d79855737686782` (working-copy snapshot; per-file SHA-256 hashes are in each `*_environment.json`). Upstream ADRec commit `0a30ef764cd85efa44f884a387198e9bf28c7d4c`.
- **Dataset checksum / split definition:** `dataset.pkl` SHA-256 `dfc5dfaa7950d1a85af4fd6bb168c38efc8213092a9f72563c6d75ab00cef8d0`. 938 users, 1,008 items; train 52,537 / val 938 / test 938 interactions. Released per-user holdout split (see `evaluation/PROTOCOL.md`).
- **Config file / random seed:** `make_config('gru4rec', 'full', 'ml-100k')` with `gru4rec_targets` set to `"all"`; per-seed configs saved as `gru4rec_seed{42..46}_config.json`. Key settings: max_len 50, hidden size 128, learning rate 0.001, weight decay 1e-05, train/eval batch size 512, up to 500 epochs, evaluation every 5 epochs, patience 4.
- **Python and package versions:** Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 (from seed 42's `gru4rec_seed42_environment.json`; full list in `requirements-lock.txt`).
- **Hardware / accelerator:** Kaggle, Tesla T4.
- **Training command or notebook:** Kaggle notebook, full mode, finalize on. The notebook filename was not recorded in the result files; fill in before submission.
- **Evaluation command:** built into `train(..., finalize=True)`: shared evaluator, protocol `adrec_upstream`, run once on the test split after training.
- **Training stages completed / pretrained weight source:** single stage, trained from scratch; no pretrained weights.
- **Validation selection rule:** Validation every 5 epochs; a checkpoint becomes best.pt when any tracked validation metric (HR or NDCG at 5, 10, 20) reaches a new best; training stops after 4 consecutive validation checks with no improvement (up to 500 epochs). Test is evaluated once, on best.pt, at the end. Best epochs (seeds 42-46): 155, 200, 245, 255, 260; epochs completed: 176, 221, 266, 276, 281.
- **Metrics and candidate-item rules:** HR@5/10/20 and NDCG@5/10/20, in percent, protocol `adrec_upstream` (see `evaluation/PROTOCOL.md`): all item columns, no filtering of padding or previously seen items, evaluation batch size 512.
- **Results / variability:** mean ± sample standard deviation (n-1) over 5 seeds, percent:
  - HR@10 12.7362 ± 0.2004, NDCG@10 6.3487 ± 0.0864, HR@20 19.8903 ± 0.5413, NDCG@20 8.1398 ± 0.1609
  Per-seed values: `gru4rec_ml-100k_all_positions_runs.csv`; summary: `gru4rec_ml-100k_all_positions_summary.csv`. Compare with the last-only runs in `results/gru4rec/ml-100k/`.
- **Training and inference timing methodology:** not measured.
- **Checkpoint location and checksum:** Weights are not committed to Git. Per-seed `gru4rec_seed<N>_checkpoints.json` records the SHA-256 hash of each weight file.
- **Known limitations / deviations from paper:**
  - Code: `src/models/gru4rec.py` (`calculate_loss_all_positions`) and `src/training/kaggle.py` (`gru4rec_targets`). The default stays `"last"`, so the main results are unchanged. These code paths are not covered by the unit tests.
  - Training on all positions is intended to match RecBole-style prefix augmentation (each GRU output depends only on earlier items). This equivalence was reasoned from the model structure, not verified against RecBole's data pipeline.
  - Histories are truncated to the last 50 items, so this may differ from full prefix augmentation for users with longer histories.
  - The repo's loader uses each user's last items as validation/test (released split), not the paper's 8:1:1 user split.
  - Paper Table 2 numbers are not copied into this record; compare with the paper separately.
