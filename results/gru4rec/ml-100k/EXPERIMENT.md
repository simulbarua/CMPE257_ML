# Experiment record: GRU4Rec on MovieLens-100K

- **Owner / date:** Mahalakshmi Rajkumar; runs in November 2026.
- **Model / run ID:** GRU4Rec (RecBole v1.2.0 CE adapter), full mode, 5 runs with seeds 42-46. Run IDs: `mahalakshmi_gru4rec_full_01` (seed 42), `mahalakshmi_gru4rec_full_seed43` ... `mahalakshmi_gru4rec_full_seed46`.
- **Team commit / upstream commit:** Kaggle code bundle base commit `e4f29db2d975f22f6193783a29fd7ac17e739717` (working-copy snapshot; per-file SHA-256 hashes are in each `*_environment.json`). Upstream ADRec commit `0a30ef764cd85efa44f884a387198e9bf28c7d4c`.
- **Dataset checksum / split definition:** `dataset.pkl` SHA-256 `dfc5dfaa7950d1a85af4fd6bb168c38efc8213092a9f72563c6d75ab00cef8d0`. 938 users, 1,008 items; train 52,537 / val 938 / test 938 interactions. Released per-user holdout split (see `evaluation/PROTOCOL.md`).
- **Config file / random seed:** `configs/shared.json` + `configs/kaggle.json` via `make_config('gru4rec', 'full', 'ml-100k')`; per-seed configs saved as `gru4rec_seed{42..46}_config.json`. Seeds 42, 43, 44, 45, 46. Key settings: max_len 50, hidden size 128, learning rate 0.001, weight decay 1e-5, train/eval batch size 512, up to 500 epochs, evaluation every 5 epochs, patience 4.
- **Python and package versions:** Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 (from seed 42's `environment.json`; full list in `requirements-lock.txt`).
- **Hardware / accelerator:** Kaggle, Tesla T4.
- **Training command or notebook:** `notebooks/mahalakshmi_gru4rec_kaggle.ipynb` (full mode, finalize on).
- **Evaluation command:** built into `train(..., finalize=True)`: shared evaluator, protocol `adrec_upstream`, run once on the test split after training.
- **Training stages completed / pretrained weight source:** single stage, trained from scratch; no pretrained weights.
- **Validation selection rule:** Validation every 5 epochs; a checkpoint becomes best.pt when any tracked validation metric (HR or NDCG at 5, 10, 20) reaches a new best; training stops after 4 consecutive validation checks with no improvement (up to 500 epochs). Test is evaluated once, on best.pt, at the end. Best epochs (seeds 42-46): 50, 60, 65, 90, 65; epochs completed: 71, 81, 86, 111, 86.
- **Metrics and candidate-item rules:** HR@5/10/20 and NDCG@5/10/20, in percent, protocol `adrec_upstream` (see `evaluation/PROTOCOL.md`): all item columns, no filtering of padding or previously seen items, evaluation batch size 512.
- **Results / variability:** mean ± sample standard deviation (n-1) over 5 seeds, percent:
  - HR@10 4.5154 ± 0.2685, NDCG@10 2.1089 ± 0.0823, HR@20 8.3149 ± 0.9113, NDCG@20 3.0688 ± 0.2091
  Per-seed values: `gru4rec_ml-100k_runs.csv`; summary: `gru4rec_ml-100k_summary.csv`.
- **Training and inference timing methodology:** not measured.
- **Checkpoint location and checksum:** Weights are not committed to Git. Per-seed `*_checkpoints.json` records the SHA-256 hash of each weight file.
- **Known limitations / deviations from paper:**
  - RecBole-derived GRU4Rec with cross-entropy loss on the final target of each history; not verified to match the paper's GRU4Rec implementation, so scores are not a direct reproduction of the paper's GRU4Rec numbers.
  - The repo's loader uses each user's last items as validation/test (released split), not the paper's 8:1:1 user split.
  - Paper Table 2 numbers are not copied into this record; compare with the paper separately.
