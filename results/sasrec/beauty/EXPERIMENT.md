# Experiment record: SASRec+ on Beauty

- **Owner / date:** Anusha Subramanian; runs in November 2026.
- **Model / run ID:** SASRec+ (vendored from the ADRec authors' repo, `src/sasrec.py`), full mode, 5 runs with seeds 42-46. Run IDs: `sasrec_full_seed42` ... `sasrec_full_seed46`.
- **Team commit / upstream commit:** Kaggle code bundle base commit `61de4e8e7b6f6e527e9664faecbb9e3e8458afcb` (working-copy snapshot; per-file SHA-256 hashes are in each `*_environment.json`). Upstream ADRec commit `0a30ef764cd85efa44f884a387198e9bf28c7d4c`.
- **Dataset checksum / split definition:** `dataset.pkl` SHA-256 `a2f6ca22b3efe2bc5030ef9a7c4f6c928db2086771d58d9a40147e660f574bc1`. 10,553 users, 6,086 items; train 73,042 / val 10,553 / test 10,553 interactions. Released per-user holdout split (see `evaluation/PROTOCOL.md`).
- **Config file / random seed:** `configs/shared.json` + `configs/kaggle.json` via `make_config('sasrec', 'full', 'beauty')`; per-seed configs saved as `sasrec_seed{42..46}_config.json`. Seeds 42, 43, 44, 45, 46. Key settings: max_len 50, hidden size 128, learning rate 0.001, weight decay 1e-5, train/eval batch size 512, up to 500 epochs, evaluation every 5 epochs, patience 4.
- **Python and package versions:** Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 (from seed 42's `environment.json`; full list in `requirements-lock.txt`).
- **Hardware / accelerator:** Kaggle, Tesla T4.
- **Training command or notebook:** `notebooks/sasrec_kaggle.ipynb` (full mode, finalize on).
- **Evaluation command:** built into `train(..., finalize=True)`: shared evaluator, protocol `adrec_upstream`, run once on the test split after training.
- **Training stages completed / pretrained weight source:** single stage, trained from scratch; no pretrained weights.
- **Validation selection rule:** Validation every 5 epochs; a checkpoint becomes best.pt when any tracked validation metric (HR or NDCG at 5, 10, 20) reaches a new best; training stops after 4 consecutive validation checks with no improvement (up to 500 epochs). Test is evaluated once, on best.pt, at the end. Best epochs (seeds 42-46): 45, 40, 45, 40, 50; epochs completed: 66, 61, 66, 61, 71.
- **Metrics and candidate-item rules:** HR@5/10/20 and NDCG@5/10/20, in percent, protocol `adrec_upstream` (see `evaluation/PROTOCOL.md`): all item columns, no filtering of padding or previously seen items, evaluation batch size 512.
- **Results / variability:** mean ± sample standard deviation (n-1) over 5 seeds, percent:
  - HR@10 10.9527 ± 0.2242, NDCG@10 6.6637 ± 0.0986, HR@20 15.0679 ± 0.2255, NDCG@20 7.7009 ± 0.0737
  Per-seed values: `sasrec_beauty_runs.csv`; summary: `sasrec_beauty_summary.csv`.
- **Training and inference timing methodology:** not measured.
- **Checkpoint location and checksum:** Weights are not committed to Git. Per-seed `*_checkpoints.json` records the SHA-256 hash of each weight file.
- **Known limitations / deviations from paper:**
  - SASRec+ here is full-item cross-entropy SASRec with a learned position table of size max_len (vendored from the ADRec authors' repo); results should be compared with the paper's SASRec+ numbers separately.
  - The repo's loader uses each user's last items as validation/test (released split), not the paper's 8:1:1 user split.
  - Paper Table 2 numbers are not copied into this record; compare with the paper separately.
