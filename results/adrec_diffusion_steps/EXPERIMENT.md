# Experiment record: ADRec diffusion steps (8, 16, 32) on Amazon Baby

- **Owner / date:** Simul Barua; session `20261115T085617054734Z` (2026-11-15).
- **Model / run ID:** ADRec, full mode. 8- and 16-step variants trained for seeds 42-46 (`adrec_steps8_seed4x`, `adrec_steps16_seed4x`); the 32-step baseline is the existing five-seed ADRec run in `results/adrec/baby/`, loaded from its checkpoints and not retrained.
- **What changes:** the number of diffusion steps, using the authors' truncated-linear noise schedule (`noise_schedules.json`). The schedule's endpoint formula changes at small step counts, so differences cannot be attributed solely to iteration count. Each variant starts from that seed's existing pretrained item embeddings.
- **Team commit / upstream commit:** Kaggle bundle base commit `400391097930be1009923bfb3db79e0d3d02e65b`, branch `codex/adrec-diffusion-steps` (from `comparisons/*/environment.json`); upstream ADRec commit `0a30ef764cd85efa44f884a387198e9bf28c7d4c`.
- **Dataset checksum / split definition:** Baby `dataset.pkl` SHA-256 `bf8f3485f9e36763e07f0234e0ddb479cb023358797615388a36f0d7238cea99`; 11,761 users, 4,731 items; train 69,301 / val 11,761 / test 11,761. Released per-user holdout split.
- **Config file / random seed:** per-run `config.json` and `experiment_input.json` in each `adrec_steps*_seed*/` folder; seeds 42-46; max_len 50, hidden size 128, learning rate 0.001, weight decay 1e-5, batch size 512, warm-up epochs 5, up to 500 epochs, evaluation every 5 epochs, patience 4; only `diffusion_steps` differs.
- **Python and package versions:** Python 3.13.16, PyTorch 2.11.0+cu130, CUDA 13.0 (`comparisons/*/environment.json`).
- **Hardware / accelerator:** Kaggle, Tesla T4.
- **Training command or notebook:** `notebooks/adrec_diffusion_steps_kaggle.ipynb`.
- **Evaluation command:** same notebook; per-seed accuracy in `comparisons/*/steps{8,16,32}_seed*_evaluation.json`, combined in `accuracy_runs.csv` / `accuracy_summary.csv`.
- **Training stages completed / pretrained weight source:** ADRec alignment + fine-tune stage only, starting from each seed's existing pretrained embeddings (SHA-256 recorded in each run's `experiment_input.json`). Pretraining was not repeated.
- **Validation selection rule:** Validation every 5 epochs; a checkpoint becomes best.pt when any tracked validation metric (HR or NDCG at 5, 10, 20) reaches a new best; training stops after 4 consecutive validation checks with no improvement (up to 500 epochs). Test is evaluated once, on best.pt, at the end.
- **Metrics and candidate-item rules:** HR@5/10/20 and NDCG@5/10/20, in percent, protocol `adrec_upstream` (see `evaluation/PROTOCOL.md`): all item columns, no filtering of padding or previously seen items, evaluation batch size 512.
- **Results / variability:** mean ± sample standard deviation (n-1) over 5 seeds, percent:
  - 8 steps: HR@10 4.0753 ± 0.1456, NDCG@10 2.3489 ± 0.0382, HR@20 6.1660 ± 0.2946, NDCG@20 2.8715 ± 0.0680
  - 16 steps: HR@10 4.1234 ± 0.1000, NDCG@10 2.3557 ± 0.0414, HR@20 6.3154 ± 0.2321, NDCG@20 2.9075 ± 0.0781
  - 32 steps: HR@10 4.1097 ± 0.1296, NDCG@10 2.3692 ± 0.0347, HR@20 6.2568 ± 0.2446, NDCG@20 2.9065 ± 0.0509
  Differences are small compared with the standard deviations. Differences and relative changes vs 32 steps are in `accuracy_summary.csv`.
- **Training and inference timing methodology:** inference latency measured per batch on a T4 with the run order recorded in (`run_order.json`, `raw_timings.json`, `latency_runs.csv`, `latency_summary.csv`):
  - 8 steps, batch size 1: mean 31.50 ms per batch, 31.8 users/s, speedup vs 32 steps 3.65x
  - 8 steps, batch size 32: mean 32.63 ms per batch, 981.4 users/s, speedup vs 32 steps 3.66x
  - 16 steps, batch size 1: mean 59.45 ms per batch, 16.8 users/s, speedup vs 32 steps 1.93x
  - 16 steps, batch size 32: mean 61.14 ms per batch, 523.5 users/s, speedup vs 32 steps 1.95x
  - 32 steps, batch size 1: mean 114.96 ms per batch, 8.7 users/s, speedup vs 32 steps 1.00x
  - 32 steps, batch size 32: mean 119.51 ms per batch, 267.8 users/s, speedup vs 32 steps 1.00x
- **Checkpoint location and checksum:** weights not committed to Git; per-run `checkpoints.json` records SHA-256 hashes.
- **Known limitations / deviations from paper:** Baby only. The paper reports ADRec is not sensitive to step count; this study is a sensitivity check, not a reproduction of a specific paper table. Inference is stochastic, so scores depend on the evaluation seed. Timing comes from a single Kaggle T4 session and may vary with load.
