# GRU4Rec maximum sequence length on MovieLens-100K (all-positions training)

Lengths 10 and 25, seeds 42 to 46, full mode, `cfg["gru4rec_targets"] = "all"`.

The maximum history length changes in both training and testing; this is not an
evaluation-only truncation of a trained model. Everything else matches the all-positions
GRU4Rec runs in `results/gru4rec/ml-100k-all-positions/`, which provide the length-50 point.

| Max length | HR@10 | HR@20 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|
| 10 | 9.13 ± 0.43 | 16.02 ± 0.48 | 4.39 ± 0.20 | 6.12 ± 0.10 |
| 25 | 12.38 ± 0.86 | 21.15 ± 0.58 | 5.85 ± 0.26 | 8.04 ± 0.18 |
| 50 (`ml-100k-all-positions`) | 12.74 ± 0.20 | 19.89 ± 0.54 | 6.35 ± 0.09 | 8.14 ± 0.16 |

Mean ± sample standard deviation (n-1) over five seeds, in percent.

- HR@20 is highest at length 25 and drops at length 50; NDCG@20 is roughly flat from 25 to 50.
  Longer history does not help GRU4Rec monotonically on this dataset.
- Length 10 is clearly worse than 25 on every metric.

Files:
- `gru4rec_ml-100k_history_length_runs.csv`: per-seed results.
- `gru4rec_ml-100k_history_length_summary.csv`: mean and std per length.
- `len10/`, `len25/`: per-seed configs, environments, histories, results and checkpoint hashes.
- `EXPERIMENT.md`: experiment record.

Weights are not committed to Git; the `*_checkpoints.json` files hold SHA-256 hashes.
