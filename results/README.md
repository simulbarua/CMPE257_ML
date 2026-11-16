Save small metrics CSVs and plots here. Include run IDs and link an experiment record. Do not store checkpoints or raw data.

## Completed ADRec runs

Each dataset has five independent runs with seeds 42–46, including separate embedding pretraining for each seed. Values below are mean ± sample standard deviation, in percentage units. These are the original strict-freeze runs; no upstream warm-up rerun was performed.

| Dataset | Result folder | HR@20 | NDCG@20 |
|---|---|---:|---:|
| Baby | [adrec](adrec/) | 6.2568 ± 0.2446 | 2.9065 ± 0.0509 |
| Beauty | [beauty/adrec](beauty/adrec/) | 15.4262 ± 0.1223 | 8.0214 ± 0.0727 |
| MovieLens-100K | [ml-100k/adrec](ml-100k/adrec/) | 21.4280 ± 0.5306 | 8.0441 ± 0.3419 |

Each folder includes per-seed configurations, results, runtime metadata, training histories, checkpoint hashes, and CSV summaries. Weights and downloaded archives remain local under `checkpoints/`, outside Git. Baby retains its existing folder layout.

The [differences and limitations section](../docs/MULTI_DATASET_TRAINING.md#report-section-differences-and-limitations) documents how the project runner differs from the released implementation. The completed Baby [diffusion-step experiment](adrec_diffusion_steps/) remains separate.
