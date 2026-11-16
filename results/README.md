Save small metrics CSVs and plots here. Include run IDs and link an experiment record. Do not store checkpoints or raw data.

## Completed ADRec runs

Each dataset has five independent runs with seeds 42–46, including separate embedding pretraining for each seed. Values below are mean ± sample standard deviation, in percentage units. These original runs strictly freeze embeddings during the five-epoch warm-up and remain unchanged. The separate zero-warm-up ablation is listed below.

| Dataset | Result folder | HR@20 | NDCG@20 |
|---|---|---:|---:|
| Baby | [adrec/baby](adrec/baby/) | 6.2568 ± 0.2446 | 2.9065 ± 0.0509 |
| Beauty | [adrec/beauty](adrec/beauty/) | 15.4262 ± 0.1223 | 8.0214 ± 0.0727 |
| MovieLens-100K | [adrec/ml-100k](adrec/ml-100k/) | 21.4280 ± 0.5306 | 8.0441 ± 0.3419 |

Each folder includes per-seed configurations, results, runtime metadata, training histories, checkpoint hashes, and CSV summaries. Weights and downloaded archives remain local under `checkpoints/`, outside Git; checkpoint paths are unchanged.

Results are organized as `<model>/<dataset>/`. Existing baseline results are in [gru4rec/baby](gru4rec/baby/) and [sasrec/baby](sasrec/baby/). The former `sacrec` folder is now named `sasrec`. New datasets for those models belong under the same model folders when results are available. Saved runtime paths and source hashes in historical metadata are retained as recorded.

The [differences and limitations section](../docs/MULTI_DATASET_TRAINING.md#report-section-differences-and-limitations) documents how the project runner differs from the released implementation. The completed Baby [diffusion-step experiment](adrec_diffusion_steps/) remains separate.

## Baby warm-up ablation

The [zero-warm-up experiment](adrec_warmup_ablation/baby/) is complete for seeds 42–46. Removing warm-up left mean HR@20 nearly unchanged (6.2568 → 6.2597) and slightly reduced mean NDCG@20 (2.9065 → 2.8847). NDCG@20 declined in all five seeds. See the [comparison summary](adrec_warmup_ablation/baby/accuracy_summary.csv) and [report discussion](../docs/MULTI_DATASET_TRAINING.md#report-section-baby-warm-up-ablation) for variability, the paper comparison, and limitations. The folder contains result metadata only; model weights and ZIP backups remain under the ignored `checkpoints/` directory.
