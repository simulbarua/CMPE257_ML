# ADRec in Practice: Reproduction, Evaluation, and Extensions

This project implements and evaluates ADRec using the authors' code. We compare ADRec, GRU4Rec, and SASRec+ on Amazon Baby, Amazon Beauty, and MovieLens-100K. Our extensions study diffusion steps, warm-up, pretrained embeddings, and maximum sequence length. A local web demo serves recommendations from a trained checkpoint.

## Model comparison across datasets

All nine model–dataset combinations have five completed runs (seeds 42–46). Each dataset has its own trained models and item vocabulary. The table shows test HR@20 and NDCG@20 as percentages, with mean ± sample standard deviation. These are the original maximum-length-50 runs; ADRec uses our per-seed pretrained embeddings and 32 diffusion steps. Each model link opens its saved summary, including the other evaluation cutoffs.

| Dataset | Model | HR@20 | NDCG@20 |
|---|---|---:|---:|
| Baby | [ADRec](results/adrec/baby/adrec_baby_summary.csv) | 6.2568 ± 0.2446 | 2.9065 ± 0.0509 |
| Baby | [SASRec+](results/sasrec/baby/sasrec_baby_summary.csv) | 5.5902 ± 0.2868 | 2.5217 ± 0.0766 |
| Baby | [GRU4Rec](results/gru4rec/baby/gru4rec_baby_summary.csv) | 3.4234 ± 0.1495 | 1.3139 ± 0.0499 |
| Beauty | [ADRec](results/adrec/beauty/adrec_beauty_summary.csv) | 15.4262 ± 0.1223 | 8.0214 ± 0.0727 |
| Beauty | [SASRec+](results/sasrec/beauty/sasrec_beauty_summary.csv) | 15.0679 ± 0.2255 | 7.7009 ± 0.0737 |
| Beauty | [GRU4Rec](results/gru4rec/beauty/gru4rec_beauty_summary.csv) | 7.0133 ± 0.3117 | 3.0502 ± 0.1165 |
| MovieLens-100K | [ADRec](results/adrec/ml-100k/adrec_ml-100k_summary.csv) | 21.4280 ± 0.5306 | 8.0441 ± 0.3419 |
| MovieLens-100K | [SASRec+](results/sasrec/ml-100k/sasrec_ml-100k_summary.csv) | 19.7153 ± 1.1902 | 7.7396 ± 0.1994 |
| MovieLens-100K | [GRU4Rec](results/gru4rec/ml-100k/gru4rec_ml-100k_summary.csv) | 8.3149 ± 0.9113 | 3.0688 ± 0.2091 |

ADRec has the highest mean HR@20 and NDCG@20 among our three implementations on each dataset. Its relative gains over SASRec+ are smaller on Beauty and MovieLens-100K than on Baby. These comparisons describe our runs, not a claim of statistical significance. The shared evaluator uses the released per-user holdout splits, full catalog scoring, and the upstream batch-mean HR/NDCG calculation. Our GRU4Rec adapter and documented training differences limit direct comparisons with the paper. See the [evaluation protocol](evaluation/PROTOCOL.md) and [implementation limitations](docs/MULTI_DATASET_TRAINING.md#report-section-differences-and-limitations).

## MovieLens-100K maximum sequence length

We retrained ADRec, SASRec+, and GRU4Rec with maximum sequence lengths of 10 and 25 for seeds 42–46, then compared them with their existing length-50 baselines. The limit applies during both training and evaluation; ADRec embedding pretraining was also repeated at each length. This measures the effect of changing the training setup, rather than shortening inputs to one fixed trained model.

The table shows five-seed means in percent. Full summaries include sample standard deviations: [ADRec](results/adrec/max-seq-length/adrec_ml-100k_history_length_summary.csv), [SASRec+](results/sasrec/max-seq-length/sasrec_ml-100k_history_length_summary.csv), and [GRU4Rec](results/gru4rec/max-seq-length/gru4rec_ml-100k_history_length_summary.csv).

| Model | Maximum length | HR@20 | NDCG@20 |
|---|---:|---:|---:|
| ADRec | 10 | 16.4078 | 6.4172 |
| ADRec | 25 | 19.5744 | 7.6131 |
| ADRec | 50 (baseline) | 21.4280 | 8.0441 |
| SASRec+ | 10 | 15.7856 | 6.0174 |
| SASRec+ | 25 | 18.2024 | 6.9458 |
| SASRec+ | 50 (baseline) | 19.7153 | 7.7396 |
| GRU4Rec | 10 | 7.5019 | 2.9115 |
| GRU4Rec | 25 | 8.2405 | 3.0903 |
| GRU4Rec | 50 (baseline) | 8.3149 | 3.0688 |

For ADRec and SASRec+, both top-20 metrics increased as the maximum length increased from 10 to 25 to 50. GRU4Rec's HR@20 also increased, while its NDCG@20 was slightly higher at length 25 than at 50. The experiment covers only MovieLens-100K and does not establish the same pattern for the Amazon datasets. Length 50 comes from the original run set, not a new run in the sequence-length notebooks.

Run details and notebooks: [ADRec](notebooks/adrec_maximum_sequence_length_kaggle.ipynb), [SASRec+](notebooks/sasrec_maximum_sequence_length_kaggle.ipynb), [GRU4Rec](notebooks/gru4rec_maximum_sequence_length_kaggle.ipynb).

## Baby diffusion-step experiment

We trained 8- and 16-step ADRec variants using our saved per-seed embeddings and compared them with the existing 32-step models. All variants were evaluated and timed in the same Kaggle Tesla T4 session. Accuracy values below are five-seed means in percent; latency is mean model inference time for a batch of one user, excluding web-request overhead.

| Diffusion steps | HR@20 | NDCG@20 | Inference time (ms) | Speedup vs. 32 steps |
|---|---:|---:|---:|---:|
| 8 | 6.1660 | 2.8715 | 31.50 | 3.65× |
| 16 | 6.3154 | 2.9075 | 59.45 | 1.93× |
| 32 | 6.2568 | 2.9065 | 114.96 | 1.00× |

Sixteen steps gave similar top-20 accuracy with about 1.93× faster inference, while eight steps gave about 3.65× faster inference with a small average accuracy decrease. Accuracy differences were small relative to variation across seeds. The truncated-linear noise schedule also changes at small step counts, so this experiment does not isolate iteration count alone. Timing is specific to this hardware and session.

See the [accuracy summary](results/adrec_diffusion_steps/comparisons/20261115T085617054734Z/accuracy_summary.csv), [latency summary](results/adrec_diffusion_steps/comparisons/20261115T085617054734Z/latency_summary.csv), and [experiment notebook](notebooks/adrec_diffusion_steps_kaggle.ipynb).

## Baby warm-up ablation

We trained five new ADRec models with `warmup_epochs = 0`, reusing each seed's original pretrained embeddings. The full-run configuration otherwise stayed unchanged, including 32 diffusion steps. Our values below are mean ± sample standard deviation across seeds 42–46, in percentage units.

| Experiment | HR@20 | NDCG@20 |
|---|---:|---:|
| Our ADRec, five warm-up epochs | 6.2568 ± 0.2446 | 2.9065 ± 0.0509 |
| Our ADRec, no warm-up | 6.2597 ± 0.2379 | 2.8847 ± 0.0440 |
| Paper, with warm-up | 7.1524 | 3.1455 |
| Paper, without warm-up | 6.7507 | 2.9535 |

Removing warm-up left HR@20 nearly unchanged (+0.0029 percentage points) and slightly reduced NDCG@20 (−0.0218 percentage points). NDCG@20 decreased in all five seeds. This suggests a small ranking benefit from warm-up in our Baby runs, but the effect was much smaller than in the paper. Published values are from [Table 4](https://arxiv.org/html/2505.19544v1#S4.SS4); the paper does not provide standard deviations in that table.

See the [saved comparison](results/adrec_warmup_ablation/baby/accuracy_summary.csv), [report discussion](docs/MULTI_DATASET_TRAINING.md#report-section-baby-warm-up-ablation), and [Kaggle notebook](notebooks/adrec_warmup_ablation_kaggle.ipynb). Model checkpoints remain outside Git.

## Baby pretrained-embedding comparison

We trained ten fresh ADRec models using the authors' released Baby embeddings: seeds 42–46 with either five or zero warm-up epochs. All runs used the same released embedding checkpoint and 32 diffusion steps. No embedding pretraining was repeated. The table compares these runs with our earlier results; project values are mean ± sample standard deviation in percentage units.

| Embedding source | Warm-up epochs | HR@20 | NDCG@20 |
|---|---:|---:|---:|
| Our embeddings | 5 | 6.2568 ± 0.2446 | 2.9065 ± 0.0509 |
| Authors' embeddings | 5 | 6.5715 ± 0.0599 | 2.9499 ± 0.0307 |
| Our embeddings | 0 | 6.2597 ± 0.2379 | 2.8847 ± 0.0440 |
| Authors' embeddings | 0 | 6.5513 ± 0.2072 | 2.9131 ± 0.0862 |
| Paper | 5 | 7.1524 | 3.1455 |
| Paper | 0 | 6.7507 | 2.9535 |

With five warm-up epochs, the authors' embeddings improved average HR@20 by 0.3148 percentage points (5.03% relative) and NDCG@20 by 0.0434 points (1.49% relative). Both improved in four of five seeds, although average NDCG@5 decreased. With the authors' embeddings, warm-up added only 0.0202 points to HR@20 and 0.0368 points to NDCG@20; both improved in three of five seeds. This was a smaller benefit than the paper reported, and our scores remained below its corresponding results. Published values are from [Table 4](https://arxiv.org/html/2505.19544v1#S4.SS4), which does not report standard deviations.

The new runs use one fixed pretrained checkpoint, while our earlier runs pretrained embeddings separately for each seed. Our training implementation also retains its documented differences from the released code. These results support a modest benefit from this released checkpoint under our setup, but do not identify pretraining as the sole cause of the gap from the paper.

See the [comparison results](results/adrec_author_embeddings/baby/accuracy_summary.csv), [report discussion](docs/MULTI_DATASET_TRAINING.md#report-section-baby-pretrained-embedding-comparison), and [Kaggle notebook](notebooks/adrec_author_embeddings_kaggle.ipynb).

## Run the web demo

The [web demo](app/README.md) accepts a history of item IDs and displays top-k recommendations, model scores, and inference time. It loads one checkpoint at startup. Use the matching dataset and trained checkpoint in the Python environment with `requirements-shared.txt` installed:

```bash
python -m app.server --checkpoint checkpoints/baby/best.pt --data data/baby/dataset.pkl
```

Open <http://localhost:8000>. The checkpoint path is an example; use the location of your local `best.pt`. Beauty and MovieLens-100K require their own checkpoints and datasets, plus `--dataset-name beauty` or `--dataset-name ml-100k`.

For a UI-only check, `python -m app.server --fake` uses the local Baby dataset and random scores. Those scores are not model predictions. The current demo displays encoded item IDs, not product names or images, and uses placeholder example histories. It filters padding from recommendations; the recorded experiment metrics retain the upstream candidate rules. See [setup and limitations](app/README.md).

## Structure

| Folder | Purpose |
|---|---|
| `src/` | Team model adapters and shared data/evaluation code |
| `configs/` | Versioned model and experiment settings |
| `notebooks/` | Colab/Kaggle launch notebooks |
| `data/` | Local datasets; actual data is ignored |
| `evaluation/` | Evaluation protocol and result comparison |
| `results/` | Metrics, summaries, configurations, histories, and experiment records |
| `checkpoints/` | Local weights; actual weights are ignored |
| `app/` | Web demo |
| `docs/` | Task plan, decisions, and experiment notes |

## Sources and attribution

- [ADRec paper](https://arxiv.org/abs/2505.19544)
- [Authors' implementation](https://github.com/Nemo-1024/ADRec)
- [Released dataset URLs, checksums, and counts](configs/datasets.json)

ADRec is the authors' work. This repository contains our training integration, evaluation, and experiments. Pinned upstream model sources, licenses, and attribution are retained in `third_party/`; datasets and model weights are excluded from Git. Implementation differences and limitations are documented in the [training guide](docs/KAGGLE_TRAINING.md#implementation-and-evaluation-protocol).

See [local validation](docs/LOCAL_VALIDATION.md) for evaluation parity and model checks.
