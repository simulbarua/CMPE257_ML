# CMPE257 — ADRec project

Team: Simul Barua, Anusha, and Mahalakshmi.

This project implements and evaluates ADRec using the authors' code. We compare ADRec with GRU4Rec and SASRec+ on Amazon Baby Products and extend ADRec training to Beauty and MovieLens-100K. Simul trains ADRec, Anusha trains SASRec+, and Mahalakshmi trains GRU4Rec. A planned web demo will serve ADRec; baseline comparison is optional.

## Start here

1. Read [team tasks](docs/TASKS.md) and [contribution workflow](CONTRIBUTING.md).
2. Read the [evaluation protocol](evaluation/PROTOCOL.md) and [dataset settings](docs/MULTI_DATASET_TRAINING.md) before running experiments.
3. Follow the [Kaggle training guide](docs/KAGGLE_TRAINING.md) to run the ADRec experiments with the pinned upstream models.

**Status:** five-seed Baby results for ADRec, GRU4Rec, and SASRec+ and the Baby diffusion-step experiment are available. Shared code and Kaggle notebooks now support Baby, Beauty, and MovieLens-100K; five-seed ADRec results are available for all three datasets. Results are organized by model and dataset under `results/<model>/<dataset>/`. See [multi-dataset training](docs/MULTI_DATASET_TRAINING.md). The web application remains to be implemented.

Start with the [shared-code walkthrough](docs/SHARED_CODE_GUIDE.md).

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

## Structure

| Folder | Purpose |
|---|---|
| `src/` | Team model adapters and shared data/evaluation code |
| `configs/` | Versioned model and experiment settings |
| `notebooks/` | Colab/Kaggle launch notebooks |
| `data/` | Local datasets; actual data is ignored |
| `evaluation/` | Evaluation protocol and result comparison |
| `results/` | Small CSV summaries and plots |
| `checkpoints/` | Local weights; actual weights are ignored |
| `app/` | Web demo |
| `docs/` | Task plan, decisions, and experiment notes |

## Sources and attribution

- [ADRec paper](https://arxiv.org/abs/2505.19544)
- [Authors' implementation](https://github.com/Nemo-1024/ADRec)
- [Baby data](https://github.com/Nemo-1024/ADRec/tree/main/datasets/data/baby)

ADRec is the authors' work. This repository contains our training integration, evaluation, and experiments. Pinned upstream model sources, licenses, and attribution are retained in `third_party/`; datasets and model weights are excluded from Git. Implementation differences and limitations are documented in the [training guide](docs/KAGGLE_TRAINING.md#implementation-and-evaluation-protocol).

See [local validation](docs/LOCAL_VALIDATION.md) for evaluation parity and model checks.
