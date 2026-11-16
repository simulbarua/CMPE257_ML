# CMPE257 — ADRec project

Team: Simul Barua, Anusha, and Mahalakshmi.

This project implements and evaluates ADRec using the authors' code. We compare ADRec with GRU4Rec and SASRec+ on Amazon Baby Products and extend ADRec training to Beauty and MovieLens-100K. Simul trains ADRec, Anusha trains SASRec+, and Mahalakshmi trains GRU4Rec. A planned web demo will serve ADRec; baseline comparison is optional.

## Start here

1. Read [team tasks](docs/TASKS.md) and [contribution workflow](CONTRIBUTING.md).
2. Read the [evaluation protocol](evaluation/PROTOCOL.md) and [dataset settings](docs/MULTI_DATASET_TRAINING.md) before running experiments.
3. Follow the [Kaggle training guide](docs/KAGGLE_TRAINING.md) to run the ADRec experiments with the pinned upstream models.

**Status:** five-seed Baby results for ADRec, GRU4Rec, and SASRec+ and the Baby diffusion-step experiment are available. Shared code and Kaggle notebooks now support Baby, Beauty, and MovieLens-100K; five-seed ADRec results are available for all three datasets. Results are organized by model and dataset under `results/<model>/<dataset>/`. See [multi-dataset training](docs/MULTI_DATASET_TRAINING.md). The web application remains to be implemented.

Start with the [shared-code walkthrough](docs/SHARED_CODE_GUIDE.md).

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
