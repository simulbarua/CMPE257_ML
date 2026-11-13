# CMPE257 — ADRec project

Team: Simul Barua, Anusha, and Mahalakshmi.

We will reproduce ADRec on Amazon Baby Products and compare it with GRU4Rec and SASRec+. Each member trains one model. A web demo will initially serve ADRec; baseline comparison is optional.

## Start here

1. Read [team tasks](docs/TASKS.md) and [contribution workflow](CONTRIBUTING.md).
2. Complete the data and evaluation agreement before full training.
3. Bring in and verify the upstream implementation before adding dependencies or training commands.

**Status:** shared data preparation and ranking evaluation are implemented. Local ADRec/SASRec+/GRU4Rec-adapter smoke checks are available; full training launchers, Kaggle notebooks, and the application are not implemented yet.

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

ADRec is the authors' work. This repository will contain our reproduction, evaluation, and demo additions. Before importing upstream code, check its license, preserve attribution and notices, and record the exact commit. Do not present imported code as original team work. Pinned upstream model sources and licenses are now retained in `third_party/`; the dataset is excluded from Git.

Existing local proposal archives are intentionally not included in the first Git commit; review them before publishing conversation exports or other course documents.

See [local validation](docs/LOCAL_VALIDATION.md) for evaluation parity and model checks.
