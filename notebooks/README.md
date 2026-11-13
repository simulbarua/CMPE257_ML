# Kaggle training notebooks

- `simul_adrec_kaggle.ipynb`: embedding pretraining, ADRec alignment, and fine-tuning.
- `anusha_gru4rec_kaggle.ipynb`: RecBole-derived GRU4Rec CE baseline.
- `mahalakshmi_sasrec_kaggle.ipynb`: authors' SASRec+ baseline.

Start with smoke mode. Each notebook needs the shared code ZIP and the Baby dataset attached to Kaggle; the notebook alone does not contain the model code. See [the full setup and resume guide](../docs/KAGGLE_TRAINING.md).

Build the local source ZIP without pushing: `python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl`. The ZIP includes a source manifest. Notebook outputs are cleared; do not commit weights or secrets.
