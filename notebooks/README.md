# Kaggle training notebooks

- [adrec_kaggle.ipynb](adrec_kaggle.ipynb): embedding pretraining, ADRec alignment, and fine-tuning.
- [gru4rec_kaggle.ipynb](gru4rec_kaggle.ipynb): RecBole-derived GRU4Rec CE baseline.
- [sasrec_kaggle.ipynb](sasrec_kaggle.ipynb): original SASRec implementation included in ADRec, used as the SASRec+ baseline.

Run the numbered sections in order: load project files, install dependencies, check the runtime, configure the run, train, inspect results, load the saved model, and save outputs. ADRec includes a separate embedding pretraining stage. Resume instructions follow the output section.

Attach the shared code ZIP and the Baby dataset to Kaggle. Run smoke mode to verify execution, then select full mode and a new run name. See [the setup and resume guide](../docs/KAGGLE_TRAINING.md) for configuration and checkpoint details.

Build the local source ZIP without pushing: `python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl`. The ZIP includes a source manifest. Notebook outputs are cleared; do not commit weights or secrets.
