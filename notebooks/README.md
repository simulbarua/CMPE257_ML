# ADRec training notebook

[adrec_kaggle.ipynb](adrec_kaggle.ipynb) sweeps seeds 42–46 directly in notebook cells through embedding pretraining, ADRec alignment, fine-tuning, test evaluation, and result export. The training cell calls the existing shared `train()` function; export functions are defined in the notebook.

1. Build the upload bundle with `python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl`.
2. Attach the updated code ZIP and Baby dataset to Kaggle.
3. Import the notebook, enable Internet and a GPU, and run the numbered sections in order.
4. Download `adrec_results.zip` and each completed seed's checkpoint ZIP.

The default is full training for all five seeds. Smoke mode runs small checks in separate output folders. Completed stages are reused when rerunning the training cell; attached run folders support cross-session resume through `RESUME_ROOT`.

See [the training instructions](../docs/KAGGLE_TRAINING.md) for the CSV format, export layout, and resume procedure. This directory contains only the ADRec notebook; baseline model implementations remain available in the shared code.
