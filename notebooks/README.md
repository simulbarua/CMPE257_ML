# ADRec training notebook

[adrec_kaggle.ipynb](adrec_kaggle.ipynb) sweeps seeds 42–46 directly in notebook cells through embedding pretraining, ADRec alignment, fine-tuning, test evaluation, and result export. The training cell calls the existing shared `train()` function; export functions are defined in the notebook.

1. Build the upload bundle with `python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl`.
2. Attach the updated code ZIP and Baby dataset to Kaggle.
3. Import the notebook, enable Internet and a GPU, and run the numbered sections in order.
4. Download `adrec_results.zip` and each completed seed's checkpoint ZIP.

The default is full training for all five seeds. Smoke mode runs small checks in separate output folders. Completed stages are reused when rerunning the training cell; attached run folders support cross-session resume through `RESUME_ROOT`.

See [the training instructions](../docs/KAGGLE_TRAINING.md) for the CSV format, export layout, and resume procedure. Baseline notebooks and model implementations remain available in the repository.

## Diffusion-step experiment

[adrec_diffusion_steps_kaggle.ipynb](adrec_diffusion_steps_kaggle.ipynb) trains 8- and 16-step ADRec variants for seeds 42–46 using the existing per-seed pretrained embeddings. It loads the saved 32-step models, re-evaluates all variants, and measures synchronized inference latency on the same runtime and inputs. No pretraining or 32-step training is repeated.

Attach the current code bundle, `dataset.pkl`, and `adrec_all_results_and_models.zip` from the completed five-seed experiment. The notebook supports intact ZIPs and the documented extracted layouts. Run `RUN_MODE = 'smoke'` first to check inputs; use `RUN_MODE = 'full'` for the experiment. Full and smoke outputs are separate.

Download `adrec_diffusion_results_full.zip` and `adrec_diffusion_models_full.zip`. Place the extracted results in `results/adrec_diffusion_steps/` and model runs in `checkpoints/adrec_diffusion_steps/`. The notebook explains checkpoint validation, resume, timing scope, and the small-step noise-schedule adjustment.
