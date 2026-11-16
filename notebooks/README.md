# Kaggle training notebooks

[adrec_kaggle.ipynb](adrec_kaggle.ipynb) sweeps seeds 42–46 directly in notebook cells through embedding pretraining, ADRec alignment, fine-tuning, test evaluation, and result export. The training cell calls the existing shared `train()` function; export functions are defined in the notebook.

1. Build the upload bundle with `python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl`.
2. Attach the updated code ZIP and selected dataset to Kaggle; set `DATASET` in the setup cell.
3. Import the notebook, enable Internet and a GPU, and run the numbered sections in order.
4. Download `adrec_results.zip` and each completed seed's checkpoint ZIP.

The default is full training for all five seeds. Smoke mode runs small checks in separate output folders. Completed stages are reused when rerunning the training cell; attached run folders support cross-session resume through `RESUME_ROOT`.

See [the training instructions](../docs/KAGGLE_TRAINING.md) for the CSV format, export layout, and resume procedure. Baseline notebooks and model implementations remain available in the repository.

## Diffusion-step experiment

[adrec_diffusion_steps_kaggle.ipynb](adrec_diffusion_steps_kaggle.ipynb) trains 8- and 16-step ADRec variants for seeds 42–46 using the existing per-seed pretrained embeddings. It loads the saved 32-step models, re-evaluates all variants, and measures synchronized inference latency on the same runtime and inputs. No pretraining or 32-step training is repeated.

Attach the current code bundle, `dataset.pkl`, and `adrec_all_results_and_models.zip` from the completed five-seed experiment. The notebook supports intact ZIPs and the documented extracted layouts. Run `RUN_MODE = 'smoke'` first to check inputs; use `RUN_MODE = 'full'` for the experiment. Full and smoke outputs are separate.

Download `adrec_diffusion_results_full.zip` and `adrec_diffusion_models_full.zip`. Place the extracted results in `results/adrec_diffusion_steps/` and model runs in `checkpoints/adrec_diffusion_steps/`. The notebook explains checkpoint validation, resume, timing scope, and the small-step noise-schedule adjustment.

## Dataset selection and baseline launchers

All launch notebooks accept `DATASET = 'baby'`, `'beauty'`, or `'ml-100k'`. All model results use `results/<model>/<dataset>/`. Baby retains its original run, checkpoint, and ZIP locations. New datasets use separate run folders and archives. See [multi-dataset training](../docs/MULTI_DATASET_TRAINING.md) for inputs, paths, and compatibility details.

- [GRU4Rec](mahalakshmi_gru4rec_kaggle.ipynb) retains the existing seed-42 run and remaining-seed loop.
- [SASRec+](sasrec_kaggle.ipynb) provides the same workflow using the retained SASRec+ implementation.

Each dataset needs its own trained weights. The optional diffusion notebook requires that dataset’s completed baseline and pretraining checkpoints; it does not require repeating the completed Baby experiment.

Result folders are `results/adrec/<dataset>/`, `results/gru4rec/<dataset>/`, and `results/sasrec/<dataset>/`. Extract each model result ZIP into its dataset folder. The existing `results/adrec_diffusion_steps/` folder stays separate and unchanged.

## Baby warm-up ablation

[adrec_warmup_ablation_kaggle.ipynb](adrec_warmup_ablation_kaggle.ipynb) trains fresh ADRec models for seeds 42–46 with `warmup_epochs = 0`. Each run reuses the exact saved pretrained embeddings from its matching Baby baseline. The full-run configuration otherwise stays unchanged, including 32 diffusion steps. No embedding pretraining is repeated.

Attach the private input `adrec_baby_warmup0_inputs.zip`, which includes the source bundle, Baby dataset, five pretraining checkpoints, and original baseline metadata. Import the notebook, enable Internet and one GPU, and run all cells. The default `RUN_FULL = True` runs a one-epoch smoke check before the five full runs; `False` runs only the smoke check. The check verifies that embeddings update from epoch zero.

Download `adrec_baby_warmup0_results.zip` and `adrec_baby_warmup0_models.zip` from `/kaggle/working/exports/`. Extract results into `results/adrec_warmup_ablation/baby/` and model runs into `checkpoints/adrec_warmup_ablation/baby/`. Completed seeds are reused, and incomplete runs resume from `latest.pt`. For a new session, attach the extracted model archive and set `RESUME_ROOT` to the folder containing `adrec_warmup0_seed<N>/`.

The comparison reports mean and sample standard deviation over matching completed seeds, plus per-seed differences in percentage points. Original baseline scores are reused as recorded. Paper Table 4 values are included separately as references. The documented training implementation differences still apply. All five full ablation runs are complete; see the [results](../results/adrec_warmup_ablation/baby/accuracy_summary.csv) and [report discussion](../docs/MULTI_DATASET_TRAINING.md#report-section-baby-warm-up-ablation).

## Baby authors' embedding comparison

[adrec_author_embeddings_kaggle.ipynb](adrec_author_embeddings_kaggle.ipynb) runs ten new ADRec experiments: seeds 42–46 with `warmup_epochs = 5` and `0`, all using one fixed authors' pretrained Baby checkpoint and 32 diffusion steps. Each backbone starts fresh. No pretraining, baseline training, or diffusion-step experiment is repeated. Historical project results provide the other two conditions in a four-condition comparison.

The original checkpoint comes from [the pinned official Baby release](https://github.com/Nemo-1024/ADRec/blob/0a30ef764cd85efa44f884a387198e9bf28c7d4c/src/saved/pretrain/baby/pretrain.pth). Its file hashes, embedding dimensions, and provenance are recorded in `configs/author_baby_embeddings.json`. Store it locally as `checkpoints/author_embeddings/baby/pretrain.pth`. The file contains no pretraining seed metadata; do not assign it a seed based on the downstream run.

Build the upload bundle from the repository root:

```bash
python -m scripts.package_author_embeddings
```

This requires the pinned checkpoint, `data/baby/dataset.pkl`, and the existing Baby results for both project warm-up settings. It writes `work/author_embeddings_upload/adrec_baby_author_embeddings_inputs.zip` and a copy of the notebook. The ZIP includes the source, dataset, authors' checkpoint, and historical result metadata. Upload only this input ZIP as a private Kaggle dataset, import the notebook, and attach that dataset. Enable Internet and a GPU, then run all cells. `RUN_FULL = True` runs two six-epoch, nine-user smoke checks before all ten full runs; `False` runs only the smoke checks.

Download both archives from `/kaggle/working/exports/`. Extract `adrec_baby_author_embeddings_results.zip` into `results/adrec_author_embeddings/baby/`, and `adrec_baby_author_embeddings_models.zip` into `checkpoints/adrec_author_embeddings/baby/`. The model archive contains ten run folders named `authors_warmup<W>_seed<S>`. Attach those folders and set `RESUME_ROOT` to their parent to resume in another Kaggle session. Completed runs are reused and incomplete runs resume from `latest.pt`.

Summaries compare the same paired seeds in all four conditions. The new runs hold pretrained embeddings fixed across seeds, while the historical project runs also varied pretraining. Their variability therefore has different sources. This experiment tests the released checkpoint under our existing trainer; it does not guarantee paper-level reproduction or isolate all differences in pretraining procedures. The shared training code and completed results remain unchanged. All ten full runs are complete. See the [comparison results](../results/adrec_author_embeddings/baby/accuracy_summary.csv) and [report discussion](../docs/MULTI_DATASET_TRAINING.md#report-section-baby-pretrained-embedding-comparison).
