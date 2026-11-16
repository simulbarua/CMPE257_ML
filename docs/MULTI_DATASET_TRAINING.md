# Training on Baby, Beauty, and MovieLens-100K

All three models use the same loader, sequence preparation, and original-style HR/NDCG evaluator. Train separate weights for each dataset. Results use `results/<model>/<dataset>/`, including Baby. Existing checkpoint locations are unchanged.

## Dataset files

`configs/datasets.json` records the upstream commit, download URL, SHA-256, and measured counts for each released file. The pinned commit is `0a30ef764cd85efa44f884a387198e9bf28c7d4c`. Use the authors' preprocessed `dataset.pkl`, not an unprocessed Amazon or MovieLens download.

| Selection | Users | Items | Train interactions | Validation targets | Test targets |
|---|---:|---:|---:|---:|---:|
| `baby` | 11,761 | 4,731 | 69,301 | 11,761 | 11,761 |
| `beauty` | 10,553 | 6,086 | 73,042 | 10,553 | 10,553 |
| `ml-100k` | 938 | 1,008 | 52,537 | 938 | 938 |

Counts above are measured from the actual released files; paper table counts can differ. Each file has aligned lists of user histories and one validation/test target per user. Item IDs are positive integers; zero is padding. Keep the released splits and ID mapping. No additional chronological sorting or ID conversion is needed.

## Shared API and compatibility

Existing calls keep using Baby:

```python
cfg = make_config('gru4rec', 'full')
```

New datasets use an optional argument:

```python
cfg = make_config('adrec', 'full', dataset='beauty')
cfg = make_config('sasrec', 'full', dataset='ml-100k')
```

The existing `train`, `load_dataset`, `SequenceDataset`, evaluation, and inference interfaces are unchanged. Baby's configuration dictionary and default hyperparameters remain unchanged, so old checkpoints, direct notebook calls, and resume configurations still work. New configurations include `dataset` and its pinned checksum. Item vocabulary size is derived from the complete selected dataset, even during smoke tests.

ADRec loads pretraining from the same dataset. Checkpoints containing configuration metadata are checked for matching dataset name and checksum as well as embedding dimensions. New datasets require that metadata. Legacy bare Baby embedding state dictionaries retain their existing dimension check for compatibility.

Pretraining, ADRec, SASRec+, and GRU4Rec use the existing shared training settings on the new datasets. These are controlled initial settings, not a claim that we have reproduced dataset-specific optimal paper results. Tune on validation only and record any changes. Model implementations, losses, optimizer, evaluation candidate rules, and averaging remain unchanged.

## Prepare Kaggle inputs

One code ZIP works for all teammates and datasets:

```bash
python3 -m scripts.package_kaggle --output-dir work/multi_dataset_upload
```

Attach that `cmpe257_kaggle_code.zip` and the selected datasets to Kaggle. Dataset files can be attached separately or kept in folders:

```text
baby/dataset.pkl
beauty/dataset.pkl
ml-100k/dataset.pkl
```

To package a single dataset alongside the code, use:

```bash
python3 -m scripts.package_kaggle --dataset-name beauty --dataset data/beauty/dataset.pkl --output-dir work/beauty_upload
python3 -m scripts.package_kaggle --dataset-name ml-100k --dataset data/ml-100k/dataset.pkl --output-dir work/ml100k_upload
```

The old Baby packaging command still works. Dataset selection is validated before creating the package. Avoid attaching several copies of the code ZIP: notebook source discovery requires one code bundle or an explicit `CODE_ZIP`/`CODE_DIR`.

## Run a notebook

- `notebooks/adrec_kaggle.ipynb`: independent pretraining and ADRec for seeds 42–46; the sweep remains inside the notebook.
- `notebooks/mahalakshmi_gru4rec_kaggle.ipynb`: preserves the existing first seed-42 run followed by seeds 43–46, with dataset-aware paths and metadata/model exports.
- `notebooks/sasrec_kaggle.ipynb`: neutral SASRec+ launcher using the shared code, with seed 42 followed by seeds 43–46. Anusha can also keep her own notebook and add the `dataset=` argument, matching checksum, and separate output paths.

Set this in the first setup cell:

```python
DATASET = 'beauty'  # 'baby', 'beauty', or 'ml-100k'
DATA_FILE = None
```

Input discovery selects the `dataset.pkl` with the matching checksum. If duplicate copies of the selected file are attached, set `DATA_FILE` to the intended path. An explicitly supplied file with the wrong checksum is rejected.

Run setup and dependencies, then choose `RUN_MODE = 'smoke'` for a small pipeline check. For a full run, start a fresh kernel, keep the selected dataset, set `RUN_MODE = 'full'`, enable a GPU, and run the notebook in order. Restart when changing dataset or source bundle to avoid stale imports and variables.

For ADRec, each new dataset/seed trains its own embeddings before ADRec. Baby embeddings cannot initialize Beauty or MovieLens. Baselines train directly without a separate pretraining stage. Baseline continuation cells use the chosen run mode, so a smoke execution does not unexpectedly start full training.

## Outputs and resume

Baby keeps its previous run and checkpoint paths. Result folders use model then dataset for all three datasets:

```text
/kaggle/working/runs/<dataset>/<run-name>/
/kaggle/working/results/<model>/<dataset>/
/kaggle/working/smoke_results/<model>/<dataset>/
/kaggle/working/exports/<dataset>/*.zip
```

Extract result archives locally into `results/<model>/<dataset>/`. Store model archives under `checkpoints/<dataset>/`, outside Git. Download the model ZIPs before ending the Kaggle session; results alone cannot restore weights.

CSV columns stay compatible with the existing summaries. Dataset labels and filenames identify the selected dataset, for example `adrec_beauty_runs.csv` and `sasrec_ml-100k_summary.csv`. Means and sample standard deviations are calculated separately by dataset/model. Do not pool metrics across datasets. Export validation rejects inconsistent dataset configurations and duplicate/conflicting seed results.

ADRec `RESUME_ROOT` points to the selected dataset's parent directory containing `adrec_full_seed<N>/`, as before. Configuration mismatches fail before resuming. For a baseline, configure the matching seed and mode and set `RESUME_CHECKPOINT` to that seed's `latest.pt` before running its single-model training cell. Only run the remaining-seeds cell for seeds that have not already completed; it does not automatically reuse completed baseline runs.

## Optional diffusion experiments

Completed Baby diffusion experiments do not need to be rerun. The diffusion notebook also accepts `DATASET`, but another dataset requires its own completed 32-step and pretraining checkpoints first. Set `CHECKPOINT_INPUT` explicitly to that dataset's downloaded baseline archives. Keep the ADRec result ZIP alongside the seed checkpoint ZIPs; the validator checks both.

New diffusion runs/results are under `adrec_diffusion_steps_<mode>/<dataset>/` and ZIPs under `exports/<dataset>/`. Locally, use `results/<dataset>/adrec_diffusion_steps/` and `checkpoints/<dataset>/adrec_diffusion_steps/`.

The strict source check permits one exact recorded Baby migration: the old training-module hash to the multi-dataset module hash in `configs/legacy_source_compatibility.json`. This change preserves Baby's training math and defaults and adds dataset selection/identity checks. All other source mismatches still fail. Accepted migrations are recorded in `baseline_source_migrations.json`. Updating training code again requires reviewing this compatibility entry; it is not a general bypass.

## Validation

```bash
python -m unittest discover -s tests -v
```

Checks cover old Baby configurations, resume and evaluation parity, dataset discovery, wrong-dataset embeddings, separate result exports, and exact source-migration matching. Local real-data smoke checks additionally exercise all four stages/model types on all three released files. ADRec GPU training is complete for seeds 42–46 on Baby, Beauty, and MovieLens-100K.

## Report section: Differences and limitations

We used the official ADRec model code and released datasets, with a shared script for training and evaluation. We kept the main hyperparameters and the released HR/NDCG evaluation procedure. For our original runs on each dataset, we trained embeddings and ADRec separately using seeds 42–46 and reported the mean and standard deviation. A later Baby experiment used the authors' released embeddings, as described below. The official quick-start uses supplied pretrained embeddings and also supports training new embeddings. The paper reports five runs but does not identify their seeds or the embedding checkpoint used for each run. [Paper](https://arxiv.org/html/2505.19544v1), [official instructions](https://github.com/Nemo-1024/ADRec#usage).

Our script keeps embeddings fully fixed during the five-epoch warm-up. The released optimizer can still update frozen embeddings through weight decay. We also use a separate random generator for data shuffling and reset the seed before final test sampling. These differences may affect the scores, but we have not measured their individual effects on accuracy. The paper's split description also differs from the released data, so we followed the released splits. Our results are therefore a reproduction with documented differences, rather than an exact match to the paper. We retained the completed runs without retraining to investigate these differences. [Released trainer](https://github.com/Nemo-1024/ADRec/blob/0a30ef764cd85efa44f884a387198e9bf28c7d4c/src/trainer.py), [gradient wrapper](https://github.com/Nemo-1024/ADRec/blob/0a30ef764cd85efa44f884a387198e9bf28c7d4c/src/pcgrad.py).

## Report section: Baby warm-up ablation

We tested whether the five-epoch warm-up stage helped ADRec on the Baby dataset. For each seed from 42 to 46, we loaded the same pretrained item embeddings used in the original run and trained a fresh ADRec model with `warmup_epochs = 0`. This allowed the embeddings and the backbone to train together from the first epoch. We did not repeat pretraining or initialize from a finished ADRec model. The other full-run settings stayed unchanged, including 32 diffusion steps, data splits, validation-based checkpoint selection, early stopping, and evaluation metrics. A smoke check confirmed that the embeddings changed during epoch zero.

The table compares our five-seed mean ± sample standard deviation with the published Baby results. All scores are percentages; differences are percentage points.

| Experiment | HR@20 | NDCG@20 |
|---|---:|---:|
| Our ADRec, five warm-up epochs | 6.2568 ± 0.2446 | 2.9065 ± 0.0509 |
| Our ADRec, no warm-up | 6.2597 ± 0.2379 | 2.8847 ± 0.0440 |
| Paper, with warm-up | 7.1524 | 3.1455 |
| Paper, without warm-up | 6.7507 | 2.9535 |

Without warm-up, HR@20 increased by only 0.0029 percentage points. Three seeds improved and two declined, so the average change was very small. NDCG@20 decreased by 0.0218 percentage points, about 0.75% relative to the original result, and declined in all five seeds. Average HR@5, HR@10, NDCG@5, and NDCG@10 also decreased slightly. These results suggest that warm-up had little effect on whether the target appeared in the top 20, but gave a small benefit to its position in the ranked list.

The paper reported larger decreases without warm-up: 0.4017 percentage points for HR@20 and 0.1920 for NDCG@20. Our results show the same direction for NDCG@20, but do not reproduce the paper's larger effect or its HR@20 decline. The published values are references, not additional project runs, and Table 4 does not report their standard deviations. [ADRec paper, Table 4 and Section 4.4](https://arxiv.org/html/2505.19544v1#S4.SS4).

We reused our historical baseline scores rather than retraining or re-evaluating them. The comparison uses the same five seeds and matching pretrained checkpoint hashes; the recorded runtimes use Tesla T4 GPUs and PyTorch 2.11.0+cu130. The source compatibility check permits only the previously audited Baby dataset-support migration. Our independently pretrained embeddings and documented differences from the released training code still limit direct comparisons with the paper. This experiment tests removal of warm-up in our implementation; it does not isolate the causes of the paper-to-project performance gap. The small changes across five seeds should not be treated as proof that warm-up is unnecessary on other datasets.

Results: [per-seed scores](../results/adrec_warmup_ablation/baby/accuracy_runs.csv), [mean and standard deviation](../results/adrec_warmup_ablation/baby/accuracy_summary.csv), and [paired differences](../results/adrec_warmup_ablation/baby/paired_differences.csv). The [experiment record](../results/adrec_warmup_ablation/baby/EXPERIMENT.md) preserves the run protocol and limitations.

## Report section: Baby pretrained-embedding comparison

We tested whether the initial item embeddings contributed to the difference between our Baby results and the paper. We loaded the authors' released Baby `pretrain.pth` and trained fresh ADRec models for seeds 42–46 with five and zero warm-up epochs. Only the item embeddings were transferred; each ADRec backbone started from a new initialization. All ten runs shared the same released checkpoint. We kept the other full-run settings unchanged, including 32 diffusion steps, data splits, validation-based checkpoint selection, early stopping, and the released evaluation procedure. Two smoke checks verified checkpoint loading and both warm-up schedules before full training.

We compared the new runs with our completed five-seed experiments using our own pretrained embeddings. The following project scores are mean ± sample standard deviation, expressed as percentages. The paper values are separate published references, not additional project runs.

| Embedding source | Warm-up epochs | HR@20 | NDCG@20 |
|---|---:|---:|---:|
| Our embeddings | 5 | 6.2568 ± 0.2446 | 2.9065 ± 0.0509 |
| Authors' embeddings | 5 | 6.5715 ± 0.0599 | 2.9499 ± 0.0307 |
| Our embeddings | 0 | 6.2597 ± 0.2379 | 2.8847 ± 0.0440 |
| Authors' embeddings | 0 | 6.5513 ± 0.2072 | 2.9131 ± 0.0862 |
| Paper | 5 | 7.1524 | 3.1455 |
| Paper | 0 | 6.7507 | 2.9535 |

With five warm-up epochs, using the authors' embeddings increased HR@20 by 0.3148 percentage points (5.03% relative) and NDCG@20 by 0.0434 points (1.49% relative). Both metrics improved in four of five seeds. Without warm-up, the gains were 0.2916 and 0.0283 points, respectively, also improving in four of five seeds. The gains were not consistent across all ranking cutoffs: average NDCG@5 decreased with the authors' embeddings under both warm-up settings. All six metrics are retained in the [comparison summary](../results/adrec_author_embeddings/baby/accuracy_summary.csv).

For the authors' embeddings, five warm-up epochs improved average HR@20 by 0.0202 points and NDCG@20 by 0.0368 points compared with zero warm-up. Both metrics improved in three of five seeds. Variation across seeds was lower with warm-up: HR@20 standard deviation fell from 0.2072 to 0.0599, and NDCG@20 from 0.0862 to 0.0307. These observations suggest a small benefit in our runs, but do not establish statistical significance or a consistent benefit for every seed. Our earlier experiment using our own embeddings also showed only a small average warm-up effect.

The paper reported larger warm-up gains of 0.4017 points in HR@20 and 0.1920 points in NDCG@20. Our five-epoch runs remained below the paper by 0.5809 and 0.1956 points, respectively. Our zero-epoch runs were closer to the corresponding paper ablation, with gaps of 0.1994 and 0.0404 points. We therefore improved our top-20 scores without reproducing the paper's absolute performance or the size of its warm-up benefit. Table 4 does not report standard deviations. [ADRec paper, Table 4](https://arxiv.org/html/2505.19544v1#S4.SS4).

This comparison has several limitations. The new seeds vary ADRec training with one fixed authors' embedding checkpoint, while our historical seeds also varied embedding pretraining. Their standard deviations therefore reflect different sources of randomness. The released checkpoint has no recorded pretraining seed, and the release does not establish which checkpoint was used for each paper run. We reused the historical scores and kept our existing trainer, including strict embedding freezing, the separate data-shuffling generator, and the final-test seed reset. The results suggest that the starting embeddings contributed to our performance gap, but cannot establish that the authors' pretraining procedure alone caused the improvement.

The [experiment records](../results/adrec_author_embeddings/baby/), [checkpoint provenance](../configs/author_baby_embeddings.json), and [Kaggle notebook](../notebooks/adrec_author_embeddings_kaggle.ipynb) document the completed runs. The Git results include scores, configurations, histories, source hashes, and checkpoint checksums. Duplicate baseline records and complete runtime package inventories remain in the local output archives; the original baselines are already stored under `results/adrec/baby/` and `results/adrec_warmup_ablation/baby/`. Model weights remain outside Git.
