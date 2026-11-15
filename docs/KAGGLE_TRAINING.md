# ADRec Kaggle experiments

Use `notebooks/adrec_kaggle.ipynb` to train seeds 42–46. Each seed runs embedding pretraining and ADRec training independently. The notebook exports the same per-seed result and CSV summary format used by the baseline experiments.

## 1. Prepare the input

From the repository root, run:

```bash
python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl
```

Upload `work/kaggle_upload/cmpe257_kaggle_code.zip` and `work/kaggle_upload/dataset.pkl` to a private Kaggle dataset. Import `notebooks/adrec_kaggle.ipynb` into Kaggle and attach that dataset through **Add Input**. Use the current source ZIP with the shared training code. The seed loop and export functions are defined directly in notebook cells; no separate sweep module is required.

The setup accepts an intact ZIP or Kaggle-extracted source files. Automatic discovery requires one matching code source and one `dataset.pkl`. Set `CODE_ZIP` or `CODE_DIR`, plus `DATA_FILE`, when multiple sources are attached.

Enable Internet for installation and select a GPU for full training. The notebook uses one GPU. Run the setup, dependency, and runtime sections before training. Kaggle input and output behavior is described in [the notebook documentation](https://www.kaggle.com/docs/notebooks).

## 2. Configure the batch

The default configuration is:

```python
RUN_MODE = 'full'
SEEDS = [42, 43, 44, 45, 46]
RESUME_ROOT = None
```

A fresh batch trains all five seeds, including seed 42. It does not import the earlier seed-42 result. Preserve previous experiment folders separately. Each seed creates a separate `runs/adrec_full_seed<N>/` folder, with `pretrain/` and `adrec/` subfolders.

Full mode uses all 11,761 users, 4,731 products, hidden size 128, 32 diffusion steps, training and evaluation batch sizes 512, and up to 500 epochs per stage with early stopping. `configs/shared.json` fixes the data checksum, sequence length, metric cutoffs, and evaluation batch size. `configs/kaggle.json` supplies the training settings. The loop sets the selected seed in both stage configurations.

For a pipeline check, set `RUN_MODE = 'smoke'`. This uses nine users, hidden size 16, four diffusion steps, and two epochs per stage. Smoke results are saved under `smoke_results/adrec/` and are rejected when mixed with full results.

## 3. Run the training loop

Run **Define the result exports**, then **Train and export seeds 42–46**. The sweep cell creates both configurations for each seed and calls the shared `train()` function directly. For each seed, the notebook completes pretraining, ADRec alignment, fine-tuning, test evaluation, and export before starting the next seed.

Pretraining uses the original SASRec implementation in `pretrain` mode, without positional embeddings, and optimizes next-item cross-entropy loss. Validation selects the embedding checkpoint. Pretraining does not evaluate the test set.

ADRec loads only that seed's pretrained item embeddings. In full mode, embeddings remain frozen during epochs 0–4 and become trainable at epoch 5. Training then updates embeddings and backbone together. Recommendation and diffusion losses follow the existing shared training launcher. Validation selects the checkpoint; final evaluation uses the shared `adrec_upstream` protocol.

Each completed epoch writes `latest.pt`. Each completed seed exports metadata and creates `adrec_full_seed<N>.zip`. The combined result archive is updated after every seed, so completed runs can be saved before the full batch finishes.

To recover from an interruption in the same session, rerun the training cell. Completed stages are reused without retraining or re-evaluation; unfinished stages resume from `latest.pt`. Configuration mismatches are rejected. To recover in another session, use the resume steps below.

## 4. Inspect the CSV files

`results/adrec/adrec_baby_runs.csv` uses these columns:

```text
run_id,model,dataset,seed,best_epoch,epochs_completed,HR@5,HR@10,HR@20,NDCG@5,NDCG@10,NDCG@20
```

`results/adrec/adrec_baby_summary.csv` uses:

```text
model,dataset,n_runs,seeds,statistic,HR@5,HR@10,HR@20,NDCG@5,NDCG@10,NDCG@20
```

The summary reports `mean` and `std (sample, n-1)` over completed test results, rounded to four decimal places. A complete batch has `n_runs = 5` and `seeds = 42-46`. Partial batches list only completed seeds. A single completed run has a mean row and no standard-deviation row. Missing runs are never filled with repeated or invented values.

Different seeds change initialization, training order, dropout, and diffusion sampling. The mean summarizes performance across runs; standard deviation describes how much the scores vary. Keep data splits and evaluation settings fixed across all seeds and models. Select models using validation, and use test metrics for final reporting.

## 5. Download results and weights

Download these files from `/kaggle/working` before ending the session:

| Archive | Local destination | Contents |
|---|---|---|
| `adrec_results.zip` | `results/adrec/` | Per-seed JSON/text metadata, CSVs, experiment record, checkpoint hashes, and pretraining metadata |
| `adrec_full_seed42.zip` through `adrec_full_seed46.zip` | `checkpoints/adrec_full_seed<N>/` | The selected seed's complete `pretrain/` and `adrec/` output folders, including weights |

Each result export contains `adrec_seed<N>_results.json`, `adrec_seed<N>_config.json`, `adrec_seed<N>_history.json`, `adrec_seed<N>_environment.json`, `adrec_seed<N>_requirements-lock.txt`, and `adrec_seed<N>_checkpoints.json`. Pretraining metadata is under `pretrain/`. For compatibility with the baseline layout, `environment.json` and `requirements-lock.txt` are copies from the lowest listed seed; per-seed files preserve all environments.

`best.pt` contains validation-selected weights and settings for inference. `latest.pt` includes the optimizer, scheduler, random states, and last completed epoch for resuming. Keep weights outside Git. Share only the small result files and record the location of privately stored checkpoints. The original downloaded `checkpoints/adrec_full_run/` is separate from the new batch.

Run the inference cell to reload a completed model and generate recommendations from visible history. The example excludes padding ID 0; benchmark evaluation keeps the upstream candidate rules.

## 6. Resume in a new Kaggle session

Save the full run directories, including any unfinished stage, before ending the session. Attach them in the new session and set `RESUME_ROOT` to the folder containing:

```text
adrec_full_seed42/
    pretrain/
    adrec/
adrec_full_seed43/
    pretrain/
    adrec/
...
```

Retain the original configuration and run the notebook from setup. Complete stages are copied into the working directory and reused. Incomplete stages resume from the last completed epoch. Seeds with no saved stages start from scratch. An early-stopped stage stays stopped. Cross-hardware training resume is not guaranteed to be numerically identical.

**Save Version → Save & Run All** starts a separate background execution from the first cell. It uses the attached inputs and `RESUME_ROOT`; it does not inherit an interactive session's unsaved files. Preserve checkpoints before ending or replacing an interactive session.

## Implementation and evaluation protocol

Original model files and licenses in `third_party/ADRec` remain unchanged. GRU4Rec uses the validated RecBole v1.2.0 CE adapter; it is **not verified as the exact GRU4Rec implementation/configuration used for the ADRec paper**.

The project training code consists of `src/training/kaggle.py`, the packaging script, configs, and launch notebooks. Training follows upstream Adam, its PCGrad wrapper/gradient combination, cosine scheduler with `T_max=500`, five-epoch ADRec alignment, and validation when a zero-based epoch is positive and divisible by five. As upstream, improvement in any tracked validation metric selects the latest improving checkpoint; metric-wise best values can come from different epochs. The shared evaluation function still uses original-style CPU top-k, full score columns, unweighted batch means, percentages, and four-decimal rounding.

These are explicit runner differences from upstream:

- Seed 42 is read from the shared configuration (upstream YAML defaults to 2025).
- Checkpoints, environment records, resume support, and history-only inference are added. A checkpoint is selected even if all initial metrics are zero. Short runs get final-epoch validation if no scheduled validation occurred.
- Saved best models are evaluated in evaluation mode, avoiding dropout during test inference. Final ADRec sampling uses the recorded seed; it is stochastic, so this does not imply bitwise equality with an upstream run's random stream.
- During ADRec alignment the runner clears the frozen embedding's gradient before stepping. The upstream PCGrad wrapper can assign zero gradients even to frozen parameters, allowing Adam weight decay to change them. This adjustment enforces the intended freeze. Early stopping cannot prevent the fine-tuning phase from starting.
- An explicit checkpoint path replaces the upstream hard-coded pretraining path. The architecture and embedding initialization are unchanged.

Metric parity on identical scores is already covered by the shared tests. Equal metric code does not establish exact paper reproduction or equal trained scores. Record any new settings and compare completed full runs fairly.

## Diffusion-step comparison

`notebooks/adrec_diffusion_steps_kaggle.ipynb` is a separate experiment using the completed five-seed run. It trains 8- and 16-step ADRec models from each seed's saved `pretrain/best.pt` and loads the corresponding 32-step `adrec/best.pt` as the baseline. The original reproduction results and checkpoints remain unchanged.

1. Build the current code bundle using `python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl`.
2. Attach the code ZIP, Baby dataset, and `checkpoints/adrec_all_results_and_models.zip` as private Kaggle inputs. Uploading the checkpoint archive is a separate step; the code packager excludes model weights.
3. Import the diffusion-step notebook and run its numbered sections. Input validation checks checkpoint hashes, saved configurations, dataset identity, and relevant source files before training.
4. Use smoke mode for a pipeline check. It retains the existing full-sized embeddings but runs two epochs on nine users; its scores are not comparable with the full experiment.
5. In full mode, train ten new ADRec runs. All three step counts are evaluated and timed in one comparison session. The timing cell may be rerun to create another separate session.
6. Download the metadata and model ZIPs. Per-run ZIPs and metadata archives are also created after each completed training run. Resume in a new session by attaching the extracted model output and setting `RESUME_ROOT` to the directory containing `runs/`.

The authors' `trunc_lin` schedule changes its endpoint formula for 8 and 16 steps. `noise_schedules.json` records the actual beta endpoints and terminal alpha product. This experiment measures configurations under that schedule, not a pure iteration-count effect with a fixed noise trajectory.

Latency includes denoising, full-catalog scoring, and device top-20 selection on preloaded histories. Model loading, padding, transfers, and application overhead are excluded. Warm-up calls are excluded, CUDA is synchronized before and after each measurement, and batch sizes 1 and 32 are measured separately. Raw timings, input hashes, checkpoint hashes, runtime details, per-seed scores, and mean/sample-standard-deviation summaries are saved. Speedups use the newly measured 32-step baseline from the same session. Original saved accuracy metrics are retained alongside the new evaluations. Timings from different sessions are not pooled.

The predefined step counts use the same splits and evaluation protocol. Checkpoints are selected using validation; test metrics are used for final reporting rather than automatic configuration selection. No training-time speedup is claimed.

## Validation commands

Run the checks from the repository root:

```bash
python -m unittest discover -s tests -v
python -m scripts.package_kaggle --dataset data/baby/dataset.pkl
```

Unit tests cover training and resume behavior, inference, CSV schema, sample standard deviation, partial summaries, conflicting results, completed-stage reuse, and independent pretraining/ADRec seed settings. To check the complete notebook on Kaggle, set `RUN_MODE = 'smoke'` and run its cells in order. Smoke checks do not constitute full five-seed GPU training.
