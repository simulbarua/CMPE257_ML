# Kaggle training instructions

| Model | Notebook | Training stages |
|---|---|---|
| ADRec | `notebooks/adrec_kaggle.ipynb` | Pretrain Baby item embeddings, align ADRec with frozen embeddings, then fine-tune |
| GRU4Rec | `notebooks/gru4rec_kaggle.ipynb` | Train the RecBole-derived GRU4Rec CE baseline |
| SASRec+ | `notebooks/sasrec_kaggle.ipynb` | Train the original SASRec implementation included in ADRec |

Each notebook imports the shared Python modules from the code attached to Kaggle. Attach the code ZIP and the Baby dataset before running the notebook. The Python scripts remain in the project folders and are imported by the notebook cells.

## 1. Prepare the input files

From the repository root, run:

```bash
python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl
```

This creates `work/kaggle_upload/cmpe257_kaggle_code.zip` and copies the checked dataset to `work/kaggle_upload/dataset.pkl`. These stay outside Git. The ZIP contains only the selected source, model licenses, configs, notebooks, requirements, and this guide. It excludes Git history, credentials, personal proposal archives, model weights, and unrelated analysis files.

The ZIP uses the **current local files**, so it works before this branch is pushed. Rebuild it after changing training code. Its manifest records the base Git commit, branch, and individual file hashes; the base commit alone does not describe uncommitted work.

Create a private Kaggle dataset containing the two files. Grant dataset access to collaborators through its sharing settings. All three notebooks use the same code ZIP and Baby dataset.

## 2. Configure Kaggle

1. Create a Kaggle notebook and import the `.ipynb` file for the selected model.
2. Attach the private dataset containing the ZIP and `dataset.pkl` through **Add Input** (the exact UI label can vary).
3. Enable Internet for dependency installation. Enable a GPU for full training; the notebooks use a single GPU even when more than one is available.
4. Run the cells from top to bottom. The setup cell finds the files automatically if there is exactly one matching file. Kaggle may automatically unpack the ZIP; the notebook also discovers the extracted `bundle_manifest.json` and copies its listed code files. If automatic discovery is ambiguous, set `CODE_ZIP` (for an intact ZIP) or `CODE_DIR` (the extracted folder containing the manifest), plus `DATA_FILE`, to their paths under `/kaggle/input`.
5. Let the installation cell finish before importing models. If the kernel requests a restart, restart and run again from the top. The code preserves Kaggle's installed CUDA PyTorch when it satisfies the requirements; the actual environment is recorded for each run.

Kaggle supports attached data and saves notebook outputs from `/kaggle/working`; **Save & Run All** runs the notebook again in a separate session. See [Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks).

## 3. Select the run mode

Leave `RUN_MODE = 'smoke'`. This uses the first nine aligned users, the complete item vocabulary, hidden size 16, four diffusion steps, batch size four, and two epochs per stage. ADRec performs one frozen epoch and one fine-tuning epoch. It exercises training, validation, test evaluation, saving, and reloading. These numbers are **not benchmark results**.

Check that losses are finite, checkpoints appear, and the inference cell returns product IDs. Zero HR/NDCG in this tiny run is not itself a pipeline failure.

Then select `RUN_MODE = 'full'` and use a new `RUN_NAME`. Full mode uses all 11,761 users, hidden size 128, 32 diffusion steps, train/evaluation batch sizes 512, and up to 500 epochs with early stopping. ADRec has separate pretraining and diffusion training runs, each with its own epoch limit. Full training requires a GPU. These settings define the starting configuration; they do not establish reproduction of the paper's reported scores.

All notebooks use `configs/shared.json`: data checksum, sequence length 50, seed 42, evaluation batch size 512, and HR/NDCG cutoffs 5/10/20. `configs/kaggle.json` holds common starting training settings. GRU4Rec trains one final target per user; SASRec+ and ADRec use per-token targets, matching the existing adapters. Retain the same data splits and evaluation protocol across models.

If a training batch runs out of memory, choose a smaller **training** batch size for a new run and record that change. Retain the evaluation batch size: upstream averages batch means, so batch boundaries affect reported metrics. Training uses one GPU without mixed precision.

## 4. Run the ADRec training stages

The notebook first trains the vendored `pretrain` model, which omits SASRec position embeddings. Its validation-selected `best.pt` supplies only item embeddings to ADRec. The SASRec+ baseline includes positional embeddings and is trained separately. Pretraining does not inspect test results.

ADRec keeps embeddings frozen for epochs 0–4, then unfreezes them at epoch 5. The optimizer and scheduler continue through that transition. Set `PRETRAINED_CHECKPOINT` to skip pretraining and load a compatible Baby checkpoint with 4,732 embedding rows, hidden size 128 for full runs, and the same product-ID mapping. Both the authors' plain state dictionary and this launcher's `best.pt` format are accepted. Shape checks cannot verify an external checkpoint's item mapping; confirm its provenance before loading. No weights are downloaded automatically. A smoke checkpoint is not compatible with a full run.

## 5. Save results and checkpoints

Each stage writes to `/kaggle/working/runs/<RUN_NAME>/<stage>/`:

| File | Purpose |
|---|---|
| `best.pt` | Validation-selected weights and model settings for inference |
| `latest.pt` | Most recent completed epoch, optimizer, scheduler, best weights, random states, and loader state for resuming |
| `config.json` | Selected settings and dataset checksum |
| `environment.json` | Python/PyTorch, GPU, CUDA, and source snapshot |
| `requirements-lock.txt` | Installed Python package versions |
| `history.json` | Epoch losses, embedding freeze status, and validation results |
| `results.json` | Completion status, chosen epoch, and final test metrics when requested |

Download these files or preserve them with a Kaggle notebook version before leaving a session. `latest.pt` updates at each completed epoch, but a file on a temporary VM is not a durable backup. **Save & Run All** executes the cells again; it does not download the existing interactive run. Keep model weights outside Git. Version the small result and configuration files separately from the model weights.

The example inference cell reloads `best.pt` and predicts from a visible history, without passing the held-out target. It hides padding ID 0 for the demo only. Evaluation retains upstream candidates. The dataset has IDs, not product names/images.

## 6. Resume a saved run

Attach the saved `latest.pt` as another Kaggle input. Set `RESUME_CHECKPOINT` to its path, restore the same run mode/settings, and choose a new writable output folder. For embedding pretraining, set `PRETRAIN_RESUME`; resuming ADRec itself skips pretraining. A completed epoch is the restart boundary.

Only the total `epochs` setting may increase when resuming; incompatible configurations fail before training. Optimizer, scheduler, data shuffle, and random states are restored. Different GPU/library versions can still change numerical results. A run that has early-stopped stays stopped.

For long runs split into sessions, use `FINALIZE = False` until the final run so intermediate chunks do not inspect the test set. Set an epoch cap that allows scheduled validation (for example 51 epochs reaches zero-based epoch 50). For pretraining, run its cell until that stage is complete before starting the ADRec cell. Changing an epoch cap before the first scheduled validation can trigger the short-run validation fallback described below, so log session boundaries.

To evaluate a completed run without additional training, resume with the same epoch cap and `FINALIZE = True`. Do not choose settings based on repeated test results.

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

## Local validation

Validated on the local CPU environment on November 13, 2026:

- All 16 unit tests passed, including unchanged shared evaluation tests.
- All three notebooks executed every code cell against the real Baby dataset and packaged source, with only Kaggle paths redirected and the pip-install cell skipped (dependencies were already installed). Pretraining, ADRec warmup/fine-tuning, both baselines, evaluation, saving, and inference all completed.
- Synthetic training tests checked exact equality of resumed versus uninterrupted GRU4Rec and ADRec runs, frozen embeddings during alignment and changed embeddings during fine-tuning, rejection of incompatible resumes/embedding dimensions, and history-only ADRec scores matching the label-conditioned upstream inference call under the same seed.

Run the validation commands from the repository root after installing the dependencies:

```bash
python -m unittest discover -s tests -v
python -m scripts.package_kaggle --dataset data/baby/dataset.pkl
python -m scripts.check_kaggle_notebooks --output-dir work/a-new-check-folder
```

The notebook checker writes logs and `summary.json` under the chosen new directory. It does not modify notebook outputs. A full ADRec run completed on a Kaggle Tesla T4. Its saved checkpoints loaded locally on CPU, and the example history produced the same ten product IDs. Full GRU4Rec and SASRec+ runs on Kaggle and cross-hardware training resume have not been verified.
