# Run the team's Kaggle notebooks

| Team member | Notebook | Training responsibility |
|---|---|---|
| Simul | `notebooks/simul_adrec_kaggle.ipynb` | Pretrain Baby item embeddings, align ADRec with frozen embeddings, then fine-tune |
| Anusha | `notebooks/anusha_gru4rec_kaggle.ipynb` | Train the RecBole-derived GRU4Rec CE baseline |
| Mahalakshmi | `notebooks/mahalakshmi_sasrec_kaggle.ipynb` | Train the authors' SASRec implementation as our SASRec+ baseline |

The notebook is the launcher. It imports the shared Python modules from the code ZIP attached to Kaggle. You need **the notebook, the code ZIP, and the Baby dataset**. You do not need to copy Python scripts into individual notebook cells.

## 1. Prepare the input once on your computer

From the repository root, run:

```bash
python3 -m scripts.package_kaggle --dataset data/baby/dataset.pkl
```

This creates `work/kaggle_upload/cmpe257_kaggle_code.zip` and copies the checked dataset to `work/kaggle_upload/dataset.pkl`. These stay outside Git. The ZIP contains only the selected source, model licenses, configs, notebooks, requirements, and this guide. It excludes Git history, credentials, personal proposal archives, model weights, and unrelated analysis files.

The ZIP uses the **current local files**, so it works before this branch is pushed. Rebuild it after changing training code. Its manifest records the base Git commit, branch, and individual file hashes; the base commit alone does not describe uncommitted work.

Create a **private Kaggle dataset** containing the two files and share it with the team, or send the two files to each teammate to upload privately. No GitHub credentials are needed. The same ZIP and Baby file work for all three notebooks.

## 2. Open your notebook on Kaggle

1. Create a Kaggle notebook and import your assigned `.ipynb` file.
2. Attach the private dataset containing the ZIP and `dataset.pkl` through **Add Input** (the exact UI label can vary).
3. Enable Internet for dependency installation. Enable a GPU for full training; the notebooks use a single GPU even when more than one is available.
4. Run the cells from top to bottom. The setup cell finds the files automatically if there is exactly one matching file. Otherwise set `CODE_ZIP` and `DATA_FILE` to their paths under `/kaggle/input`.
5. Let the installation cell finish before importing models. If the kernel requests a restart, restart and run again from the top. The code preserves Kaggle's installed CUDA PyTorch when it satisfies the requirements; the actual environment is recorded for each run.

Kaggle supports attached data and saves notebook outputs from `/kaggle/working`; **Save & Run All** runs the notebook again in a separate session. See [Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks).

## 3. Start with a small run

Leave `RUN_MODE = 'smoke'`. This uses the first nine aligned users, the complete item vocabulary, hidden size 16, four diffusion steps, batch size four, and two epochs per stage. ADRec performs one frozen epoch and one fine-tuning epoch. It exercises training, validation, test evaluation, saving, and reloading. These numbers are **not benchmark results**.

Check that losses are finite, checkpoints appear, and the inference cell returns product IDs. Zero HR/NDCG in this tiny run is not itself a pipeline failure.

Then select `RUN_MODE = 'full'` and use a new `RUN_NAME`. Full mode uses all 11,761 users, hidden size 128, 32 diffusion steps, train/evaluation batch sizes 512, and up to 500 epochs with early stopping. Simul has a separate pretraining run and ADRec run, each with its own maximum. These are initial experiment settings, not a guarantee of the paper's scores. Full training requires a GPU; runtime and memory usage still need measurement on Kaggle.

All members share `configs/shared.json`: data checksum, sequence length 50, seed 42, evaluation batch size 512, and HR/NDCG cutoffs 5/10/20. `configs/kaggle.json` holds common starting training settings. GRU4Rec trains one final target per user; SASRec+ and ADRec use per-token targets, matching the existing adapters. Do not silently change the data split or evaluator for a model.

If a training batch runs out of memory, choose a smaller **training** batch size for a new run and record that change. Do not change evaluation batch size casually: upstream averages batch means, so batch boundaries affect reported metrics. The notebooks do not enable mixed precision or multi-GPU training; these would need separate validation.

## 4. Simul's ADRec stages

The notebook first trains the vendored `pretrain` model, which omits SASRec position embeddings. Its validation-selected `best.pt` supplies only item embeddings to ADRec. This is distinct from Mahalakshmi's SASRec+ model. Pretraining does not inspect test results.

ADRec keeps embeddings frozen for epochs 0–4, then unfreezes them at epoch 5. The optimizer and scheduler continue through that transition. If you set `PRETRAINED_CHECKPOINT`, pretraining is skipped; use only a trusted, compatible Baby checkpoint with 4,732 embedding rows, hidden size 128 for full runs, and the same product-ID mapping. Both the authors' plain state dictionary and this launcher's `best.pt` format are accepted. Shape checks cannot verify an external checkpoint's item mapping; confirm its provenance yourself. No weights are downloaded automatically. A smoke checkpoint is not compatible with a full run.

## 5. Keep results and checkpoints

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

Download these files or preserve them with a Kaggle notebook version before leaving a session. `latest.pt` updates at each completed epoch, but a file on a temporary VM is not a durable backup. **Save & Run All** executes the cells again; it does not simply download your current interactive run. Keep model weights outside Git. Each teammate should contribute their small results/configs and model discussion through their own branch later.

The example inference cell reloads `best.pt` and predicts from a visible history, without passing the held-out target. It hides padding ID 0 for the demo only. Evaluation retains upstream candidates. The dataset has IDs, not product names/images.

## 6. Resume a saved run

Attach the saved `latest.pt` as another Kaggle input. Set `RESUME_CHECKPOINT` to its path, restore the same run mode/settings, and choose a new writable output folder. For Simul's pretraining stage use `PRETRAIN_RESUME`; resuming ADRec itself skips pretraining. A completed epoch is the restart boundary.

Only the total `epochs` setting may increase when resuming; incompatible configurations fail before training. Optimizer, scheduler, data shuffle, and random states are restored. Different GPU/library versions can still change numerical results. A run that has early-stopped stays stopped.

For long runs split into sessions, use `FINALIZE = False` until the final run so intermediate chunks do not inspect the test set. Set an epoch cap that allows scheduled validation (for example 51 epochs reaches zero-based epoch 50). For pretraining, run its cell until that stage is complete before starting the ADRec cell. Changing an epoch cap before the first scheduled validation can trigger the short-run validation fallback described below, so log session boundaries.

After training is complete, you may resume with the same epoch cap and `FINALIZE = True` to load the saved best model and evaluate without additional training. Do not choose settings based on repeated test results.

## What is original and what we added

Original model files and licenses in `third_party/ADRec` remain unchanged. GRU4Rec is our previously validated RecBole v1.2.0 CE adapter; it is **not verified as the exact GRU4Rec implementation/configuration used for the ADRec paper**.

The new team code is `src/training/kaggle.py`, the packaging script, configs, and launch notebooks. Training follows upstream Adam, its PCGrad wrapper/gradient combination, cosine scheduler with `T_max=500`, five-epoch ADRec alignment, and validation when a zero-based epoch is positive and divisible by five. As upstream, improvement in any tracked validation metric selects the latest improving checkpoint; metric-wise best values can come from different epochs. The shared evaluation function still uses original-style CPU top-k, full score columns, unweighted batch means, percentages, and four-decimal rounding.

These are explicit runner differences from upstream:

- Team seed 42 is read from the shared configuration (upstream YAML defaults to 2025).
- Checkpoints, environment records, resume support, and history-only inference are added. A checkpoint is selected even if all initial metrics are zero. Short runs get final-epoch validation if no scheduled validation occurred.
- Saved best models are evaluated in evaluation mode, avoiding dropout during test inference. Final ADRec sampling uses the recorded team seed; it is stochastic, so this does not imply bitwise equality with an upstream run's random stream.
- During ADRec alignment the runner clears the frozen embedding's gradient before stepping. The upstream PCGrad wrapper can assign zero gradients even to frozen parameters, allowing Adam weight decay to change them. This adjustment enforces the intended freeze. Early stopping cannot prevent the fine-tuning phase from starting.
- An explicit checkpoint path replaces the upstream hard-coded pretraining path. The architecture and embedding initialization are unchanged.

Metric parity on identical scores is already covered by the shared tests. Equal metric code does not establish exact paper reproduction or equal trained scores. Record any new settings and compare completed full runs fairly.

## Local validation

Validated on the local CPU environment on November 13, 2026:

- All 16 unit tests passed, including unchanged shared evaluation tests.
- All three notebooks executed every code cell against the real Baby dataset and packaged source, with only Kaggle paths redirected and the pip-install cell skipped (dependencies were already installed). Pretraining, ADRec warmup/fine-tuning, both baselines, evaluation, saving, and inference all completed.
- Synthetic training tests checked exact equality of resumed versus uninterrupted GRU4Rec and ADRec runs, frozen embeddings during alignment and changed embeddings during fine-tuning, rejection of incompatible resumes/embedding dimensions, and history-only ADRec scores matching the label-conditioned upstream inference call under the same seed.

Reproduce these checks from the repository root with your installed dependencies:

```bash
python -m unittest discover -s tests -v
python -m scripts.package_kaggle --dataset data/baby/dataset.pkl
python -m scripts.check_kaggle_notebooks --output-dir work/a-new-check-folder
```

The notebook checker writes logs and `summary.json` under the chosen new directory. It does not modify notebook outputs. Kaggle's dependency installation, GPU runtime, full-duration training, and cross-hardware resume reproducibility have not yet been tested.
