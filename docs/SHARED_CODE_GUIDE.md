# Shared data and evaluation walkthrough

This branch supplies common infrastructure and local model smoke checks, not full training launchers or Kaggle notebooks yet. All three notebooks will import these modules after downloading this repository. No GPU is required for the core checks.

## 1. `configs/shared.json` — agree on settings

Contains the exact released Baby file checksum, maximum sequence length (50), seed, metric cutoffs, and evaluation batch size. Read this file in each notebook; do not independently change split definitions. The dataset is attached separately to Kaggle and its path is passed explicitly.

## 2. `src/data/load.py` — read and check the data

`load_dataset(path, expected_sha256)` reads the supplied pickle and validates aligned train/val/test lists, positive integer item IDs, and one held-out target per user. Its restricted unpickler rejects imported classes. It does not sort, remap IDs, or create new splits. Zero is reserved for padding.

The verified file has 11,761 users, 4,731 products, and 92,823 interactions. The paper's 92,613 count differs. Preserve the checksum and disclose that difference. Timestamps and product metadata are absent.

## 3. `src/data/datasets.py` — turn sequences into examples

`SequenceDataset(data, split, max_len, target_mode)` returns dictionaries containing `input_ids`, `attention_mask`, `labels`, `target_id`, and `user_index`.

For training history `[2588,1310,1]`, validation `[1082]`, test `[1283]`:

- Training: input `[2588,1310]`, shifted labels `[1310,1]`.
- Validation: input `[2588,1310,1]`, next target `1082`.
- Test: input `[2588,1310,1,1082]`, next target `1283`.

Input and shifted-label sequences are truncated together from the left, then left-padded with zeros. `per_token` supervises all nonpadding positions; `last` supervises only the final position. The model's loss must ignore zero labels. Full-catalog evaluation uses only `target_id`.

This implements the basic upstream loader contract independently. It does not implement DiffuRec's optional prefix-expansion mode; that is not a shared requirement for the three selected models. Model integration must verify the intended training mode.

## 4. `src/data/torch_adapter.py` — create Kaggle batches

Converts the plain records to PyTorch tensors and creates a DataLoader. IDs and labels are `long`; masks are Boolean, with True meaning a real token. Attention APIs may expect the inverse mask, so each model adapter must translate it. Training can shuffle users, but must never shuffle items inside their histories. Validation and test loaders should not shuffle.

From the repository root in Kaggle:

```python
import json
from src.data import load_dataset, SequenceDataset
from src.data.torch_adapter import make_loader
from src.utils.reproducibility import seed_everything

cfg = json.load(open('configs/shared.json'))
seed_everything(cfg['seed'])
data = load_dataset('/kaggle/input/YOUR_ATTACHMENT/dataset.pkl', cfg['dataset_sha256'])
train = SequenceDataset(data, 'train', cfg['max_len'], target_mode='per_token')
loader = make_loader(train, batch_size=128, shuffle=True, seed=cfg['seed'])
batch = next(iter(loader))
print(batch['input_ids'].shape)  # (128, 50)
```

The notebook is the launcher. The imported Python modules are downloaded from GitHub along with the notebook's supporting code.

## 5. `src/evaluation/upstream_metrics.py` — original HR/NDCG

Retains the authors' metric functions. `metrics.py` and `RankingMetrics` remain only for the optional per-user alternative.

## 6. `src/evaluation/evaluate.py` — original evaluation by default

`evaluate` now uses upstream CPU torch.topk behavior without added padding/seen-item filtering, averages batch means, and returns percentages rounded to four decimals. Use the same score tensors and batch size as upstream; default batch size is 512. There is no custom tie-breaking.

Supply `predict(histories, masks)` returning a tensor shaped `[batch, 4732]` for Baby. Preserve score dtype. Do not pass test labels to prediction. The local smoke runner demonstrates how to create ADRec's shifted conditioning sequence from visible history without its held-out answer.

```python
from src.evaluation import evaluate
test = SequenceDataset(data, 'test', cfg['max_len'])
# After implementing the trained-model predict adapter:
# report = evaluate(test, predict, num_items=4731,
#                   batch_size=cfg['evaluation_batch_size'], ks=cfg['metric_ks'])
```

The previous behavior is explicitly named `evaluate_user_weighted`: padding excluded, item-ID tie-breaking, user-weighted fractions. Do not mix its output with reproduction percentages.

See [local validation](LOCAL_VALIDATION.md) for actual model smoke checks and limitations.

## 7. `scripts/check_shared_pipeline.py` and `tests/test_shared.py`

```bash
python -m unittest discover -s tests -v
python -m scripts.check_shared_pipeline --dataset data/baby/dataset.pkl
```

The first checks small examples with known answers. The second verifies the pinned real file and constructs every train/val/test example. It does not train a model. Run from the repository root.

## Next steps and ownership

- Anusha: review data examples and use this loader in the GRU4Rec integration.
- Mahalakshmi: review metrics and connect SASRec+ scores to `evaluate`.
- Simul: connect ADRec's training stages and score output to these interfaces.
- Together: verify one batch per model before full Kaggle runs.

Local model integration checks now exist, but full training launchers and the three Kaggle notebooks remain next steps.

## Provenance

Contracts checked against https://github.com/Nemo-1024/ADRec/tree/0a30ef764cd85efa44f884a387198e9bf28c7d4c (src/utils.py, src/trainer.py, src/metrics.py). Original model sources are now retained under third_party/ADRec with their license and pinned provenance. The provided dataset remains ignored by Git.
