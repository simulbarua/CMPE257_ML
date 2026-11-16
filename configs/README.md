# Shared configurations

- `shared.json`: existing Baby checksum, sequence length, seed, metric cutoffs, and evaluation batch size; retained for backward compatibility.
- `kaggle.json`: shared training defaults, unchanged by dataset selection.
- `datasets.json`: pinned dataset URLs, SHA-256 values, and counts measured from the released Baby, Beauty, and MovieLens-100K files.
- `legacy_source_compatibility.json`: exact audited old/new source hashes accepted when reusing existing Baby checkpoints in the diffusion notebook.

`make_config(model, mode, dataset="baby")` selects the checksum and model dataset name. Legacy calls without a dataset still return the original Baby settings. See [multi-dataset training](../docs/MULTI_DATASET_TRAINING.md).
