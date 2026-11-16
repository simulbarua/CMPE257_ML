"""Pinned released datasets and Kaggle input discovery (no downloads)."""
import hashlib
import json
from pathlib import Path

CATALOG = Path(__file__).resolve().parents[2] / "configs/datasets.json"


def dataset_spec(name="baby"):
    catalog = json.loads(CATALOG.read_text())
    if name not in catalog["datasets"]:
        raise ValueError(f"Unknown dataset {name!r}; choose baby, beauty, or ml-100k")
    return dict(catalog["datasets"][name])


def find_dataset(input_root, name="baby", override=None):
    """Resolve by the agreed checksum, even when several dataset.pkl files exist."""
    expected = dataset_spec(name)["sha256"]
    if override is not None:
        candidates = [Path(override)]
    else:
        candidates = sorted(Path(input_root).rglob("dataset.pkl"))
    matches = [p for p in candidates if p.is_file()
               and hashlib.sha256(p.read_bytes()).hexdigest() == expected]
    if len(matches) != 1:
        raise ValueError(f"Expected one verified {name} dataset.pkl; found {matches}. "
                         "Attach the selected dataset or set DATA_FILE to its exact path.")
    return matches[0]
