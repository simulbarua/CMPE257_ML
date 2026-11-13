"""Load the released ADRec list-only pickle without importing pickle globals."""
import hashlib
import io
import pickle
from pathlib import Path

BABY_SHA256 = "bf8f3485f9e36763e07f0234e0ddb479cb023358797615388a36f0d7238cea99"

class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Dataset must contain built-in data, not pickle globals")

def validate_dataset(data):
    if not isinstance(data, dict) or set(data) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test lists")
    if any(not isinstance(data[k], list) for k in data):
        raise ValueError("Splits must be lists")
    n = len(data["train"])
    if not n or any(len(data[k]) != n for k in data):
        raise ValueError("Splits must have the same nonzero number of users")
    for split, rows in data.items():
        for row in rows:
            if not isinstance(row, list) or not row:
                raise ValueError("Every row must be a nonempty list")
            if any(type(item) is not int or item < 1 for item in row):
                raise ValueError("Item IDs must be positive integers; 0 is reserved")
            if split == "train" and len(row) < 2:
                raise ValueError("Training histories need at least two items")
            if split != "train" and len(row) != 1:
                raise ValueError("Validation/test need one target per user")
    return data

def load_dataset(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Dataset checksum does not match the agreed version")
    return validate_dataset(_DataOnlyUnpickler(io.BytesIO(raw)).load())

def dataset_summary(data):
    validate_dataset(data)
    items = {i for rows in data.values() for row in rows for i in row}
    return {"users": len(data["train"]), "unique_items": len(items),
            "max_item_id": max(items),
            "interactions": {k: sum(map(len, v)) for k, v in data.items()}}
