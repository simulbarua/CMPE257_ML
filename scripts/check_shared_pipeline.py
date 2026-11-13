"""Run from repository root: python -m scripts.check_shared_pipeline --dataset PATH."""
import argparse
import json
from pathlib import Path
from src.data.load import load_dataset, dataset_summary
from src.data.datasets import SequenceDataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--config", default="configs/shared.json")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
data = load_dataset(args.dataset, config["dataset_sha256"])
print(json.dumps(dataset_summary(data), indent=2))
for split in ("train", "val", "test"):
    ds = SequenceDataset(data, split, max_len=config["max_len"])
    for row in ds:
        assert len(row["input_ids"]) == len(row["labels"]) == config["max_len"]
        assert len(row["attention_mask"]) == config["max_len"]
    print(f"{split}: checked {len(ds)} examples; first target={ds[0]['target_id']}")
