"""Team implementation of the upstream sequence contract.

Reference: Nemo-1024/ADRec src/utils.py at 0a30ef764cd85efa44f884a387198e9bf28c7d4c.
Returns plain Python records; the optional torch adapter creates tensors.
"""
from .load import validate_dataset

class SequenceDataset:
    def __init__(self, data, split, max_len=50, target_mode="per_token"):
        validate_dataset(data)
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        if type(max_len) is not int or max_len < 1:
            raise ValueError("max_len must be a positive integer")
        if target_mode not in ("per_token", "last"):
            raise ValueError("target_mode must be per_token or last")
        self.data, self.split = data, split
        self.max_len, self.target_mode = max_len, target_mode

    def __len__(self):
        return len(self.data["train"])

    def __getitem__(self, index):
        history = list(self.data["train"][index])
        if self.split == "train":
            targets = history[1:]
            history = history[:-1]
        else:
            if self.split == "test":
                history += self.data["val"][index]
            targets = history[1:] + self.data[self.split][index]
        history, targets = history[-self.max_len:], targets[-self.max_len:]
        padding = self.max_len - len(history)
        if self.target_mode == "last":
            labels = [0] * (self.max_len - 1) + [targets[-1]]
        else:
            labels = [0] * padding + targets
        return {"user_index": index,
                "input_ids": [0] * padding + history,
                "attention_mask": [False] * padding + [True] * len(history),
                "labels": labels,
                "target_id": targets[-1]}
