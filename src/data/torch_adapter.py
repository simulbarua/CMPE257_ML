"""Optional PyTorch integration. Requires torch in the training environment."""
import torch
from torch.utils.data import DataLoader, Dataset

class TorchSequenceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        row = self.dataset[index]
        return {key: torch.tensor(value, dtype=torch.bool if key == "attention_mask"
                                  else torch.long) for key, value in row.items()}

def make_loader(dataset, batch_size=128, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TorchSequenceDataset(dataset), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, num_workers=0)
