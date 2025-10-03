import torch
from torch.utils.data import Dataset

class ProportionalDataset(Dataset):
    def __init__(self, components, proportions):

        assert set(components) == set(proportions), "Components and proportions must have identical keys"

        total = sum(proportions.values())
        assert abs(total - 1.0) < 0.0001, f"Proportions must sum to 1.0 (got {total})"

        self.names = list(components.keys())
        self.components = components
        self.proportions = proportions

        # Ensures that our proportions are robust to dropped tokens (when the sequence length is imperfect)
        lower_bound_length = min(
            int(len(components[n]) // proportions[n])
            for n in self.names
            if proportions[n] > 0
        )

        self.counts = [int(proportions[n] * lower_bound_length) for n in self.names]

        self.offsets = [0]
        for c in self.counts[:-1]:
            self.offsets.append(self.offsets[-1] + c)
        
        self.length = sum(count for count in self.counts)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for comp_idx, start in enumerate(self.offsets):
            if idx < start + self.counts[comp_idx]:
                local_idx = idx - start
                comp_name = self.names[comp_idx]
                return self.components[comp_name][local_idx]
        raise IndexError(idx)