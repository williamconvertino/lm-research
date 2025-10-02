import torch
from torch.utils.data import Dataset

class ProportionalDataset(Dataset):
    def __init__(self, components, proportions):

        assert set(components) == set(proportions), "Components and proportions must have identical keys"

        total = sum(proportions.values())
        assert total == 1.0, f"Proportions must sum to 1.0 (got {total})"

        self.names = list(components.keys())
        self.components = components
        self.proportions = proportions

        # Adjust the length to roughly align with the lower-bound (in case the datasets are unequally sized to the proportions)
        # Technically they should wrap and have no issues, but this ensures no issues.
        self.length = min(
            int(len(components[n]) // proportions[n])
            for n in self.names
            if proportions[n] > 0
        )

        self.counts = [int(round(proportions[n] * self.length)) for n in self.names]

        self.offsets = [0]
        for c in self.counts[:-1]:
            self.offsets.append(self.offsets[-1] + c)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        for comp_idx, start in enumerate(self.offsets):
            if idx < start + self.counts[comp_idx]:
                local_idx = idx - start
                comp_name = self.names[comp_idx]
                return self.components[comp_name][local_idx]
        raise IndexError(idx)