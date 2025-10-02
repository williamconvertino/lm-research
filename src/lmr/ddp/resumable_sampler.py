import math
import torch
from torch.utils.data import DistributedSampler

# WARNING: Do NOT change your GPU count when using this sampler- it will not resume properly
class ResumableSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index = 0
        self.batch_size = 0

    # Follows the default DistributedSampler exactly, but with the addition of the start_index term
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        skip = self.start_index * self.batch_size
        
        self.start_index = 0 # Reset for future epochs
        
        if skip >= len(indices):
            return iter([])
        indices = indices[skip:]

        return iter(indices)

    # Make sure this is per-rank-step (do NOT multiply by worldsize, but DO include gradient acumulations when necessary)
    def set_start_index(self, start_index, batch_size):
        self.start_index = start_index
        self.batch_size = batch_size
