import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from .resumable_sampler import ResumableSampler
from lmr.utils.logger import Logger

def setup_ddp():
    world_size = torch.cuda.device_count()
    rank = int(os.environ.get("RANK", 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    Logger.get_instance().rank = rank
    return rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()
    
def initialize_model_ddp(model, rank):
    return DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=False)

def unwrap_model(model):
        return model.module if hasattr(model, "module") else model
    
def initialize_samplers_ddp(splits, rank, world_size):
    
    samplers = {}
    
    for split_name, split in splits.items():
        samplers[split_name] = ResumableSampler(split, num_replicas=world_size, rank=rank)
        
    return samplers