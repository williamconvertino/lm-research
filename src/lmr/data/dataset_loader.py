import hydra
from hydra.utils import get_method
from pathlib import Path

from lmr.data.disk_dataset import DiskDataset
from lmr.data.proportional_dataset import ProportionalDataset

from lmr.config import initialize_config

def initialize_dataset(dataset_config, dataset_dir):
    get_method(dataset_config.init_fn)(dataset_config, dataset_dir)

def get_dataset_splits(dataset_config, max_seq_len, dataset_dir, split_names=('train', 'validation', 'test')):

    output_dir = dataset_dir / dataset_config.dataset_name
    
    # Initialize dataset if folder missing
    if not output_dir.exists():
        initialize_dataset(dataset_config, dataset_dir)
        
    splits = {}
    
    for split_name in split_names:
        stride_fraction = 0.5 if dataset_config.use_sliding_window and split_name == "train" else None
        if dataset_config.sampling_type == 'proportional':
            components = {}
            for component_name in dataset_config.proportions.keys():
                file_path = output_dir / component_name / f"{split_name}.bin"
                component_dataset = DiskDataset(file_path, max_seq_len, stride_fraction=stride_fraction, allow_cycling=True)
                components[component_name] = component_dataset
            splits[split_name] = ProportionalDataset(components, dataset_config.proportions)
        else:
            file_path = output_dir / f"{split_name}.bin"
            splits[split_name] = DiskDataset(file_path, max_seq_len, stride_fraction=stride_fraction, allow_cycling=False)
    
    return splits