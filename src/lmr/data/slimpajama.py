import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login

from lmr.tokenizer import Tokenizer
from .disk_dataset import DiskDataset
from .proportional_dataset import ProportionalDataset

HF_BASE = "MBZUAI-LLM/SlimPajama-627B-DC"

def initialize_dataset(dataset_config, dataset_dir):
        
    tokenizer = Tokenizer.get_instance()
    
    for component_name, split_token_limits in dataset_config.component_token_limits.items():
        
        if dataset_config.component_whitelist is not None and component_name not in dataset_config.component_whitelist:
            print(f"Component {component_name} not on whitelist, skipping...")
            continue
        
        for split_name in ("train", "validation", "test"):
            
            token_limit = split_token_limits[split_name]
            component_dir = dataset_dir / dataset_config.dataset_name / component_name
            
            ds_stream = load_dataset(
                HF_BASE,
                data_files={split_name: f"{split_name}/{component_name}/*.jsonl.zst"},
                split=split_name,
                streaming=True
            )
            
            output_path = component_dir / f"{split_name}.bin"
            metadata_path = component_dir / f"metadata_{split_name}.json"
            
            DiskDataset.generate_bin(ds_stream, tokenizer, output_path, token_limit=token_limit, metadata_path=metadata_path)