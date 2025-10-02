import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

from lmr.tokenizer import Tokenizer
from .disk_dataset import DiskDataset

HF_PATH = "DKYoon/SlimPajama-6B"

def initialize_dataset(dataset_config, dataset_dir):
    
    load_dotenv()
    try:
        login(token=os.getenv("HF_TOKEN"))  
    except:
        print("No HF_TOKEN found, skipping login...")

    tokenizer = Tokenizer.get_instance()
    
    for split_name in ("train", "validation", "test"):
        
        token_limit = split_token_limits[split_name]
        output_dir = dataset_dir / dataset_config.dataset_name
        
        ds_stream = load_dataset(HF_PATH, split=split_name)
        
        output_path = output_dir / f"{split_name}.bin"
        metadata_path = output_dir / f"metadata_{split_name}.json"
        
        DiskDataset.generate_bin(ds_stream, tokenizer, output_path, token_limit=token_limit, metadata_path=metadata_path)