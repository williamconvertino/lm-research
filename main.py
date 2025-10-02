import os
from pathlib import Path

import torch
from dotenv import load_dotenv

import hydra
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from lmr.config import initialize_config
from lmr.tokenizer import Tokenizer
from lmr.models import get_model
from lmr.datasets import initialize_dataset, get_dataset_splits
from lmr.training import Trainer
from lmr.checkpointing import Checkpointing
from lmr.utils.seed import set_seed

DATASET_DIR = Path("datasets")
CHECKPOINT_DIR = Path("checkpoints")

def train_model(config):
    
    tokenizer = Tokenizer(config.tokenizer_base)
    model = get_model(config.model, tokenizer.vocab_size)
    
    # Changes dataset_dir based on selected tokenizer (to avoid tokenization mismatch)
    tokenized_dataset_dir = DATASET_DIR / config.tokenizer_base
    splits = get_dataset_splits(config.dataset, model.config.max_seq_len, tokenized_dataset_dir)

    checkpointing = Checkpointing(model, CHECKPOINT_DIR / config.checkpoint_name)
    
    trainer = Trainer(config.training, model, tokenizer, splits, checkpointing)
    trainer.train()

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(config):
    
    load_dotenv()
    set_seed(config)
    initialize_config(config)
    
    if config.mode == "train":
        train_model(config)
    
if __name__ == "__main__":
    main()