import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")

def setup_cache_dir():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    os.environ["HF_HOME"] = f"{CACHE_DIR}"
    os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"
    os.environ["TMPDIR"] = f"{CACHE_DIR}"

setup_cache_dir() # Set up a local cache directory (helps when using a server with limited user storage)

import torch
import json
from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from types import SimpleNamespace
import importlib
    
def load_most_recent_checkpoint(model):
    checkpoint_path = f"checkpoints/{model.name}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from {checkpoint_path} with epoch {epoch}")
    return model

def get_model(config):
    def get_model_class():
        model_file = importlib.import_module(f"models.{config.model.type.lower().replace(' ', '_')}")
        for attr in dir(model_file):
            if attr.lower() == config.model.type.lower():
                return getattr(model_file, attr)
        raise ValueError(f"Model class not found: {config.model.type}")
    model = get_model_class()(config.model)
    model.name = config.model.name
    return model

def get_datasets(config):
    tokenizer = Tokenizer()
    train_loader, val_loader, test_loader = TinyStoriesDataset.get_splits(tokenizer, config.model.max_seq_len, config.training.batch_size)
    return tokenizer, train_loader, val_loader, test_loader

def train(config):
    print(f"Training model [{config.model.name}]")
    model = get_model(config)
    try:
        model = load_most_recent_checkpoint(model)
    except FileNotFoundError:
        print(f"No checkpoint found for model [{config.model.name}], training from scratch")
    _, train_loader, val_loader, _ = get_datasets(config)
    trainer = Trainer(model, config.training, train_loader, val_loader)
    trainer.train()

def eval(config, eval_type):
    print(f"Evaluating model [{config.model.name}]")
    model = get_model(config)
    model = load_most_recent_checkpoint(model) # Throws error if no checkpoint found
    tokenizer, _, _, test_loader = get_datasets(config)
    evaluator = Evaluator(model, config, test_loader, tokenizer)
    if eval_type == "beam":
        evaluator.show_beams()
    elif eval_type == "greedy":
        evaluator.show_generations()
    else:
        raise ValueError(f"Invalid evaluation type: {eval_type}")

def dict_to_namespace(d, d_default=None):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    if d_default is not None:    
        for k, v in d_default.items():
            if k not in d:
                if isinstance(v, dict):
                    d[k] = dict_to_namespace(v)
                else:
                    d[k] = v
    return SimpleNamespace(**d)

def get_config(config_name):
    config_dict = json.load(open(f"configs/{config_name}.json"))
    default_dict = json.load(open(f"configs/default.json"))
    config = dict_to_namespace(config_dict, default_dict)
    config.model.name = config_name
    return config

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str, nargs="+")
    args = parser.parse_args()

    assert args.train or args.eval, "Must specify either training or evaluation"
    assert not (args.train and args.eval), "Cannot specify both training and evaluation"

    if args.train:
        config = get_config(args.train)
        train(config)
    elif args.eval:
        config = get_config(args.eval[0])
        eval_type = args.eval[1] if len(args.eval) > 1 else "greedy"
        eval(config, eval_type)

if __name__ == "__main__":
    main()