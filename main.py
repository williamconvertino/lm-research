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
        return None
    return torch.load(checkpoint_path)

def load_model(config):
    def get_model_class():
        model_file = importlib.import_module(f"models.{config.type.lower().replace(' ', '_')}")
        for attr in dir(model_file):
            if attr.lower() == config.type.lower():
                return getattr(model_file, attr)
        raise ValueError(f"Model class not found: {config.type}")
    model = get_model_class()(config)
    model.name = config.name
    model.tokenizer = config.tokenizer
    return model

def train(config):
    print(f"Training model [{config.name}]")
    model = load_model(config)
    checkpoint = load_most_recent_checkpoint(model)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    splits = TinyStoriesDataset.get_splits(model.tokenizer, config.max_seq_len)
    trainer = Trainer(model, splits, checkpoint)
    trainer.train()

def eval(config, eval_flags):
    print(f"Evaluating model [{config.name}]")
    model = load_model(config)
    checkpoint = load_most_recent_checkpoint(model)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found for model [{config.name}], cannot evaluate")
    model.load_state_dict(checkpoint["model_state_dict"])
    splits = TinyStoriesDataset.get_splits(model.tokenizer, config.max_seq_len)
    evaluator = Evaluator(model, splits, model.tokenizer)
    if "beam" in eval_flags:
        evaluator.eval_beam()
    if "greedy" in eval_flags:
        evaluator.eval_greedy()
    
def get_config(config_name):
    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    config = dict_to_namespace(json.load(open(f"configs/{config_name}.json")))
    config.name = config_name
    return config

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str, nargs="+")
    args = parser.parse_args()

    assert args.train or args.eval, "Must specify either training or evaluation"
    assert not (args.train and args.eval), "Cannot specify both training and evaluation"

    tokenizer = Tokenizer()

    config = get_config(args.train) if args.train else get_config(args.eval[0])
    config.tokenizer = tokenizer
    config.vocab_size = tokenizer.vocab_size

    if args.train:
        train(config)
    elif args.eval:
        eval_flags = args.eval[1:] if len(args.eval) > 1 else ["greedy", "beam"]
        eval(config, eval_flags)

if __name__ == "__main__":
    main()