import os
import torch
import json
from argparse import ArgumentParser
from models.gpt import GPT
from models.gpt2 import GPT2
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.dataset import get_dataloaders, get_tokenizer
from types import SimpleNamespace

def setup_cache_dir(dir = "./data"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.environ['HF_HOME'] = f'{dir}'
    os.environ['TRANSFORMERS_CACHE'] = f'{dir}/transformers'
    os.environ['HF_DATASETS_CACHE'] = f'{dir}/datasets'
    os.environ['TMPDIR'] = f'{dir}'

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path} with epoch {epoch}")
    return model

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

def train(model, config):
    print(f"Training model [{config.model.name}]")
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer,
        config.model.max_seq_len,
        config.training.batch_size
    )
    trainer = Trainer(model, config.training, train_loader, val_loader)
    trainer.train()

def eval(model, config, eval_type=None):
    print(f"Evaluating model [{config.model.name}]")
    checkpoint_path = f"checkpoints/{config.model.name}.pt"
    model = load_checkpoint(model, checkpoint_path)
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer,
        config.model.max_seq_len,
        config.training.batch_size
    )
    evaluator = Evaluator(model, config, test_loader, tokenizer)
    if eval_type == "beam":
        evaluator.show_beams()
    else:
        evaluator.show_generations()

def main():
    setup_cache_dir()
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--eval_type", type=str, default="beam", choices=["beam", "greedy"])
    args = parser.parse_args()
    
    config_dict = json.load(open(f"configs/{args.config}.json"))
    config = dict_to_namespace(config_dict)
    
    if config.model.type == "gpt":
        model = GPT(config.model)
    elif config.model.type == "gpt2":
        model = GPT2(config.model)
    else:
        raise ValueError(f"Invalid model type: {config.model.type}")

    assert args.train or args.eval, "Must specify either training or evaluation"
    assert not (args.train and args.eval), "Cannot specify both training and evaluation"

    if args.train:
        train(model, config)
    elif args.eval:
        eval(model, config, args.eval_type)

if __name__ == "__main__":
    main()