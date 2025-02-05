import torch
import json
from argparse import ArgumentParser
from models.gpt import GPT
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.dataset import get_dataloaders, get_tokenizer
from types import SimpleNamespace

def load_checkpoint(model, optimizer, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
    except:
        print("Failed to load checkpoint")
        return None
    epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print(f"Loaded checkpoint from {checkpoint_path} with epoch {epoch}")
    return model, optimizer

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

def train(model, config):
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer,
        config.model.max_seq_len,
        config.training.batch_size
    )
    trainer = Trainer(model, config.training, train_loader, val_loader)
    trainer.train()

def eval(model, config, eval_type=None):
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer,
        config.model.max_seq_len,
        config.training.batch_size
    )
    evaluator = Evaluator(model, config, test_loader, tokenizer)
    if eval_type == "beam":
        evaluator.show_generations_beam()
    else:
        evaluator.show_generations()

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--config", type=str, default="default.json")
    parser.add_argument("--eval_type", type=str, default="beam", choices=["beam", "greedy"])
    args = parser.parse_args()
    
    config_dict = json.load(open(f"configs/{args.config}"))
    config = dict_to_namespace(config_dict)
    
    if config.model.type == "gpt":
        model = GPT(config.model)
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