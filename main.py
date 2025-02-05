import json
from argparse import ArgumentParser
from models.gpt import GPT
from util.trainer import Trainer
from dataset.dataset import get_dataloaders, get_tokenizer

def train(model, config):
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer,
        config["model"]["max_seq_len"],
        config["training"]["batch_size"]
    )
    trainer = Trainer(model, config["training"], train_loader, val_loader)
    trainer.train()

def eval(model):
    pass

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--config", type=str, default="default.json")
    args = parser.parse_args()

    config = json.load(f"configs/{args.config}")
    if config["model"]["type"] == "gpt":
        model = GPT(config["model"])
    else:
        raise ValueError(f"Invalid model type: {config['model']['type']}")

    assert args.train or args.eval, "Must specify either training or evaluation"
    assert not (args.train and args.eval), "Cannot specify both training and evaluation"

    if args.train:
        train(model, config)
    elif args.eval:
        eval(model, config)

if __name__ == "__main__":
    main()