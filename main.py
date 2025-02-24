import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from util.loading import load_most_recent_checkpoint, load_config
import os
import importlib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_model(config):
    
    model_name = config.model
    model_dir = os.path.join(MODELS_DIR)
    model_cls = None
    
    try:
        model_file = importlib.import_module(model_dir)
    except ModuleNotFoundError:
        raise ValueError(f"Model file not found: {model_dir}")
    
    for attr in dir(model_file): # Find class in module
        if attr.lower() == model_name.lower().replace("_", ""):
            model_cls = getattr(model_file, attr)
        break
    
    if model_cls is None:
        raise ValueError(f"Model class not found: {model_name}")    
        
    return model_cls(config)

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str)
    parser.add_argument("--dataset", type=str, default="tiny_stories")
    args = parser.parse_args()

    assert args.train or args.eval, "Must specify either training or evaluation"
    assert not (args.train and args.eval), "Cannot specify both training and evaluation"

    config = load_config(args.train) if args.train else load_config(args.eval[0])
    
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size
    
    if args.dataset == "tiny_stories":
        splits = TinyStoriesDataset.get_splits(tokenizer, config.max_seq_len)
    else:
        raise ValueError(f"Dataset not found: {args.dataset}")
    
    model = load_model(config)
    checkpoint = load_most_recent_checkpoint(model)
    
    if args.train:
        trainer = Trainer(model, splits, checkpoint)
        trainer.train()
    elif args.eval:
        assert checkpoint is not None, "No checkpoint found for model, cannot evaluate"
        model.load_state_dict(checkpoint["model_state_dict"])
        
        eval_flags = args.eval[1:] if len(args.eval) > 1 else ["greedy", "beam"]
        evaluator = Evaluator(model, splits, tokenizer)
        
        if "beam" in eval_flags:
            evaluator.eval_beam()
        if "greedy" in eval_flags:
            evaluator.eval_greedy()

if __name__ == "__main__":
    main()