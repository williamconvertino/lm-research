import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from dataset.bookcorpus import BookCorpusDataset
from util.loading import load_model, load_most_recent_checkpoint, load_config

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
    elif args.dataset == "bookcorpus":
        splits = BookCorpusDataset.get_splits(tokenizer, config.max_seq_len)
    else:
        raise ValueError(f"Dataset not found: {args.dataset}")
    
    model = load_model(config)
    checkpoint = load_most_recent_checkpoint(model)
    
    if args.train:
        trainer = Trainer(model, splits, tokenizer, checkpoint)
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