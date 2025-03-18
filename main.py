import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments

import time
import torch
from argparse import ArgumentParser
from util.trainer import Trainer
from util.evaluator import Evaluator
from dataset.tokenizer import Tokenizer
from dataset.tinystories_dataset import TinyStoriesDataset
from dataset.bookcorpus import BookCorpusDataset
from util.loading import load_model, load_most_recent_checkpoint, load_config

def wait_for_free_gpu(vram=13):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, cannot use GPU")
    
    print("Waiting for free GPU...")
    gpu = None
    count = 0
    
    while True:
        
        print(f"Waiting for {count} seconds", end="\r")
        count += 1

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu = torch.device(f'cuda:{i}')
            free_memory, total_memory = torch.cuda.mem_get_info(gpu)
            total_memory = int(total_memory / 1024**3)
            free_memory = int(free_memory / 1024**3)  
            if free_memory > vram:
                print(f"Using GPU [{i}]: {props.name} with {free_memory:.2f}GB")
                return torch.device(f'cuda:{i}')
                
        time.sleep(1)

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str)
    parser.add_argument("--eval", type=str, nargs="+")
    parser.add_argument("--dataset", type=str, default="tiny_stories")
    parser.add_argument("--wait", action="store_true")
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
    
    if args.wait:
        wait_for_free_gpu()
    
    if args.train:
        trainer = Trainer(model, splits, tokenizer, checkpoint)
        trainer.train()
    elif args.eval:
        assert checkpoint is not None, "No checkpoint found for model, cannot evaluate"
        model.load_state_dict(checkpoint["model_state_dict"])
        
        eval_flags = args.eval[1:] if len(args.eval) > 1 else ["loss", "greedy", "beam", "topk", "nucleus"]
        evaluator = Evaluator(model, splits, tokenizer)
        if "loss" in eval_flags:
            evaluator.eval_loss()
        evaluator.eval(eval_flags)

if __name__ == "__main__":
    main()