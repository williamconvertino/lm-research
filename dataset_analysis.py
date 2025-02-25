from datasets import load_dataset
import re

HUGGINGFACE_PATH = "bookcorpus"

dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"../data/datasets/raw", streaming=True)["train"]


bad_chars = ['=', '>', '<', '[', ']', '{', '}', '|', '\\', '/', '*', '+', '-', '&', '%', '$', '#', '@', '^', '~', '`']
             
for i, example in enumerate(dataset):
    
    if any(char in example["text"] for char in bad_chars):
        print(f"{example['text']}")
    
    if i > 10000:
        break
