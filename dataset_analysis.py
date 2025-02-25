from datasets import load_dataset
import re

HUGGINGFACE_PATH = "bookcorpus"

dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"../data/datasets/raw", streaming=True)["train"]


for i, example in enumerate(dataset):
    text = example['text']
    if re.search(r'[^a-zA-Z0-9\s\p{P}]', text):
        print(text)
    if i > 1000:
        break
