from datasets import load_dataset

HUGGINGFACE_PATH = "bookcorpus"

dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"../data/datasets/raw", streaming=True)["train"]

for i, example in enumerate(dataset):
    print("=" * 10)
    print(example)    
    if i > 50:
        break