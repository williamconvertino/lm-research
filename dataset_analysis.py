from datasets import load_dataset

HUGGINGFACE_PATH = "bookcorpus"

dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"../data/datasets/raw", streaming=True)["train"]

count = 0

for i, example in enumerate(dataset):
    print("=" * 10)
    if "=" in example["text"]:
        print(example["text"])
        count = 3
    if count > 0:
        print(example["text"])
    count -= 1
    if i > 1000:
        break