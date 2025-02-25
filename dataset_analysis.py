from datasets import load_dataset

HUGGINGFACE_PATH = "bookcorpus"

dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"../data/datasets/raw", streaming=True)["train"]

flag = False
for i, example in enumerate(dataset):
    if "=" in example["text"]:
        flag = True
        
    if flag:
        print(example["text"])
        
    if i > 1000:
        break