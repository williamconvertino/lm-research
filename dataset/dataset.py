from datasets import load_dataset, concatenate_datasets
from transformers import GPT2TokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, token_ids, max_seq_len):
        self.max_seq_len = max_seq_len
        self.examples = []
        for i in range(0, len(token_ids) - max_seq_len, max_seq_len):
            self.examples.append(token_ids[i:i+max_seq_len])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.examples[idx], dtype=torch.long)}

def prepare_datasets(tokenizer, max_seq_len, cache_dir="./data"):
    dataset = concatenate_datasets([
        load_dataset("text", data_files="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt", cache_dir=cache_dir)['train'],
        load_dataset("text", data_files="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt", cache_dir=cache_dir)['train']
    ])

    tokenized_datasets = {}
    for split in ["train", "validation", "test"]:
        texts = dataset[split]["text"]
        full_text = "\n\n".join(texts)
        tokens = tokenizer.encode(full_text)
        tokenized_datasets[split] = tokens
    
    train_dataset = TextDataset(tokenized_datasets["train"], max_seq_len)
    val_dataset = TextDataset(tokenized_datasets["validation"], max_seq_len)
    test_dataset = TextDataset(tokenized_datasets["test"], max_seq_len)
    
    return train_dataset, val_dataset, test_dataset

def get_tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")

def get_dataloaders(tokenizer, max_seq_len, batch_size, cache_dir="./data"):
    train_dataset, val_dataset, test_dataset = prepare_datasets(tokenizer, max_seq_len, cache_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader