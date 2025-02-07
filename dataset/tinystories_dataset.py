import os
from datasets import load_dataset, concatenate_datasets
from transformers import GPT2TokenizerFast
from .dataset import Dataset, DATASET_DIR
from torch.utils.data import DataLoader

HUGGINGFACE_PATH = 'roneneldan/TinyStories'

class TinyStoriesDataset(Dataset):
    
    def __init__(self, tokenizer, split, context_size, stride=0.5, batch_size=64):
    
        file_path = f'{DATASET_DIR}/tinystories/{split}.bin'    
        
        if not os.path.exists(file_path):

            print(f'Creating TinyStories [{split}] dataset files...')
            
            dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f'{DATASET_DIR}/raw')
            dataset = concatenate_datasets([dataset['train'], dataset['validation']])
            
            train_test_splits = dataset.train_test_split(test_size=10000, shuffle=True)

            train_dataset = train_test_splits['train']
            test_dataset = train_test_splits['test']
            
            train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True)
            train_dataset = train_val_split['train']
            val_dataset = train_val_split['test']

            Dataset.generate_data_file(train_dataset, f'{DATASET_DIR}/tinystories/train.bin', tokenizer)
            Dataset.generate_data_file(test_dataset, f'{DATASET_DIR}/tinystories/test.bin', tokenizer)
            Dataset.generate_data_file(val_dataset, f'{DATASET_DIR}/tinystories/val.bin', tokenizer)
    
        super().__init__(file_path, context_size)

def get_ts_tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2")

def get_ts_dataloaders(tokenizer, max_seq_len, batch_size):
    
    train_dataset = TinyStoriesDataset(tokenizer, 'train', max_seq_len)
    val_dataset = TinyStoriesDataset(tokenizer, 'val', max_seq_len)
    test_dataset = TinyStoriesDataset(tokenizer, 'test', max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader