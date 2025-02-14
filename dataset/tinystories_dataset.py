import os
from datasets import load_dataset, concatenate_datasets
from .disk_dataset import DiskDataset, DATASET_DIR

HUGGINGFACE_PATH = 'roneneldan/TinyStories'

class TinyStoriesDataset(DiskDataset):
    
    def __init__(self, tokenizer, split, context_size, stride=0.5, batch_size=64, shuffle=False, shuffle_buffer_size=1024):
    
        file_path = f'{DATASET_DIR}/tinystories/{split}.bin'    
        
        if not os.path.exists(file_path):

            print(f'Creating TinyStories [{split}] dataset files...')
            
            dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f'{DATASET_DIR}/raw')
            dataset = concatenate_datasets([dataset['train'], dataset['validation']])
            
            train_test_splits = dataset.train_test_split(test_size=10000, shuffle=True, seed=42)

            train_dataset = train_test_splits['train']
            test_dataset = train_test_splits['test']
            
            train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True, seed=42)
            train_dataset = train_val_split['train']
            val_dataset = train_val_split['test']

            DiskDataset.generate_data_file(train_dataset, f'{DATASET_DIR}/tinystories/train.bin', tokenizer)
            DiskDataset.generate_data_file(test_dataset, f'{DATASET_DIR}/tinystories/test.bin', tokenizer)
            DiskDataset.generate_data_file(val_dataset, f'{DATASET_DIR}/tinystories/val.bin', tokenizer)
    
        super().__init__(file_path, context_size, stride, batch_size, shuffle, shuffle_buffer_size)

def get_ts_datasets(tokenizer, max_seq_len, batch_size):
    
    train_dataset = TinyStoriesDataset(tokenizer, 'train', max_seq_len, batch_size=batch_size, shuffle=True)
    val_dataset = TinyStoriesDataset(tokenizer, 'val', max_seq_len, batch_size=batch_size)
    test_dataset = TinyStoriesDataset(tokenizer, 'test', max_seq_len, batch_size=batch_size)

    return train_dataset, val_dataset, test_dataset