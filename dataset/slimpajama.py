import os
from datasets import load_dataset
from .disk_dataset import DiskDataset, DATASET_DIR

HUGGINGFACE_PATH = "DKYoon/SlimPajama-6B"

class SlimPajamaDataset(DiskDataset):
    
    def __init__(self, split, tokenizer, context_size, do_shuffle=False, allow_overlap=True):
    
        file_path = f"{DATASET_DIR}/slimpajama/{split}.bin"    
        
        if not os.path.exists(file_path):

            print(f"Creating SlimPajama [{split}] dataset files...")
            
            dataset = load_dataset(HUGGINGFACE_PATH, cache_dir=f"{DATASET_DIR}/raw")
            
            # train_test_splits = dataset.train_test_split(test_size=10000, shuffle=True, seed=42)

            # train_dataset = train_test_splits["train"]
            # test_dataset = train_test_splits["test"]
            
            # train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True, seed=42)
            # train_dataset = train_val_split["train"]
            # val_dataset = train_val_split["test"]

            train_dataset = dataset["train"]
            test_dataset = dataset["test"]
            val_dataset = dataset["validation"]

            DiskDataset.generate_data_file(train_dataset, f"{DATASET_DIR}/slimpajama/train.bin", tokenizer, separate_lines=False)
            DiskDataset.generate_data_file(test_dataset, f"{DATASET_DIR}/slimpajama/test.bin", tokenizer, separate_lines=False)
            DiskDataset.generate_data_file(val_dataset, f"{DATASET_DIR}/slimpajama/val.bin", tokenizer, separate_lines=False)
    
        super().__init__(file_path, tokenizer, context_size, do_shuffle=do_shuffle, allow_overlap=allow_overlap)

    def get_splits(tokenizer, max_seq_len):
        return {
            "train": SlimPajamaDataset("train", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=True),
            "val": SlimPajamaDataset("val", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=False),
            "test": SlimPajamaDataset("test", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=False)
        }