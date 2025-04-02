import os
from datasets import load_dataset
from .disk_dataset import DiskDataset, DATASET_DIR

HUGGINGFACE_PATH = "DKYoon/BabyLM-6B"

class BabyLMDataset(DiskDataset):
    
    def __init__(self, split, tokenizer, context_size, do_shuffle=False, allow_overlap=True):
    
        file_path = f"{DATASET_DIR}/babylm/{split}.bin"    
        
        if not os.path.exists(file_path):

            print(f"Creating BabyLM [{split}] dataset files...")
            
            train_path = "{DATASET_DIR}/raw/train_100M"
            val_path = "{DATASET_DIR}/raw/dev"
            test_path = "{DATASET_DIR}/raw/test"

            train_dataset = self.load_dataset_from_files(train_path)
            test_dataset = self.load_dataset_from_files(test_path)
            val_dataset = self.load_dataset_from_files(val_path)

            DiskDataset.generate_data_file(train_dataset, f"{DATASET_DIR}/babylm/train.bin", tokenizer, separate_lines=False)
            DiskDataset.generate_data_file(test_dataset, f"{DATASET_DIR}/babylm/test.bin", tokenizer, separate_lines=False)
            DiskDataset.generate_data_file(val_dataset, f"{DATASET_DIR}/babylm/val.bin", tokenizer, separate_lines=False)
    
        super().__init__(file_path, tokenizer, context_size, do_shuffle=do_shuffle, allow_overlap=allow_overlap)

    def load_dataset_from_files(folder_path):
        data = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                data.append({"text": text})
        return Dataset.from_list(data)

    def get_splits(tokenizer, max_seq_len):
        return {
            "train": BabyLMDataset("train", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=True),
            "val": BabyLMDataset("val", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=False),
            "test": BabyLMDataset("test", tokenizer, max_seq_len, do_shuffle=False, allow_overlap=False)
        }