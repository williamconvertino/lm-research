import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/datasets")

class DiskDataset:
    
    uc_translation_table = str.maketrans("", "", "�â€œ™") # Table to remove unwanted characters

    stride_multiplier = 0.5
    shuffle_buffer_size=1024

    def __init__(self, file_path, tokenizer, max_seq_len, do_shuffle=False, batch_size=64):
        
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = int(max_seq_len * self.stride_multiplier)
        self.do_shuffle = do_shuffle
        self.batch_size = batch_size

        self.data = np.memmap(file_path, dtype="int32", mode="r")
        self.file_size = os.path.getsize(self.file_path) // np.dtype("int32").itemsize

    def __len__(self):
        num_windows = (self.file_size - self.max_seq_len) // self.stride + 1
        return math.ceil(num_windows / self.batch_size)
    
    def __iter__(self):
        read_pointer = 0
        buffer = []
        batch = []

        def pop_buffer():
            pop_index = random.randint(0, len(buffer) - 1) if self.do_shuffle else 0 # Randomly select index if shuffling is enabled
            seq = buffer.pop(pop_index).copy()
            eos_id = self.tokenizer.eos_id
            pad_id = self.tokenizer.pad_id

            # Replace all tokens after an EOS with the pad token.
            indices = np.where(seq == eos_id)[0]
            if indices.size > 0:
                first_eos = indices[0]
                seq[first_eos + 1 :] = pad_id
            return torch.tensor(seq)

        # Read chunks from the memmapped data until there isn’t enough left.
        while read_pointer + self.max_seq_len <= len(self.data):
            chunk = self.data[read_pointer : read_pointer + self.max_seq_len].copy()
            buffer.append(chunk)
            read_pointer += self.stride

            # When the buffer is full enough, drain enough sequences into a batch.
            if len(buffer) >= self.shuffle_buffer_size:
                while buffer and len(batch) < self.batch_size:
                    batch.append(pop_buffer())
                if len(batch) == self.batch_size:
                    yield torch.stack(batch).long()
                    batch = []

        # Drain any remaining sequences in the buffer.
        while buffer:
            batch.append(pop_buffer())
            if len(batch) == self.batch_size:
                yield torch.stack(batch).long()
                batch = []
        
        # If there’s an incomplete batch left, yield it just once.
        if batch:
            yield torch.stack(batch).long()


    def preprocess(examples, tokenizer):
        # Remove unwanted characters
        texts = [text.translate(DiskDataset.uc_translation_table) for text in examples["text"]]
    
        # Tokenize text and add EOS and BOS tokens to each sequence
        examples["input_ids"] = tokenizer.encode(texts, eos=True, bos=True)
        
        return examples

    def generate_data_file(dataset, file_path, tokenizer, buffer_size=1024):
        
        # Preprocess data
        dataset = dataset.map(lambda x: DiskDataset.preprocess(x, tokenizer), batched=True, remove_columns=["text"])
        file_size = sum([len(example) for example in dataset["input_ids"]])
        
        # Initialize memmap array
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        memmap_array = np.memmap(file_path, dtype="int32", mode="w+", shape=(file_size,))
        
        # Write data to memmap array
        buffer = []
        write_pointer = 0
        
        for sequence in tqdm(dataset["input_ids"], desc="Generating dataset files"):
            buffer.extend(sequence)
            if len(buffer) >= buffer_size:
                memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
                write_pointer += len(buffer)
                buffer = []
        
        if len(buffer) > 0:
            memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
            write_pointer += len(buffer)
            buffer = []
            
        memmap_array.flush()
        return memmap_array