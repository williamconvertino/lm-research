from pathlib import Path
import json
import os
from datetime import datetime

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

class DiskDataset(Dataset):
    def __init__(self, file_path, max_seq_len, stride_fraction=None, allow_cycling=False):
        self.file_path = Path(file_path).resolve()
        assert self.file_path.is_file(), f"File not found: {self.file_path}"
        
        self.max_seq_len = max_seq_len
        self.stride_fraction = stride_fraction if stride_fraction is not None else 1.0
        self.stride = int(self.max_seq_len * self.stride_fraction)
        
        self.allow_cycling = allow_cycling # Allows the dataset to extend indefinitely, for use in mixed source datasets.

        self.data = np.memmap(self.file_path, dtype="int32", mode="r")
        self.file_size = len(self.data)
        
        self.n_batches = 1 + max(0, (self.file_size - self.max_seq_len) // self.stride)

    def __len__(self):
        return self.n_batches

    def get_token_count(self):
        return self.file_size

    def __getitem__(self, idx):
        if self.allow_cycling:
            idx = idx % self.n_batches
        start = idx * self.stride
        end = start + self.max_seq_len
        seq = np.array(self.data[start:end], dtype=np.int32, copy=True)
        return torch.from_numpy(seq).long()

    @staticmethod
    def generate_bin(
        dataset_iterator,
        tokenizer,
        output_path,
        add_eos=True,
        column="text",
        token_limit=None,
        metadata_path=None,
        initial_alloc_tokens=1_000_000
    ):
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        def tokenize_fn(text):
            ids = tokenizer.encode(text)
            if add_eos:
                ids.append(tokenizer.eos_token_id)
            return ids

        # Progress bar setup
        try:
            dataset_length = len(dataset_iterator)
            total_for_tqdm = dataset_length if token_limit is None else token_limit
            unit = "samples" if token_limit is None else "tokens"
        except TypeError:
            total_for_tqdm = token_limit if token_limit is not None else None
            unit = "tokens" if token_limit is not None else "samples"

        print(f"Building binary file at: {output_path}")
        pbar = tqdm(total=total_for_tqdm, unit=unit)

        # Helper: ensure memmap has capacity >= needed
        dtype = np.int32
        bytes_per_token = np.dtype(dtype).itemsize

        # Start with an empty file
        if output_path.exists():
            output_path.unlink()

        allocated = 0
        pos = 0
        mm = None  # np.memmap

        def _grow(new_alloc_tokens):
            nonlocal mm, allocated
            # Close old memmap (if any) before changing file size
            if mm is not None:
                mm.flush()
                del mm
                mm = None
            # Extend file to new size (bytes)
            new_bytes = new_alloc_tokens * bytes_per_token
            with open(output_path, "a+b") as f:
                if new_bytes > 0:
                    f.seek(new_bytes - 1)
                    f.write(b"\0")
                    f.flush()
                    os.fsync(f.fileno())
            allocated = new_alloc_tokens
            # Re-map with new size
            mm = np.memmap(output_path, dtype=dtype, mode="r+", shape=(allocated,))

        # Initialize with a small allocation
        _grow(initial_alloc_tokens)

        token_count = 0
        done = False

        for example in dataset_iterator:
            if done:
                break

            ids = tokenize_fn(example[column])
            if token_limit is not None:
                remaining = token_limit - token_count
                if remaining <= 0:
                    break
                if len(ids) > remaining:
                    ids = ids[:remaining]

            if not ids:
                # Empty text -> possibly just EOS; skip if became empty
                pbar.update(len(ids) if unit == "tokens" else 1)
                continue

            arr = np.asarray(ids, dtype=dtype)
            needed = pos + arr.size
            if needed > allocated:
                # Grow geometrically to reduce number of resizes
                new_alloc = max(allocated * 2, needed)
                _grow(new_alloc)

            mm[pos:pos + arr.size] = arr
            pos += arr.size
            token_count += arr.size

            if unit == "tokens":
                pbar.update(arr.size)
            else:
                pbar.update(1)

            if token_limit is not None and token_count >= token_limit:
                done = True

        pbar.close()

        # Flush and truncate to exact size
        if mm is not None:
            mm.flush()
            del mm
            mm = None

        with open(output_path, "r+b") as f:
            f.truncate(pos * bytes_per_token)
            f.flush()
            os.fsync(f.fileno())

        # Optional metadata
        if metadata_path is not None:
            meta = {
                "last_modified": datetime.now().isoformat(),
                "tokenizer_base_type": getattr(tokenizer, "base_type", None),
                "total_tokens": int(token_count),
                "dtype": str(np.dtype(dtype)),
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        print(f"✅ Wrote {token_count} tokens to {output_path} (int32)")

        return token_count