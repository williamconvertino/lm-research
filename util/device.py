import torch

def get_device():
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    min_vram = 9
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            gpu = torch.device(f'cuda:{i}')
            free_memory, total_memory = torch.cuda.mem_get_info(gpu)
            total_memory = int(total_memory / 1024**3)
            free_memory = int(free_memory / 1024**3)  
            if free_memory > min_vram:
                print(f"Using GPU [{i}]: {props.name} with {free_memory:.2f}GB")
                return torch.device(f'cuda:{i}')
            else:
                print(f"GPU [{i}]: {props.name} only has {free_memory:.2f}GB free memory, skipping")
        except Exception:
            print(f"Error reading GPU [{i}], skipping")
    raise RuntimeError(f"No GPU with at least {min_vram}GB of free memory found")