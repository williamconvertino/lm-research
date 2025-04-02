import torch

class DictionaryLearning:
    
    def __init__(self, model, splits):
        self.model = model
        self.splits = splits
        self.device = self._get_device()
        
    def _get_device(self):
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return torch.device('cpu')
        vram_required = 5
        print(f"Estimated VRAM required: {vram_required:.2f}GB")
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu = torch.device(f'cuda:{i}')
                free_memory, total_memory = torch.cuda.mem_get_info(gpu)
                total_memory = int(total_memory / 1024**3)
                free_memory = int(free_memory / 1024**3)  
                if free_memory > vram_required:
                    print(f"Using GPU [{i}]: {props.name} with {free_memory:.2f}GB")
                    return torch.device(f'cuda:{i}')
                else:
                    print(f"GPU [{i}]: {props.name} only has {free_memory:.2f}GB free memory, skipping")
            except Exception:
                print(f"Error reading GPU [{i}], skipping")
        raise RuntimeError(f"No GPU with at least {vram_required}GB of free memory found")
    
        
    def collect_data(self, k=1):
        
        self.model.eval()
        self.model.to(self.device)
        self.model.config.gather_neurons = True
        
        with torch.no_grad():
            for i, batch in enumerate(self.splits["test"]):
                if i >= k:
                    break
                batch = batch.to(self.device)
                _, _ = self.model(batch)
                
                