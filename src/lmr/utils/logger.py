import os

class Logger:
    _instance = None

    def __init__(self):
        Logger._instance = self
        self.enabled = True
        self.rank = self._get_true_rank()

    @staticmethod
    def _get_true_rank():
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except ImportError:
            pass
        return int(os.environ.get("RANK", 0))

    @staticmethod
    def log(message, prefix="[LOG] "):
        instance = Logger.get_instance()
        if not instance.enabled or instance.rank != 0:
            return
        if prefix is None:
            print(message, flush=True)
        else:
            print(prefix + message, flush=True)

    @staticmethod
    def get_instance():
        if Logger._instance is None:
            Logger._instance = Logger()
        return Logger._instance