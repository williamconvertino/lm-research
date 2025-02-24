import os
import torch
import importlib
from types import SimpleNamespace
import json
import sys

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "../checkpoints")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "../configs")

def load_most_recent_checkpoint(model):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model.config.name}.pt")
    if not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path)

def load_model(config):
    
    sys.path.append(MODELS_DIR)
    model_name = config.model
    module_name = f"models.{model_name}"
    model_cls = None
    
    try:
        model_file = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Module not found: {module_name}")
    
    for attr in dir(model_file): # Find class in module
        print(attr)
        if attr.lower() == model_name.lower().replace("_", ""):
            model_cls = getattr(model_file, attr)
        break
    
    if model_cls is None:
        raise ValueError(f"Model class not found: {model_name}")    
        
    return model_cls(config)

def load_config(config_name):
    
    config_dir = os.path.join(CONFIG_DIR, f"{config_name}.json")
    
    if not os.path.exists(config_dir):
        raise ValueError(f"Config file not found: {config_dir}")
    
    def dict_to_namespace(d):
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)
    
    config = dict_to_namespace(json.load(open(config_dir)))
    config.name = config_name
    
    return config