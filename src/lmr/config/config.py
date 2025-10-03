from omegaconf import OmegaConf

from .model_config import set_model_defaults, set_size_defaults, merge_model_and_size_configs
from .dataset_config import set_proportion_token_limits
from .training_config import set_training_defaults

def initialize_config(config):
    OmegaConf.set_struct(config, False)

    model_config = config.model
    size_config = config.size
    training_config = config.training
    
    # Set save name if not already specified
    full_name = f"{model_config.model_name}_{size_config.size_name}"
    model_config.full_name = full_name
    
    if config.checkpoint_name == "auto":
        config.checkpoint_name = full_name
    
    # Set tokenizer base
    if config.tokenizer_base == "auto":
        config.tokenizer_base = "gpt2"
    
    # Set model/size default values
    set_model_defaults(model_config)
    set_size_defaults(size_config)
        
    # Merge model and size configs (pruning unused fields)
    merge_model_and_size_configs(model_config, size_config) 
            
    # Set training default values
    set_training_defaults(training_config, model_config)
            
    # Initialize dataset
    if hasattr(config, "dataset") and config.dataset is not None:
        dataset_config = config.dataset
        if dataset_config.sampling_type == "proportional":
            set_proportion_token_limits(dataset_config)
        
        dataset_config["use_sliding_window"] = training_config.use_sliding_window
                
    OmegaConf.set_struct(config, True)
    
    return config