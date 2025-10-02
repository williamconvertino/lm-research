from lmr.utils.parsing import formatted_string_to_int

def set_dataset_defaults(dataset_config):    
    
    # Verify proportional sampling specifications
    if dataset_config.sampling_type == "proportional":
        
        assert hasattr(dataset_config, "proportions"), "Must pass proportion dict when sampling type 'proportional' is set."
        
        total_proportion = sum(proportion for proportion in dataset_config.proportions.values())
        assert total_proportion == 1.0, f"Proportions must sum to 1.0, got {total_proportions}"

def set_proportion_token_limits(dataset_config):    
        
        proportions = dataset_config.proportions
        component_token_limits = {}
        
        for component_name, component_proportion in proportions.items():
            
            split_token_limits = {}
            
            tokens_buffer = formatted_string_to_int(dataset_config.tokens_buffer) if hasattr(dataset_config, "tokens_buffer") and dataset_config.tokens_buffer is not None else 0
            
            # Set the token limits for each split, or set to None if not specified
            for split_name in ("train", "validation", "test"):    
                if hasattr(dataset_config, f"max_tokens_{split_name}"):            
                    max_tokens = formatted_string_to_int(getattr(dataset_config, f"max_tokens_{split_name}"))
                    split_token_limits[split_name] = int(component_proportion * max_tokens) + tokens_buffer
                else:
                    split_token_limits[split_name] = None
                    
            component_token_limits[component_name] = split_token_limits
        
        dataset_config.component_token_limits = component_token_limits