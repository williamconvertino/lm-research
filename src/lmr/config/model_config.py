def set_model_defaults(model_config):
    model_config["vocab_size"] = None

def set_size_defaults(size_config):
    def set_default(config, k, v):
        if config.get(k, "auto") == "auto":
            config[k] = v

    # Transformer defaults
    set_default(size_config, "embed_dim", size_config.get("hidden_dim"))
    set_default(size_config, "mlp_dim", size_config.get("hidden_dim") * 4 if size_config.get("hidden_dim") != "auto" else "auto")

    set_default(size_config, "n_heads_f", size_config.get("n_heads"))
    set_default(size_config, "n_heads_phi", size_config.get("n_heads"))

    set_default(size_config, "hidden_dim_f", size_config.get("hidden_dim"))
    set_default(size_config, "hidden_dim_phi", size_config.get("hidden_dim"))

    # FST defaults
    set_default(size_config, "embed_dim_f",
                size_config.get("embed_dim") if size_config.get("embed_dim") != "auto" else size_config.get("hidden_dim_f"))
    set_default(size_config, "mlp_dim_f",
                size_config.get("mlp_dim") if size_config.get("mlp_dim") != "auto" else size_config.get("hidden_dim_f") * 4)

    set_default(size_config, "embed_dim_phi",
                size_config.get("embed_dim") if size_config.get("embed_dim") != "auto" else size_config.get("hidden_dim_phi"))
    set_default(size_config, "mlp_dim_phi",
                size_config.get("mlp_dim") if size_config.get("mlp_dim") != "auto" else size_config.get("hidden_dim_phi") * 4)
        
def merge_model_and_size_configs(model_config, size_config):

    for k, v in size_config.items():
        
        # Prune unused fields for each model (for clarity)
        if "fst" in model_config.model_name.lower() and k in ("hidden_dim", "embed_dim", "mlp_dim"):
            continue
        if "transformer" in model_config.model_name.lower() and k in ("n_heads_f","n_heads_phi", "hidden_dim_f", "hidden_dim_phi", "embed_dim_f","embed_dim_phi", "mlp_dim_f", "mlp_dim_phi"):
            continue   
        
        model_config[k] = v