from hydra.utils import get_class

from lmr.tokenizer import Tokenizer

def get_model(model_config, vocab_size):
    model_config["vocab_size"] = vocab_size
    model = get_class(model_config.model_class)(model_config)
    model.name = model_config.model_name
    model.full_name = f"{model_config.model_name}_{model_config.size_name}"
    return model 