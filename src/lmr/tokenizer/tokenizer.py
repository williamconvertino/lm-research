import tiktoken
import re

class Tokenizer:
    
    _instance = None
    
    def __init__(self, base_type="gpt2"):
        
        self.base_type = base_type
        
        base_enc = tiktoken.get_encoding(base_type)

        special_tokens = {
            "<|pad|>": base_enc.n_vocab,
            "<|bos|>": base_enc.n_vocab + 1,
            "<|eos|>": base_enc.n_vocab + 2,
        }
        
        for i in range(len(special_tokens), 10):
            special_tokens[f"<|reserved_{i}|>"] = base_enc.n_vocab + i

        self.set_special_tokens(special_tokens)

        number_pattern = r"\d{1,3}"  # Match 0–999 as separate tokens
        fallback_pattern = base_enc._pat_str
        custom_pat_str = f"{number_pattern}|{fallback_pattern}"

        self.tokenizer = tiktoken.Encoding(
            name="tokenizer",
            pat_str=custom_pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens=special_tokens,
        )
    
    def set_special_tokens(self, token_map):
        self.special_tokens = token_map
        for token, idx in self.special_tokens.items():
            name_match = re.match(r"<\|([a-zA-Z0-9_]+)\|>", token)
            if name_match:
                name = name_match.group(1)
                setattr(self, f"{name}_token_id", idx)

    def is_special(self, token_id):
        return token_id in self.special_tokens.values()

    def clean_tokens(self, tokens):
        return [t for t in tokens if t != self.pad_token_id and t != self.eos_token_id]

    def encode(self, text):
        return self.tokenizer.encode(text, disallowed_special=())

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab
    
    @staticmethod
    def get_instance():
        if Tokenizer._instance is None:
            Tokenizer._instance = Tokenizer()
        return Tokenizer._instance
    
    @staticmethod
    def set_instance(tokenizer):
        Tokenizer._instance = tokenizer