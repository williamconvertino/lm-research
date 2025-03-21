from models.tttransformer import TTTransformer
from types import SimpleNamespace

config = {
    "model": "tttransformer",
    "max_seq_len": 128,
    "d_embed": 512,
    "n_layers": 3,
    "n_heads": 8,
    "dropout": 0.1,
    "ss_pct": 0.99,
    "vocab_size": 50257
}

config = SimpleNamespace(**config)

model = TTTransformer(config)

total_params = sum(p.numel() for p in model.parameters())
embed_params = sum(p.numel() for p in model.embedding.parameters())
non_embed_params = total_params - embed_params

print(f"Total parameters: {total_params}")
print(f"Parameters without embedding: {non_embed_params}")