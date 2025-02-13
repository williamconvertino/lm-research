import tiktoken

class Tokenizer:

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self):

        tokenizer_base = tiktoken.get_encoding("r50k_base")
        num_base_tokens = len(tokenizer_base)

        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|end_of_text|>"

        special_tokens = [
            self.bos_token,
            self.eos_token
        ]

        self.special_tokens = {
            token: i + num_base_tokens for i, token in enumerate(special_tokens)
        }

        self.tokenizer = tiktoken.Encoding(
            name="tokenizer",
            pat_str=self.pat_str,
            mergeable_ranks=tokenizer_base.mergeable_ranks,
            special_tokens=self.special_tokens
        )
    
    def encode(self, text, eos=True, bos=True):
        
        sequence = []
        if bos:
            sequence.append(self.special_tokens["<|begin_of_text|>"])

        sequence.extend(self.tokenizer.encode(text))

        if eos:
            sequence.append(self.special_tokens["<|end_of_text|>"])

        return sequence

    def decode(self, sequence):
        return self.tokenizer.decode(sequence)
