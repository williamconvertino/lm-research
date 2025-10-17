import torch
import torch.nn.functional as F
from lm_eval.api.model import LM

class BenchmarkWrapper(LM):
    def __init__(self, model, tokenizer, device="cuda", batch_size=1, dtype=torch.float32):
        super().__init__()
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._dtype = dtype

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_seq_len

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests):
        outputs = []

        for batch_start in range(0, len(requests), self.batch_size):
            batch = requests[batch_start : batch_start + self.batch_size]

            context_enc = [self.tok_encode(req.args[0]) for req in batch]
            cont_enc = [self.tok_encode(req.args[1]) for req in batch]

            input_ids = [
                torch.tensor(c + t, dtype=torch.long)
                for c, t in zip(context_enc, cont_enc)
            ]

            max_len = max(x.size(0) for x in input_ids)
            input_tensor = torch.full((len(batch), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
            for i, x in enumerate(input_ids):
                input_tensor[i, :x.size(0)] = x

            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor).to(dtype=self._dtype)

            for i, req in enumerate(batch):
                c_ids = context_enc[i]
                t_ids = cont_enc[i]
                ctx_len = len(c_ids)

                logits_slice = logits[i, ctx_len - 1 : ctx_len - 1 + len(t_ids)]
                targets = torch.tensor(t_ids, device=self.device, dtype=torch.long)

                if logits_slice.size(0) != targets.size(0):
                    outputs.append((0.0, False))
                    continue

                log_probs = F.log_softmax(logits_slice, dim=-1)
                selected_log_probs = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)

                if torch.isnan(selected_log_probs).any() or torch.isinf(selected_log_probs).any():
                    outputs.append((0.0, False))
                else:
                    outputs.append((selected_log_probs.sum().item(), True))

        return outputs

    def loglikelihood_rolling(self, requests):
        max_seq_len = self.model.config.max_seq_len  # e.g., 1024

        res = []
        for context, continuation in requests:
            # Concatenate and tokenize
            full_input = self.tokenizer.encode(context + continuation)
            full_input_ids = torch.tensor(full_input, dtype=torch.long, device=self.device)

            # Rolling evaluation: split into overlapping windows
            rolling_loglikelihood = 0.0
            total_tokens = 0

            for i in range(0, len(full_input_ids), max_seq_len):
                input_ids = full_input_ids[i : i + max_seq_len]

                if len(input_ids) <= 1:
                    continue

                input_ids = input_ids.unsqueeze(0)  # [1, L]
                with torch.no_grad():
                    logits = self.model(input_ids[:, :-1])  # [1, L-1, V]
                    log_probs = torch.log_softmax(logits, dim=-1)  # [1, L-1, V]

                target_ids = input_ids[:, 1:]  # [1, L-1]

                L = min(log_probs.shape[1], target_ids.shape[1])
                log_probs = log_probs[:, -L:, :]
                target_ids = target_ids[:, -L:]

                selected_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # [1, L]
                rolling_loglikelihood += selected_log_probs.sum().item()
                total_tokens += L

            res.append((rolling_loglikelihood, True))  # True = no truncation flag

        return res

    def generate_until(self, requests):
        generations = []
        for context, _ in requests:
            input_ids = torch.tensor([self.tok_encode(context)], device=self.device)
            output_ids = self.model.generate(
                input_ids,
                max_generation_length=self.max_length,
                tokenizer=self.tokenizer,
                return_generation_only=True
            )
            generations.append(self.tok_decode(output_ids[0].tolist()))
        return generations