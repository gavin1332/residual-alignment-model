import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k: int, top_p: float, temperature: float=1.0, filter_value: float=-float("Inf"), min_tokens_to_keep: int=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits / temperature, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0 
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0 

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


class LogitsWarper:
    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, base_out=None):
        raise NotImplementedError('Not implemented yet.')


class ParLogitsWarper(LogitsWarper):
    def __init__(self, base_model, top_k: int, top_p: float, temperature: float):
        self.base_model = base_model
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, base_out=None):
        base_out = self.base_model(input_ids=input_ids, use_cache=True,
                                   past_key_values=base_out.past_key_values if base_out else None)

        base_logits = top_k_top_p_filtering(base_out.logits[:, -1, :], top_k=self.top_k, top_p=self.top_p, temperature=self.temperature)
        if self.base_model.generation_config.eos_token_id == torch.argmax(base_logits, dim=-1)[0]:
            logits_warped = base_logits.fill_(-float('Inf'))
            logits_warped[0][self.base_model.generation_config.eos_token_id] = 1
        else:
            logits_warped = torch.where(base_logits == -float('Inf'), -float('Inf'), logits)
        return logits_warped, base_out

class ParEosLogitsWarper(LogitsWarper):
    def __init__(self, base_model, top_k: int, top_p: float, temperature: float):
        self.base_model = base_model
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, base_out=None):
        base_out = self.base_model(input_ids=input_ids, use_cache=True,
                                   past_key_values=base_out.past_key_values if base_out else None)

        base_logits = top_k_top_p_filtering(base_out.logits[:, -1, :], top_k=self.top_k, top_p=self.top_p, temperature=self.temperature)
        next_tokens = torch.argmax(base_logits, dim=-1)
        if next_tokens[0] == self.base_model.generation_config.eos_token_id:
            logits_warped = base_logits.fill_(-float('Inf'))
            logits_warped[0][self.base_model.generation_config.eos_token_id] = 1
        else:
            logits_warped = torch.where(base_logits == -float('Inf'), -float('Inf'), logits)
        return logits_warped, base_out

class ParKlLogitsWarper(LogitsWarper):
    def __init__(self, base_model, top_k: int, top_p: float, temperature: float, kl_div_threshold: float):
        self.base_model = base_model
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.kl_div_threshold = kl_div_threshold

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, base_out=None):
        base_out = self.base_model(input_ids=input_ids, use_cache=True,
                                   past_key_values=base_out.past_key_values if base_out else None)

        base_logits = base_out.logits[:, -1, :]
        base_probs = F.softmax(base_logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
        # higher base_probs -> higher probs, but no need for vice versa
        kl_div = F.kl_div(logprobs, base_probs, reduction='sum')

        if kl_div.item() <= self.kl_div_threshold:
            base_logits = top_k_top_p_filtering(base_logits, top_k=self.top_k, top_p=self.top_p,
                    temperature=self.temperature)
            logits_warped = torch.where(base_logits == -float('Inf'), -float('Inf'), logits)
        else:
            logits_warped = base_logits
        return logits_warped, base_out


class MdsLogitsWarper(LogitsWarper):
    def __init__(self, base_model, base_temperature: float, expert_temperature: float, eos_priority: bool=False):
        self.base_model = base_model
        self.base_temperature = base_temperature
        self.expert_temperature = expert_temperature
        self.eos_priority = eos_priority

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, base_out=None):
        base_out = self.base_model(input_ids=input_ids, use_cache=True,
                past_key_values=base_out.past_key_values if base_out else None)

        base_logits = base_out.logits[:, -1, :]

        if self.eos_priority and self.base_model.generation_config.eos_token_id == torch.argmax(base_logits, dim=-1)[0]:
            logits_warped = base_logits.fill_(-float('Inf'))
            logits_warped[0][self.base_model.generation_config.eos_token_id] = 1
        else:
            base_probs = F.softmax(base_logits / self.base_temperature, dim=-1)
            probs = F.softmax(logits / self.expert_temperature, dim=-1)
            probs_mul = base_probs * probs
            probs = probs_mul / probs_mul.sum(dim=-1, keepdim=True)

            epsilon = 1e-10
            logits_warped = torch.log(probs + epsilon)

        return logits_warped, base_out

