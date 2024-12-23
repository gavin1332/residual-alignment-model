import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
import warnings
import argparse

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template
from fastchat.utils import is_partial_stop, is_sentence_complete
from peft import PeftModel
from tqdm import tqdm


def info(message):
    print(message, file=sys.stderr)


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


class SPARLogitsWarper(LogitsWarper):
    def __init__(self, base_model, top_k: int, top_p: float, temperature: float):
        self.base_model = base_model
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, base_out=None):
        base_out = self.base_model(input_ids=input_ids, use_cache=True,
                                   past_key_values=base_out.past_key_values if base_out else None)

        import torch.nn.functional as F
        # def js_div(a, b, reduction):
        # #非标准JS散度，阈值0.3
        #     return 0.5 * F.kl_div(F.log_softmax(a, dim=-1), F.softmax(b, dim=-1), reduction=reduction) + \
        #            0.5 * F.kl_div(F.log_softmax(b, dim=-1), F.softmax(a, dim=-1), reduction=reduction)

        def calculate_kl_divergence(probs_p, probs_q):
            epsilon = 1e-10
            probs_p = probs_p + epsilon
            probs_q = probs_q + epsilon
            kl_div = F.kl_div(probs_q.log(), probs_p, reduction='sum')  # 计算KL散度
            return kl_div

        def calculate_js_divergence(logits_p, logits_q):
            p = F.softmax(logits_p, dim=-1)
            q = F.softmax(logits_q, dim=-1)
            m = 0.5 * (p + q)
            kl_pm = calculate_kl_divergence(p, m)
            kl_qm = calculate_kl_divergence(q, m)
            js_div = 0.5 * (kl_pm + kl_qm)
            return js_div

        js = calculate_js_divergence(base_out.logits[:, -1, :], logits)#, reduction='sum')
        # print(js)
        base_logits = top_k_top_p_filtering(base_out.logits[:, -1, :],
                                            top_k=self.top_k, top_p=self.top_p, temperature=self.temperature)
        if js.item() <= 0.1:
            logits_warped = torch.where(base_logits == -float('Inf'), -float('Inf'), logits)
        else:
            logits_warped = base_logits
        return logits_warped, base_out

class MDSLogitsWarper(LogitsWarper):
    def __init__(self, base_model, base_temperature: float, temperature: float):
        self.base_model = base_model
        self.base_temperature = base_temperature
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, base_out=None):
        base_out = self.base_model(input_ids=input_ids, use_cache=True,
                past_key_values=base_out.past_key_values if base_out else None)

        base_probs = torch.nn.functional.softmax(base_out.logits[:, -1, :] / self.base_temperature, dim=-1)
        probs = torch.nn.functional.softmax(logits / self.temperature, dim=-1)
        probs_mul = base_probs * probs
        probs = probs_mul / probs_mul.sum(dim=-1, keepdim=True)

        epsilon = 1e-10
        logits_warped = torch.log(probs + epsilon)
        return logits_warped, base_out


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
    logits_warper: LogitsWarper = None,
):
    if hasattr(model, "device"):
        device = model.device

    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    logprobs = params.get("logprobs", None)  # FIXME: Support logprobs>1.
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    start_ids = torch.as_tensor([input_ids], device=device)

    past_key_values = out = None
    token_logprobs = [None]  # The first token has no logprobs.
    sent_interrupt = False
    finish_reason = None
    stopped = False
    base_out = None
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            out = model(input_ids=start_ids, use_cache=True)
            logits = out.logits
            past_key_values = out.past_key_values

            if logprobs is not None:
                # Prefull logprobs for the prompt.
                shift_input_ids = start_ids[..., 1:].contiguous()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = torch.log_softmax(shift_logits, dim=-1).tolist()
                for label_id, logit in zip(
                    shift_input_ids[0].tolist(), shift_logits[0]
                ):
                    token_logprobs.append(logit[label_id])
        else:  # decoding
            start_ids = torch.as_tensor([[token] if not sent_interrupt else output_ids], device=device)
            out = model(input_ids=start_ids, use_cache=True,
                past_key_values=past_key_values if not sent_interrupt else None)
            sent_interrupt = False
            logits = out.logits
            past_key_values = out.past_key_values

        last_token_logits = logits[:, -1, :]

        if logits_warper:
            last_token_logits, base_out = logits_warper(start_ids, last_token_logits,
                    base_out if i > 0 else None)

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, last_token_logits)[0]

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)
        if logprobs is not None:
            # Cannot use last_token_logits because logprobs is based on raw logits.
            token_logprobs.append(
                torch.log_softmax(logits[0, -1, :], dim=-1)[token].tolist()
            )

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            ret_logprobs = None
            if logprobs is not None:
                ret_logprobs = {
                    "text_offset": [],
                    "tokens": [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    "token_logprobs": token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    "top_logprobs": [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs["tokens"]:
                    ret_logprobs["text_offset"].append(curr_pos)
                    curr_pos += len(text)

            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "logprobs": ret_logprobs,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = "length"

    if stopped:
        finish_reason = "stop"

    yield {
        "text": output,
        "logprobs": ret_logprobs,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    if base_out:
        del base_out.past_key_values, base_out
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_path: str, dtype: str='bfloat16', lora_path: str=None, tokenizer_path: str=None):
    if dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = 'auto'

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config,
            torch_dtype=torch_dtype, device_map='auto', trust_remote_code=True)

    if lora_path:
        info('Loading lora model ...')
        model = PeftModel.from_pretrained(model, lora_path, device_map='auto')

    tokenizer = None
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True,
                trust_remote_code=True, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--lora_model', type=str, help='If None, perform inference on the base model')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'auto'], default='auto')
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--template_name', type=str, required=True, help='Prompt template name, such as "qwen2", "llama2". Refers to conversation.py')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    # corpus
    parser.add_argument('--data_file', type=str, help='A file that contains instructions')
    parser.add_argument('--output_file', type=str, help='output file, default to sys.stdout')
    parser.add_argument('--response_prefix', type=str, default='')
    parser.add_argument('--return_prefix', action='store_true')
    parser.add_argument('--input_key', type=str, default='prompt')
    parser.add_argument('--output_key', type=str, default='output')
    # generate config
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--context_len', type=int, default=1024, help='Total number of tokens in prompt and generated text')
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    # arguments for sPAR and MDS
    parser.add_argument('--base_model', type=str, default=None)
    parser.add_argument('--sample_mode', choices=['spar', 'mds'], default='spar')
    parser.add_argument('--base_top_k', type=int, default=-1)
    parser.add_argument('--base_top_p', type=float, default=0.95)
    parser.add_argument('--base_temperature', type=float, default=1.0)
    args = parser.parse_args()
    info(args)

    # prepare data
    if args.data_file is None:
        examples = [{args.input_key: 'How are you?'},
                    {args.input_key: 'Where are you from?'}]
    else:
        with open(args.data_file, encoding='utf-8') as fin:
            data_ext = os.path.splitext(args.data_file)[1]
            if data_ext == '.json':
                examples = json.load(fin)
            elif data_ext == '.jsonl':
                examples = [json.loads(line) for line in fin]
            else:
                try:
                    examples = [json.loads(line) for line in fin]
                except json.JSONDecodeError as e:
                    fin.seek(0)
                    examples = json.load(fin)

    info('first 10 examples:')
    for example in examples[:10]:
        info(example)

    if args.output_file:
        file_out = open(args.output_file, 'w', encoding='utf-8')
        output_ext = os.path.splitext(args.output_file)[1]
        # write to json file could not flush one line per example
        flush = False if output_ext == '.json' else True
    else:
        file_out = sys.stdout
        flush = True

    prompt_template = get_conv_template(args.template_name)
    print(f'prompt_template:\n{prompt_template}')

    stop_str = prompt_template.sep + prompt_template.roles[0]
    stop_str_display = stop_str.replace("\n", "\\n")
    print(f'stop_str: {stop_str_display}')

    device = torch.device(0)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model
    model, tokenizer = load_model(args.model, args.dtype, args.lora_model, args.tokenizer_path)
    model.eval()

    info(tokenizer)

    base_model = None
    if args.base_model:
        base_model, _ = load_model(args.base_model, args.dtype)
        base_model.eval()

        if args.resize_emb:
            base_model_vocab_size = base_model.get_input_embeddings().weight.size(0)
            model_vocab_size = model.get_input_embeddings().weight.size(0)
            tokenzier_vocab_size = len(tokenizer)
            info(f'Vocab of the base model: {base_model_vocab_size}')
            info(f'Vocab of the expert model: {model_vocab_size}')
            info(f'Vocab of the tokenizer: {tokenzier_vocab_size}')
            if base_model_vocab_size != tokenzier_vocab_size or model_vocab_size != tokenzier_vocab_size:
                info('Resize model embeddings to fit tokenizer')
                base_model.resize_token_embeddings(tokenzier_vocab_size)
                model.resize_token_embeddings(tokenzier_vocab_size)

    logits_warper = None
    if base_model:
        if args.sample_mode == 'spar':
            logits_warper = SPARLogitsWarper(base_model,
                    top_k=args.base_top_k, top_p=args.base_top_p, temperature=args.base_temperature)
        elif args.sample_mode == 'mds':
            logits_warper = MDSLogitsWarper(base_model,
                    base_temperature=args.base_temperature, temperature=args.temperature)
            args.temperature = 1.0
        else:
            raise NotImplementedError('Not supported yet.')
  
    info(f'Start inference at device {device} ...')

    for i, example in enumerate(tqdm(examples, desc='Generating outputs')):
        conv = get_conv_template(args.template_name)
        # single round
        conv.append_message(conv.roles[0], example[args.input_key])
        conv.append_message(conv.roles[1], args.response_prefix if args.response_prefix else None)
        prompt = conv.get_prompt()

        # FIXME(liuyi): used only for training format matching
        if prompt.startswith('\n\n'):
            prompt = prompt[2:]

        if i == 0:
            print(f'{"="*30} 1st prompt\n{prompt}\n{"="*30}')
        
        gen_params = dict(
            prompt=prompt,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            stop=stop_str,
            echo=False,
        )

        for ans in generate_stream(
                model=model,
                tokenizer=tokenizer,
                params=gen_params,
                device=device,
                context_len=args.context_len,
                stream_interval=5, # force this for-loop running less iterations
                logits_warper=logits_warper):
            pass

        response = ans['text'].rstrip()

        if i == 0:
            print(f'{"="*30} 1st response\n{response}\n{"="*30}')

        if args.return_prefix:
            respose = args.response_prefix + response

        example[args.output_key] = response
        if flush: # to jsonl format or sys.stdout
            print(json.dumps(example, ensure_ascii=False), file=file_out, flush=True)

    if not flush: # to json format
        json.dump(examples, ensure_ascii=False, file=file_out, indent=2)

    info(f'save to {args.output_file}')

    if args.output_file:
        file_out.close()


if __name__ == '__main__':
    main()

