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

from logits_warper import (
    LogitsWarper,
    ParKlLogitsWarper,
    ParLogitsWarper,
    MdsLogitsWarper,
)


def info(message):
    print(message, file=sys.stderr)


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
    if hasattr(model, 'device'):
        device = model.device

    # Read parameters
    prompt = params['prompt']
    len_prompt = len(prompt)
    temperature = float(params.get('temperature', 1.0))
    repetition_penalty = float(params.get('repetition_penalty', 1.0))
    top_p = float(params.get('top_p', 1.0))
    top_k = int(params.get('top_k', -1))  # -1 means disable
    max_new_tokens = int(params.get('max_new_tokens', 256))
    logprobs = params.get('logprobs', None)  # FIXME: Support logprobs>1.
    echo = bool(params.get('echo', True))
    stop_str = params.get('stop', None)
    stop_token_ids = params.get('stop_token_ids', None) or []
    if tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt, add_special_tokens=False).input_ids

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
                    'text_offset': [],
                    'tokens': [
                        tokenizer.decode(token)
                        for token in (
                            output_ids if echo else output_ids[input_echo_len:]
                        )
                    ],
                    'token_logprobs': token_logprobs
                    if echo
                    else token_logprobs[input_echo_len:],
                    'top_logprobs': [{}]
                    * len(token_logprobs if echo else token_logprobs[input_echo_len:]),
                }
                # Compute text_offset
                curr_pos = 0
                for text in ret_logprobs['tokens']:
                    ret_logprobs['text_offset'].append(curr_pos)
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
                    raise ValueError('Invalid stop field type.')

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    'text': output,
                    'logprobs': ret_logprobs,
                    'usage': {
                        'prompt_tokens': input_echo_len,
                        'completion_tokens': i,
                        'total_tokens': input_echo_len + i,
                    },
                    'finish_reason': None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    else:
        finish_reason = 'length'

    if stopped:
        finish_reason = 'stop'

    yield {
        'text': output,
        'logprobs': ret_logprobs,
        'usage': {
            'prompt_tokens': input_echo_len,
            'completion_tokens': i,
            'total_tokens': input_echo_len + i,
        },
        'finish_reason': finish_reason,
    }

    # Clean
    del past_key_values, out
    if base_out:
        del base_out.past_key_values, base_out
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_path: str, dtype: str, archive: str=None, lora_path: str=None, tokenizer_path: str=None):
    if dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = 'auto'

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch_dtype,
            device_map='auto', attn_implementation='flash_attention_2', trust_remote_code=True)
            #device_map='auto', trust_remote_code=True)

    if archive:
        state_dict = torch.load(archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')

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


class Inference:
    def __init__(self, args):
        info(f'Inference arguments:\n{args}')

        self.device = torch.device(0)

        if args.tokenizer_path is None:
            args.tokenizer_path = args.model
        self.model, self.tokenizer = load_model(args.model, args.dtype,
                archive=args.archive, lora_path=args.lora_model, tokenizer_path=args.tokenizer_path)
        self.model.eval()

        info(self.tokenizer)

        self.base_model = None
        if args.base_model:
            self.base_model, _ = load_model(args.base_model, args.dtype)
            self.base_model.eval()

            if args.resize_emb:
                base_model_vocab_size = self.base_model.get_input_embeddings().weight.size(0)
                model_vocab_size = self.model.get_input_embeddings().weight.size(0)
                tokenzier_vocab_size = len(self.tokenizer)
                info(f'Vocab of the base model: {base_model_vocab_size}')
                info(f'Vocab of the expert model: {model_vocab_size}')
                info(f'Vocab of the tokenizer: {tokenzier_vocab_size}')
                if base_model_vocab_size != tokenzier_vocab_size or model_vocab_size != tokenzier_vocab_size:
                    info('Resize model embeddings to fit tokenizer')
                    self.base_model.resize_token_embeddings(tokenzier_vocab_size)
                    self.model.resize_token_embeddings(tokenzier_vocab_size)

        self.logits_warper = None
        if self.base_model:
            if args.sample_mode == 'par':
                self.logits_warper = ParLogitsWarper(self.base_model,
                        top_k=args.base_top_k, top_p=args.base_top_p, temperature=args.base_temperature)
            elif args.sample_mode == 'par_kl':
                self.logits_warper = ParKlLogitsWarper(self.base_model, top_k=args.base_top_k, top_p=args.base_top_p,
                        temperature=args.base_temperature, kl_div_threshold=args.div_threshold)
            elif args.sample_mode == 'mds':
                self.logits_warper = MdsLogitsWarper(self.base_model, base_temperature=args.base_temperature,
                        expert_temperature=args.expert_temperature, eos_priority=args.eos_priority)
            else:
                raise NotImplementedError(f'Not supported yet. sample_mode={args.sample_mode}')
  
        info(f'Start inference at device {self.device} ...')

        self.args = args

    def generate(self, prompt):
        gen_params = dict(
            prompt=prompt,
            temperature=self.args.temperature,
            repetition_penalty=self.args.repetition_penalty,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_new_tokens=self.args.max_new_tokens,
            stop=self.args.stop_str,
            echo=False,
        )
        for ans in generate_stream(
                model=self.model,
                tokenizer=self.tokenizer,
                params=gen_params,
                device=self.device,
                context_len=self.args.max_length,
                stream_interval=5, # force this for-loop running less iterations
                logits_warper=self.logits_warper):
            pass

        return ans['text'].rstrip()

