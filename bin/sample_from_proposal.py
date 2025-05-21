import sys
import os
import argparse
import subprocess
from typing import Dict
from sampling import ParallelSampler, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--max_length', type=int, default=2048)
parser.add_argument('--max_prompt_length', type=int, default=1024)
parser.add_argument('--multiround', action='store_true')
parser.add_argument('--for_aligner', action='store_true')
parser.add_argument('-g', '--gpu_ids', type=str, default='0,1,2,3,4,5,6,7')
args = parser.parse_args()

#input_file = 'data/test/alpaca_eval_ques.json'
#output_file = f'haha.json'
#model_path = '/private/model/meta-llama/Llama-3.1-8B'

#input_file = 'data/train/alpaca_data_cleaned.json'
#output_file = f'data/train/alpaca_sampling_{generator}.json'
#model_path = '/private/model/Qwen/Qwen2.5-14B'

#input_file = 'data/train/ultrachat_round_one.json'
#output_file = 'data/train/ultrachat_round_one_sampling_llama8bwu.json'
#model_path = '_output/model/wu_sft_uc_llama8b_b64_lr1e-06/LATEST'

#input_file = 'data/train/openai_summarize_tldr.jsonl'
#output_file = 'data/train/openai_summarize_tldr_sampling_qwen14b.jsonl'
#model_path = '_output/model/wu_sft_summ-sft_qwen14b_b64_lr1e-06/LATEST'

print(f'Start sampling from {args.model_path} ...')

sampling_params = SamplingParams(
    model=args.model_path,
    max_length=args.max_length, #1152,
    max_prompt_length=args.max_prompt_length, #640,
    temperature=0.5,
    top_p=0.95,
    top_k=10,
    repetition_penalty=1.05,
    multiround=args.multiround,
    for_aligner=args.for_aligner,
)

def build_record(record: Dict, output: str, generator: str=None, output_key: str='output'):
    record['sampling'] = output 
    return record

sampler = ParallelSampler(gpu_ids_str=args.gpu_ids,
                          sampling_params=sampling_params,
                          build_record=build_record,
                          use_vllm=True)

sampler.do_sample(args.input_file, args.output_file)
