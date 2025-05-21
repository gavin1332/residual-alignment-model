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
