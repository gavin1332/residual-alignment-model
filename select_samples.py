import re
import json
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from sampling import default_build_prompt as build_prompt


def build_sample(ex, input_key='instruction', output_key='output'):
    prompt = build_prompt(ex)
    output = ex[output_key]
    return prompt, output


max_length = 2048
max_prompt_length = 1024
max_new_tokens = max_length - max_prompt_length - 1

tokenizer = AutoTokenizer.from_pretrained('/private/model/Qwen/Qwen2.5-14B')

dataset = 'uc'
if dataset == 'uc':
    json_file_path = '_data/train/bak/ultrachat_round_one_sampling_qwen14bwu.json'
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)
        random.shuffle(samples)
    
    N = 120000
    filtered_samples = []
    for sample in tqdm(samples):
        del sample['sampling']
        prompt, output = build_sample(sample)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        output_tokens = tokenizer.encode(output, add_special_tokens=False)
        if len(prompt_tokens) <= max_prompt_length and len(output_tokens) <= max_new_tokens:
            filtered_samples.append(sample)
            if len(filtered_samples) == N:
                break
    
    with open('_data/train/ultrachat_round_one.json', 'w') as fout:
        json.dump(filtered_samples, fout, ensure_ascii=False, indent=2)

elif dataset == 'hh':
    split = 'train'
    N = 300000000000
    json_file_path = f'hub/Anthropic/hh-rlhf/{split}.json'
    
    with open(json_file_path, 'r', encoding='utf-8') as file:
        samples = json.load(file)
        random.shuffle(samples)
    
    filtered_samples = []
    for sample in tqdm(samples):
        instruction, chosen, rejected, prompt = build_hh_sample(sample, tokenizer)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        chosen_tokens = tokenizer.encode(chosen, add_special_tokens=False)
        rejected_tokens = tokenizer.encode(rejected, add_special_tokens=False)
        if len(prompt_tokens) <= max_prompt_length and len(chosen_tokens) <= max_new_tokens and len(rejected_tokens) <= max_new_tokens:
            filtered_samples.append(dict(instruction=instruction, chosen=chosen, rejected=rejected))
            if len(filtered_samples) == N:
                break

    print(f'valid ratio: {len(filtered_samples)/len(samples)}')
    
    with open(f'hh_{split}.json', 'w') as fout:
        json.dump(filtered_samples, fout, ensure_ascii=False, indent=2)

