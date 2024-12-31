# encoding:utf-8
#    Author:  a101269
#    Date  :  2024/9/27
import torch
import torch.nn as nn
import transformers
import os
import re
from tqdm import tqdm
import json
from transformers import GenerationConfig
from transformers import GPTNeoXForCausalLM
from transformers import AutoTokenizer
from template import summ_prompt
import argparse
import random

random.seed(123)
parser = argparse.ArgumentParser()
parser.add_argument('--filepath', type=str, required=True)
parser.add_argument('--file_prefix', type=str, required=True)
parser.add_argument('--part', type=int, required=True)
parser.add_argument('--gpu_num', type=int, required=True)
args = parser.parse_args()

def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]

generation_config = GenerationConfig(
    temperature=1,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=800
)
if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    model_name_or_path = "/private/home/liuyi/code/exp/mine/.cache/root/hf/pythia12b_hh_sft_b32.1218/step-160000"
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = 0
    model.config.pad_token_id = 1
    model.generation_config.pad_token_id = 1
    model.config.eos_token_id = 0
    model.generation_config.eos_token_id = 0
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    print(f"Vocab of the base model: {model_vocab_size}")
    print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
    if model_vocab_size!=tokenizer_vocab_size:
        print("Resize model embeddings to fit tokenizer")
        model.resize_token_embeddings(tokenizer_vocab_size)

    if device==torch.device('cpu'):
        model.float()
    model.eval()

    import datasets
    # dataset = datasets.load_dataset('Anthropic/hh-rlhf', split="train")
    # print('done')
    # def split_prompt_and_responses(ex):
    #     prompt = extract_anthropic_prompt(ex['chosen'])
    #     chosen_response = ex['chosen'][len(prompt):]
    #     rejected_response = ex['rejected'][len(prompt):]
    #     # prompt = re.sub("Human:","### Instruction: \n", prompt)
    #     # prompt = re.sub("Assistant:", "### Response:", prompt)
    #     return prompt, chosen_response, rejected_response
    # examples=[]
    # i = 0
    # for row in dataset:
    #     prompt, chosen, rejected = split_prompt_and_responses(row)
    #     examples.append({"instruction":prompt,"response":chosen})
    dataset = datasets.load_dataset('/private/home/liudianqing/corpus/rlhf/openai_summarize_tldr_sft', split="train")
    print('done')
    def split_prompt_and_responses(ex):
        ex['prompt']=re.sub("TL;DR:\s?","",ex['prompt'])
        prompt = summ_prompt.format(post=ex['prompt'])
        chosen_response = ex['label']
        rejected_response = ex['label']
        return prompt, chosen_response, rejected_response
    examples=[]
    i = 0
    for row in dataset:
        prompt, chosen, rejected = split_prompt_and_responses(row)
        examples.append({"instruction": prompt, "response": chosen})

    random.shuffle(examples)
    print(examples[0])
    inter = len(examples) // args.gpu_num
    part = args.part
    if part != args.gpu_num:
        examples = examples[(part - 1) * inter:part * inter]
    else:
        examples = examples[(part - 1) * inter:]
    print(len(examples))
    fw=open(args.filepath+'/'+args.file_prefix + str(part)+".jsonl", 'w')
    data=[]
    batchsize = 7
    with torch.no_grad():
        dataiter = tqdm(range(0, len(examples), batchsize))
        i = 0
        for start in dataiter:
            end = min((start + batchsize), len(examples))
            for example in examples[start:end]:
                if i==0:
                    print(example["instruction"])
                i+=1
            input_text = [example["instruction"] for example in examples[start:end]]
            model_inputs = tokenizer(input_text, add_special_tokens=True, padding=True, return_tensors="pt")
            model_inputs = model_inputs.to(device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responsees = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for idx in range(end - start):
                examples[start + idx]["sample_res"] = responsees[idx]
                data.append(examples[start + idx])
                fw.write(json.dumps(examples[start + idx]) + '\n')
                fw.flush()
    fw.close()
    with open(args.filepath+'/'+args.file_prefix + str(part) + ".json", 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
