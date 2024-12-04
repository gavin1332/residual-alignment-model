# encoding:utf-8
#    Author:  a101269
#    Date  :  2024/11/29
import argparse
import json
import os
import re
import torch
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=str,
    required=True,
    help="The path to the file",
)
parser.add_argument(
    "--model_id",
    type=str,
    default="/private/model/PKU-Alignment/beaver-7b-v1.0-reward/",
    help="The path to the model",
)

args = parser.parse_args()
model_id = args.model_id
"""
Llama-Guard-3 :"/private/model/meta-llama/Llama-Guard-3-8B"
PKU-Alignment/beaver-7b-v3.0-reward
"""
device = "cuda"
dtype = torch.bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForScore.from_pretrained(model_id, torch_dtype=dtype, device_map="auto")
model.eval()
# l /private/model/PKU-Alignment/beaver-7b-v1.0-reward/
def formating(file):
    if "PKU" in file:
        with open("/private/home/liudianqing/projects/direct-preference-optimization-main/data/PKU-SafeRLHF-test.json") as fr:
            questions=[re.sub("Human: |\n\nAssistant:","",q) for q in json.load(fr)]
    elif "hh" in file:
        questions=[]
        with open("/private/home/liudianqing/projects/direct-preference-optimization-main/data/hh-test.json") as fr:
            for line in fr:
                obj = json.loads(line.strip())
                lst = [_ for _ in re.split("\n\nHuman:|\n\nAssistant:", obj["prompt"]) if _]
                questions.append(lst[0])

    chat=[]
    with open(file) as fr:
        for line in fr:
            obj = json.loads(line.strip())
            ques=questions[obj["question_id"]-1]
            answ=obj["answer"]
            prompt = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT: {output}'
            input = prompt.format(input=ques,output=answ)
            chat.append(input)
        return chat


if __name__ == "__main__":

    chats=formating(args.file)
    scores=0
    i=0
    with torch.no_grad():
        for chat in tqdm(chats):
            if len(chat.split(" "))>300:
                chat=" ".join(chat.split(" ")[:300])
            input_ids = tokenizer(chat, return_tensors='pt').to(device)
            res = model(**input_ids)
            scores+=res["end_scores"]
            # print(res["end_scores"])
    scores/=len(chats) # pythia12b_sft_norm_mix-2.8639

    print(f"BeaverReward result: {scores}/{len(chats)}:", scores)
    """
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python LlamaBeaverReward.py --file data/PKU/model_answer/pythia28b-sft-norm-mix.jsonl
pythia12b_sft_norm_mix       -2.8639
pythia12b-dpo-norm-safe-sftpku-2e7-05.jsonl  1.6281
Ours-DPO-MDS-pku.jsonl  0.6904                          
pythia28b-sft-norm-mix.jsonl -3.7318
pythia28b-dpo-norm-safe-sftpku-2e7-01  -3.1494

    """