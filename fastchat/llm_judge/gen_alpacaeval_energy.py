"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from models.llama.pythia_modeling_emb_xxxxx import GPTNeoXForCausalLM
from transformers import GPTNeoXForCausalLM as AlpacaLM
from transformers import AutoTokenizer
import re

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    energy_model_path
):
    get_model_answers(
            model_path,
            model_id,
            answer_file,
            max_new_token,
            num_choices,
            num_gpus_per_model,
            max_gpu_memory,
            dtype=dtype,
            revision=revision,
            energy_model_path=energy_model_path
        )


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    energy_model_path
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = 0
    model = GPTNeoXForCausalLM.from_pretrained(model_path,device_map = 'auto', torch_dtype=torch.bfloat16)
    expert=AlpacaLM.from_pretrained(energy_model_path,device_map = 'auto',torch_dtype=torch.bfloat16)
    expert.config.pad_token_id = 1
    expert.generation_config.pad_token_id = 1
    expert.config.eos_token_id = 0
    expert.generation_config.eos_token_id = 0
    import re
    if 1:
        # tokenzier_vocab_size = 50432
        tokenzier_vocab_size = len(tokenizer)
        print("tokenzier_vocab_size ", tokenzier_vocab_size)
        model.resize_token_embeddings(tokenzier_vocab_size)
        expert.resize_token_embeddings(tokenzier_vocab_size)
    # model.resize_token_embeddings(tokenzier_vocab_size)
    model.add_module('energy', expert)
    model.config.pad_token_id = 1
    model.generation_config.pad_token_id = 1
    model.config.eos_token_id = 0
    model.generation_config.eos_token_id = 0
    model.eval()

    questions = []
    import datasets
    i = 0
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    for example in eval_set:
        i += 1
        example["instruction"]= "Human:\n"+ example["instruction"]+"\n\nAssistant:"
        print(example["instruction"])
        questions.append({"turns": [], "question_id": i})
        questions[-1]["turns"].append(example["instruction"])
    examples=[e for e in eval_set]
    eid=-1
    for question in tqdm(questions):
        eid+=1
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("zero_shot")
            turns = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids
                print(prompt)

                # some models may error out when generating long outputs
                # try:
                if 1:
                    model_inputs = tokenizer([prompt], add_special_tokens=True, return_tensors="pt")
                    device = torch.device(0)
                    model_inputs = model_inputs.to(device)
                    output_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_token,
                        temperature=1,
                        do_sample=True,
                    )
                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                # except RuntimeError as e:
                #     print("ERROR question ID: ", question["question_id"])
                #     output = "ERROR"

                conv.update_last_message(output)
                turns.append(output)

            answer=turns[0]

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "answer": answer,
            }
            examples[eid]["output"] = answer
            examples[eid]["generator"] = model_id
            print("answer:\n", answer)
            # fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
        name = answer_file.split('/')[-1].split('.')[0]
        path = "/".join(answer_file.split('/')[:-1])
        newname = path + '/' + name + '.json'
        with open(os.path.expanduser(newname), "w") as fout:
            fout.write(json.dumps(examples, ensure_ascii=False))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="PKU",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--energy_model_path",
        type=str,
        default="",
        help="energy model path",
    )
    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        energy_model_path=args.energy_model_path
    )

    reorg_answer_file(answer_file)
