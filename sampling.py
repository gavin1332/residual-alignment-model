import os
import re
import json
import itertools
from dataclasses import dataclass, field
from tqdm import tqdm
from multiprocessing import Process, Queue
from typing import List, Dict, Callable
from fastchat.conversation import get_conv_template
import transformers


@dataclass
class SamplingParams:
    # model
    model: str = None
    archive: str = None
    lora_model: str = None
    dtype: str = 'bfloat16'
    tokenizer_path: str = None
    resize_emb: bool = True # Whether to resize model token embeddings
    # corpus
    system_key: str = None
    input_key: str = 'instruction'
    output_key: str = 'output'
    generator: str = 'ours'
    multiround: bool = False
    for_aligner: bool = False
    # generate config
    temperature: float = 0.5
    top_p: float = 0.9
    top_k: int = -1
    max_length: int = 2048
    max_prompt_length: int = 1024
    max_new_tokens: int = field(init=False)
    repetition_penalty: float = 1.05
    stop_str: str = '\n\nHuman'
    # arguments for aligning strategy
    base_model: str = None
    base_temperature: float = 0.5
    base_top_p: float = 0.95
    base_top_k: int = -1
    sample_mode: str = 'par_kl' # par, mds, par_kl
    expert_temperature: float = None # for MDS
    div_threshold: float = 0.1
    eos_priority: bool = False # for MDS sample mode

    def __post_init__(self):
        self.max_new_tokens = self.max_length - self.max_prompt_length

    def mark(self):
        tag = f't{self.temperature}_p{self.top_p}_rp{self.repetition_penalty}'
        if self.top_k > 0:
            tag += '_k{self.top_k}'
        if self.base_model:
            tag += f'_base_t{self.base_temperature}_p{self.base_top_p}'
            if self.base_top_k > 0:
                tag += f'_k{self.base_top_k}'

            tag += f'_{self.sample_mode}'
            if self.sample_mode == 'mds':
                tag += f'_et{self.expert_temperature}'
            elif self.sample_mode != 'par':
                tag += f'_dt{self.div_threshold}'
        return tag


def default_build_prompt(record: Dict, input_key: str='instruction', output_key: str='output', multiround: bool=False, for_aligner: bool=False):
    if multiround:
        prompt = record[input_key]
    else:
        prompt = f'\n\nHuman: {record[input_key]}\n\nAssistant: '

    if for_aligner:
        prompt += f'{record[output_key]}\n\nCorrection: '

    return prompt


def default_build_record(record: Dict, output: str, generator: str=None, output_key: str='output'):
    record[output_key] = output 
    if generator:
        record['generator'] = generator
    return record


class VLLMTokenizerWrapper:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text, add_special_tokens=False, **kwargs):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._tokenizer, attr)

    def __len__(self):
        return len(self._tokenizer)


class ParallelSampler:
    def __init__(self,
                 gpu_ids_str: str,
                 sampling_params: Dict,
                 build_prompt: Callable=None,
                 build_record: Callable=None,
                 use_vllm: bool=False):
        self.gpu_ids = [int(gpu_id) for gpu_id in gpu_ids_str.split(',')]
        self.sampling_params = sampling_params
        self.build_prompt = build_prompt or default_build_prompt
        self.build_record = build_record or default_build_record
        self.use_vllm = use_vllm
        tokenizer_path = self.sampling_params.tokenizer_path
        if not self.sampling_params.tokenizer_path:
            tokenizer_path = self.sampling_params.model

    def read_json_file(self, file_path: str) -> List[Dict]:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r') as file:
                for line in file:
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError("Unsupported file format. Only JSON and JSONL are supported.")
    
    def split_data(self, data: List[Dict], num_groups: int) -> List[List[Dict]]:
        group_size = len(data) // num_groups
        remainder = len(data) % num_groups
        groups = []
        start = 0
        for i in range(num_groups):
            size = group_size + (1 if i < remainder else 0)
            groups.append(data[start:start + size])
            start += size
        return groups

    def process_group(self, worker_id: int, group: List[Dict], output_queue: Queue):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)

        if self.use_vllm:
            assert not self.sampling_params.base_model, 'base_model is not supported in vLLM inference'

            from transformers import AutoTokenizer
            import vllm

            vllm_sampling_params = vllm.SamplingParams(temperature=self.sampling_params.temperature,
                                                       top_p=self.sampling_params.top_p,
                                                       repetition_penalty=self.sampling_params.repetition_penalty,
                                                       max_tokens=self.sampling_params.max_new_tokens,
                                                       stop=self.sampling_params.stop_str,
                                                       truncate_prompt_tokens=self.sampling_params.max_prompt_length)
            llm = vllm.LLM(model=self.sampling_params.model,
                           gpu_memory_utilization=0.97,
                           device=f'cuda:0',
                           # add max_length buffer
                           max_model_len=self.sampling_params.max_length + 10)

            tokenizer_path = self.sampling_params.tokenizer_path or self.sampling_params.model
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            llm.set_tokenizer(VLLMTokenizerWrapper(tokenizer))

            prompts = []
            for i, record in enumerate(group):
                prompt = self.build_prompt(record,
                                           input_key=self.sampling_params.input_key,
                                           output_key=self.sampling_params.output_key,
                                           multiround=self.sampling_params.multiround,
                                           for_aligner=self.sampling_params.for_aligner)
                if i == 0:
                    print(f'{"="*30} Prompt\n{prompt}\n{"="*30}', flush=True)
                prompts.append(prompt)
            
            outputs = llm.generate(prompts, vllm_sampling_params)

            results = []
            for record, output in zip(group, outputs):
                record = self.build_record(record,
                                           output=output.outputs[0].text,
                                           generator=self.sampling_params.generator,
                                           output_key=self.sampling_params.output_key)
                results.append(record)

        else:
            from inference import Inference
            infer = Inference(self.sampling_params)

            results = []
            for i, record in enumerate(tqdm(group, desc='Generating outputs')):
                prompt = self.build_prompt(record,
                                           input_key=self.sampling_params.input_key,
                                           output_key=self.sampling_params.output_key,
                                           multiround=self.sampling_params.multiround,
                                           for_aligner=self.sampling_params.for_aligner)
                output = infer.generate(prompt)
                if i == 0:
                    print(f'{"="*30} Prompt\n{prompt}\n{"-"*30} Output\n{output}\n{"="*30}')

                record = self.build_record(record,
                                           output=output,
                                           generator=self.sampling_params.generator,
                                           output_key=self.sampling_params.output_key)
                results.append(record)
 
        output_queue.put((worker_id, results))
        print(f'Process {worker_id} finished.')
    
    def do_sample(self, input_path: str, output_path: str):
        print(f'Sampling parameters:\n{self.sampling_params}')

        data = self.read_json_file(input_path)
        groups = self.split_data(data, len(self.gpu_ids))

        output_queue = Queue()
        processes = []
        for worker_id, group in zip(self.gpu_ids, groups):
            p = Process(target=self.process_group, args=(worker_id, group, output_queue))
            processes.append(p)
            p.start()

        results = [output_queue.get() for _ in range(len(self.gpu_ids))]
        results.sort(key=lambda x: x[0])
        results = [item for pair in results for item in pair[1]]

        dir_path = os.path.dirname(output_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(output_path, 'w') as fout:
            if output_path.endswith('.json'):
                json.dump(results, fout, indent=2, ensure_ascii=False)
            else:
                for obj in results:
                    print(json.dumps(obj, ensure_ascii=False), file=fout)
        print(f'Finish writing to {output_path}')
    
        for p in processes:
            p.join()

