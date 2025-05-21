import sys
import os
import argparse
import subprocess
import shutil
import threading
from pprint import pprint
from typing import Dict, List, Callable
from sampling import ParallelSampler, SamplingParams


class Pipeline:
    def __init__(self):
        self.datasets_info = {
            'alpaca': dict(
                examples_in_train=51760,
                test_files=dict(helpful='_data/test/alpaca_eval_gpt4_baseline.json'),
                ),
            'hh': dict(
                examples_in_train=160800,
                test_files=dict(
                    helpful='_data/test/hh_helpful_test_chosen_300.json',
                    harmless='_data/test/hh_harmless_test_chosen_300.json'),
                ),
            'harmless': dict(
                examples_in_train=42537,
                test_files=dict(
                    helpful='_data/test/hh_helpful_test_chosen_300.json',
                    harmless='_data/test/hh_harmless_test_chosen_300.json'),
                ),
            'uc': dict(
                examples_in_train=120000,
                test_files=dict(helpful='_data/test/alpaca_eval_gpt4_baseline.json'),
                ),
            'summ-sft': dict(
                examples_in_train=116722,
                test_files=dict(summ='_data/test/openai_summarize_tldr_test_300.json'),
                ),
            } 

        parser = argparse.ArgumentParser()
        arg_group = {}
        group = parser.add_argument_group('common')
        group.add_argument('-g', '--gpu_ids', type=str, default='0,1,2,3,4,5,6,7')
        group.add_argument('-fc', '--force_clean', action='store_true')
        group.add_argument('-ds', '--dataset', type=str)
        group.add_argument('--max_length', type=int, default=2048)
        group.add_argument('--max_prompt_length', type=int, default=1024)
        group.add_argument('--exp_flag', type=str)
        group.add_argument('--eval_flag', type=str)
        group.add_argument('--output_dir', type=str, default='_output')
        group.add_argument('--log_dir', type=str, default='_log')
        arg_group['common'] = group

        group = parser.add_argument_group('model')
        group.add_argument('-m', '--model', type=str, required=True)
        group.add_argument('--archive', type=str)
        arg_group['model'] = group

        group = parser.add_argument_group('train')
        group.add_argument('-t', '--train', action='store_true')
        group.add_argument('-s', '--accum_steps', type=int, default=1)
        group.add_argument('-bs', '--batch_size', type=int, default=64)
        group.add_argument('-lr', '--learning_rate', type=float, default=1e-6)
        group.add_argument('-ep', '--epochs', type=int, default=1)
        arg_group['train'] = group

        group = parser.add_argument_group('sampling')
        group.add_argument('-ik', '--input_key', type=str, default='instruction')
        group.add_argument('-ok', '--output_key', type=str, default='output')
        group.add_argument('--skip_sampling', action='store_true')
        group.add_argument('--temperature', type=float, default=0.8)
        group.add_argument('--top_p', type=float, default=0.9)
        group.add_argument('--top_k', type=int, default=-1)
        group.add_argument('--repetition_penalty', type=float, default=1.05)
        group.add_argument('--multiround', action='store_true')
        group.add_argument('--for_aligner', action='store_true')
        arg_group['sampling'] = group

        group = parser.add_argument_group('base_sampling')
        group.add_argument('--proposal', type=str, help='The upstreaming model responsible for proposing candidate tokens')
        group.add_argument('--pre_sampling', action='store_true', help='pre-sampling from upstreaming model')
        group.add_argument('--base_temperature', type=float, default=0.8)
        group.add_argument('--base_top_p', type=float, default=0.95)
        group.add_argument('--base_top_k', type=int, default=-1)
        group.add_argument('--sample_mode', choices=['par_kl', 'par'], default='par_kl')
        group.add_argument('--div_threshold', type=float, default=0.1)
        arg_group['base_sampling'] = group

        group = parser.add_argument_group('eval')
        group.add_argument('--skip_eval', action='store_true')
        group.add_argument('--eval_ckpts', type=str, help='comma separated ckpt names')
        group.add_argument('--eval_task', default='helpful')
        group.add_argument('--silent', action='store_true', help='No exception raised if experiment does not exist.')
        group.add_argument('--max_instances', type=int)
        arg_group['eval'] = group

        self.add_args(arg_group)
        self.args = parser.parse_args()

        self.train_dataset = None
        self.loss = None
        self.n_examples = None
        self.warmup_steps = 150
        self.eval_every = None
        self.extra_training_args = {}

        self.use_vllm = False
        self.extra_sampling_args = {}

        self.init_properties()
        self.validate_args(parser)
        self.ensure_dataset()

        self.test_file = None
        if self.args.for_aligner:
            self.test_file = f'_data/test/{self.args.dataset}_{self.args.eval_task}_{self.proposal_model()}.json'

        pprint(vars(self.args))

    def add_args(self, arg_group):
        pass

    def init_properties(self):
        pass

    def validate_args(self, parser):
        pass

    def pre_sampling(self):
        assert self.test_file and self.args.for_aligner, 'Only used for aligner with pre assigned test_file'

        if os.path.exists(self.test_file) and not self.args.force_clean:
            print(f'pre-sampling output exists. Skip: {self.test_file}')
            return
            
        print(f'Start sampling from proposal model to {self.test_file} ...')
        sampling_params = SamplingParams(
            model=self.args.proposal,
            generator=self.proposal_model(),
            temperature=0.5,
            top_p=0.95,
            top_k=10,
            repetition_penalty=1.05,
            max_length=2048,
            max_prompt_length=1024,
            multiround=self.args.multiround
        )
        sampler = ParallelSampler(gpu_ids_str=self.args.gpu_ids,
                                  sampling_params=sampling_params,
                                  use_vllm=True)
        sampler.do_sample(self.get_test_file(), self.test_file)

    def exp_name(self):
        prefix = self.exp_name_prefix()
        postfix = self.exp_name_postfix()
        return f'{prefix}_{self.args.dataset}_{self.args.model}_b{self.args.batch_size}_lr{self.args.learning_rate}{postfix}'

    def exp_name_prefix(self):
        return 'aligner' if self.args.for_aligner else self.loss

    def exp_name_postfix(self):
        return f'_@{self.args.exp_flag}' if self.args.exp_flag else ''

    def model_path(self):
        return f'{self.args.output_dir}/model/{self.exp_name()}'

    def proposal_model(self):
        if self.args.model.startswith('qwen'):
            return 'qwen14b'
        elif self.args.model.startswith('llama'):
            return 'llama8b'
        else:
            raise NotImplementedError(f'Unsupported model: {model}')

    def pairwise_dataset(self):
        if self.args.dataset in ['hh']:
            return self.args.dataset
        else:
            return f'{self.args.dataset}_{self.proposal_model()}'

    def run(self):
        if self.args.train:
            self.do_train()

        if self.args.pre_sampling:
            self.pre_sampling()

        if not (self.args.skip_sampling and self.args.skip_eval):
            self.do_evaluate()

    def do_train(self):
        print(f'exp_name: {self.exp_name()}')

        examples_per_epoch = self.calc_examples_per_epoch()
        print(f'training examples per epoch: {examples_per_epoch}')

        train_args = {
            'local_dirs': f'[{self.args.output_dir}/model]',
            'exp_name': self.exp_name(),
            'model': self.args.model,
            'datasets': f'[{self.train_dataset}]',
            'max_length': self.args.max_length,
            'max_prompt_length': self.args.max_prompt_length,
            'loss': self.loss,
            'gradient_accumulation_steps': self.args.accum_steps,
            'batch_size': self.args.batch_size,
            'lr': self.args.learning_rate,
            'warmup_steps': self.warmup_steps,
            'eval_batch_size': self.args.batch_size,
            'trainer': 'FSDPTrainer',
            'model.archive': self.args.archive if self.args.archive else 'null',
            'model.fsdp_policy_mp': 'bfloat16',
            'n_epochs': 'null' if self.n_examples else self.args.epochs,
            'n_examples': self.n_examples if self.n_examples else 'null',
            'eval_every': self.eval_every if self.eval_every else examples_per_epoch
            }

        if self.extra_training_args:
            train_args.update(self.extra_training_args)

        self.clean_exp_if_exists()

        train_command = ['python', '-u', 'train.py', *[f'{k}={v}' for k, v in train_args.items()]]
        os.environ.update({
            'HF_ENDPOINT': 'https://hf-mirror.com',
            'CUDA_VISIBLE_DEVICES': self.args.gpu_ids,
        })

        print('Start training ...')
        process = subprocess.Popen(train_command, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(f'{self.args.log_dir}/{self.exp_name()}.log', 'w') as fout:
            for line in process.stdout:
                print(line, end='')
                print(line, end='', file=fout)

        process.wait()
        if process.returncode != 0:
            print(f'Failed to run command: {" ".join(train_command)}')
            exit(-1)

    def ensure_dataset(self):
        assert self.args.dataset in self.datasets_info, f'Unsupported dataset: {self.args.dataset}'
    
    def calc_examples_per_epoch(self):
        return self.datasets_info[self.args.dataset]['examples_in_train'] // self.args.batch_size * self.args.batch_size

    def calc_examples_half_epoch(self):
        return (self.calc_examples_per_epoch() + self.args.batch_size - 1) // 2

    def get_test_file(self):
        return self.datasets_info[self.args.dataset]['test_files'][self.args.eval_task]

    def clean_exp_if_exists(self, sampling_phase: bool=False, evaluating_phase: bool=False):
        exp_dir = f'{self.args.output_dir}/model/{self.exp_name()}'
        if os.path.exists(exp_dir):
            if self.args.force_clean:
                user_input = 'y'
            else:
                user_input = input(f'The experiment\n{exp_dir}\nexist. Do you want to override them? [y|N]: ')

            if user_input.lower() == 'y':
                if os.path.exists(exp_dir):
                    shutil.rmtree(exp_dir)
            else:
                exit()
    
    def do_evaluate(self):
        if not self.test_file:
            self.test_file = self.get_test_file()

        model_dir = f'{self.args.output_dir}/model/{self.exp_name()}'
        if not os.path.isdir(model_dir):
            msg = f'Experiment is not found: {self.exp_name()}'
            if self.args.silent:
                print(msg)
                sys.exit(0)
            else:
                raise ValueError(f'Experiment is not found: {self.exp_name()}')
    
        ckpts = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        for ckpt in ckpts:
            if self.args.eval_ckpts and ckpt not in self.args.eval_ckpts.split(','):
                continue

            checkpoint_dir = os.path.join(model_dir, ckpt)
            sampling_params = SamplingParams(
                model=checkpoint_dir,
                input_key=self.args.input_key,
                output_key=self.args.output_key,
                max_length=self.args.max_length,
                max_prompt_length=self.args.max_prompt_length,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                repetition_penalty=self.args.repetition_penalty,
                multiround=self.args.multiround,
                for_aligner=self.args.for_aligner,
                base_model=self.args.proposal if not self.args.for_aligner else None,
                base_temperature=self.args.base_temperature,
                base_top_p=self.args.base_top_p,
                base_top_k=self.args.base_top_k,
                sample_mode=self.args.sample_mode,
                div_threshold=self.args.div_threshold,
                **self.extra_sampling_args
            )

            eval_tag = sampling_params.mark()
            postfix = f'_@{self.args.eval_flag}' if self.args.eval_flag else ''
            generator = f'{self.exp_name()}/{ckpt}#{eval_tag}{postfix}'
            sampling_params.generator = generator
    
            work_dir = f'{self.args.output_dir}/{self.args.dataset}/{self.args.eval_task}/{generator}'
            os.makedirs(work_dir, exist_ok=True)
            output_file = f'{work_dir}/output.json'
                
            if not self.args.skip_sampling:
                do_sample = True
                if os.path.exists(output_file):
                    if self.args.force_clean:
                        shutil.rmtree(work_dir)
                        os.makedirs(work_dir)
                    else:
                        print(f'Sampling output exists. Skip: {output_file}')
                        do_sample = False

                if do_sample:
                    print(f'Start sampling {generator} on {self.test_file}...')
                    sampler = ParallelSampler(gpu_ids_str=self.args.gpu_ids,
                                              sampling_params=sampling_params,
                                              use_vllm=self.use_vllm)
                    sampler.do_sample(self.test_file, output_file)
        
            if not self.args.skip_eval:
                eval_output = f'{work_dir}/leaderboard.csv'
                if os.path.exists(eval_output):
                    if not self.args.force_clean:
                        print(f'Evaluation output exists. Skip: {eval_output}')
                        continue

                if os.path.exists(output_file):
                    print(f'Start evaluating {generator} ...')
                    alpaca_eval(model_output=output_file,
                                output_path=work_dir,
                                reference_output=self.get_test_file(),
                                max_instances=self.args.max_instances,
                                eval_task=self.args.eval_task)
                else:
                    print(f'Missing output file: {output_file}')
    

def alpaca_eval(model_output: str, output_path: str=None, reference_output: str=None, max_instances: int=None, client_config: str=None, eval_task: str=None):
    command_args = f'--model_outputs {model_output}'

    conf_prefix = 'evaluate/alpaca_eval/evaluators_configs'
    annotator_postfix = f'_{eval_task}' if eval_task != 'helpful' else ''
    if client_config:
        os.environ['OPENAI_CLIENT_CONFIG_PATH'] = client_config
        annotator = f'weighted_alpaca_eval_gpt4_turbo{annotator_postfix}'
    else:
        os.environ['OPENAI_API_KEY'] = 'EMPTY'
        annotator = f'weighted_alpaca_eval_qwen25_72b_int4{annotator_postfix}'

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        shutil.copytree(os.path.join(conf_prefix, annotator), os.path.join(output_path, annotator),
                        dirs_exist_ok=True)
        annotator_path = os.path.realpath(os.path.join(output_path, annotator))
        command_args = f'{command_args} --output_path {output_path} --annotators_config {annotator_path}'
    else:
        annotations_file = os.path.join(conf_prefix, annotator, 'annotations_seed0_configs.json')
        if os.path.exists(annotations_file):
            os.remove(annotations_file)
        annotator_path = os.path.realpath(os.path.join(conf_prefix, annotator))
        command_args = f'{command_args} --annotators_config {annotator_path}'

    if reference_output:
        command_args = f'{command_args} --reference_outputs {reference_output}'

    if max_instances:
        command_args = f'{command_args} --max_instances {max_instances}'

    #os.environ['https_proxy'] = '10.211.30.6:8888'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    command = f'alpaca_eval {command_args}'
    try:
        print(f'Run command: {command}')
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Fail to run command:\n{command}')
        raise e
    print(f'Finish evaluating file: {model_output}')
    return True

