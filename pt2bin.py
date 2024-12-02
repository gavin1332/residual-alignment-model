import torch
import re
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource

OmegaConf.register_new_resolver("get_local_run_dir",
                                lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    # with open(config_path, 'w') as f:
    #     OmegaConf.save(config, f)

    # print('=' * 80)
    # print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    # print('=' * 80)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True,
        torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)


    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(
            f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])

        step=re.search("step-\d{1,}",config.model.archive)
        print(config.model)
        model_name=re.split("/",config.model.name_or_path)[-1]
        policy.save_pretrained('/private/home/liudianqing/tmp_model_path/pythias/'+config.exp_name,safe_serialization=False)

if __name__ == '__main__':
    main()

    """
   python -u train.py model=llama2-7b model.archive=$pt_path datasets=[alpaca] loss=sft exp_name=alpaca_llama_7b_sft gradient_accumulation_steps=2 batch_size=10 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=float32
CUDA_VISIBLE_DEVICES=0,1,5 python -u train.py model=llama2-7b datasets=[hh] loss=sft exp_name=llama2-7b-sft_our gradient_accumulation_steps=1 batch_size=10 eval_batch_size=10 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
CUDA_VISIBLE_DEVICES=0,1,5 python -u train.py model=llama2-7b datasets=[hh] loss=sft exp_name=llama2-7b-sft_our gradient_accumulation_steps=1 batch_size=10 eval_batch_size=10 trainer=TensorParallelTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py model=pythia28 datasets=[alpaca_sample] loss=sft exp_name=pythia28-sft_our_5e-8 gradient_accumulation_steps=1 batch_size=16 eval_batch_size=5 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

.cache/root/pythia160m_our_5e-8_2024-09-30_13-09-34_620784/step-1616/policy.pt
.cache/root/pythia160m_our_5e-8_2024-09-30_13-09-34_620784/step-3232/policy.pt
.cache/root/pythia1b_our_5e-8_2024-09-30_13-07-20_106317/step-1616/policy.pt 
.cache/root/pythia1b_our_5e-8_2024-09-30_13-07-20_106317/step-3232/policy.pt
.cache/root/pythia28-sft_our_5e-8_2024-09-30_12-51-26_155868/step-1616/policy.pt 
.cache/root/pythia28-sft_our_5e-8_2024-09-30_12-51-26_155868/step-3232/policy.pt
.cache/root/pythia410m_our_5e-8_2024-09-30_13-10-58_180367/step-1616/policy.pt 
.cache/root/pythia410m_our_5e-8_2024-09-30_13-10-58_180367/step-3232/policy.pt
.cache/root/pythia69-sft_our_5e-8_2024-09-30_12-56-57_420916/step-1616/policy.pt 
.cache/root/pythia69-sft_our_5e-8_2024-09-30_12-56-57_420916/step-3232/policy.pt
ebm对齐dpo阶段，在summary任务进行调参，经过多此调整，目前指标已达到与DPO论文实验结果中的水平
pt_path=/private/home/liudianqing/projects/direct-preference-optimization-main/.cache/root/alpaca_llama_7b_sft_2024-09-26_06-35-07_525983/step-6250/policy.pt
CUDA_VISIBLE_DEVICES=0,5 python -u train.py model=llama2-7b datasets=[hh] loss=dpo loss.beta=0.1 exp_name=our_dpo_llama7b gradient_accumulation_steps=2 batch_size=12 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=$pt_path
model=pythia160m
pt_path=.cache/root/pythia160m_our_5e-8_2024-09-30_13-09-34_620784/step-1616/policy.pt
python -u pt2bin.py model=$model model.archive=$pt_path datasets=[alpaca] loss=sft exp_name=alpaca_llama_7b_sft gradient_accumulation_steps=2 batch_size=10 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=float32
   """
