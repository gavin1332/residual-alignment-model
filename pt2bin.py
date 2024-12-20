import torch
import transformers
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Process model paths.')
    parser.add_argument('--base_model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--src_model_path', type=str, required=True, help='Path to the source model')
    parser.add_argument('--dst_model_path', type=str, required=True, help='Path to the destination model')
    parser.add_argument('--dst_dtype', choices=['float16', 'bfloat16', 'auto'], default='auto')
    args = parser.parse_args()

    model_kwargs = {'device_map': 'cpu'}
    policy = transformers.AutoModelForCausalLM.from_pretrained(args.base_model_path, low_cpu_mem_usage=True, **model_kwargs)
    state_dict = torch.load(args.src_model_path, map_location='cpu')
    policy.load_state_dict(state_dict['state'])

    if args.dst_dtype == 'bfloat16':
        policy = policy.bfloat16()
    elif args.dst_dtype == 'float16':
        policy = policy.half()
    else:
        pass

    policy.save_pretrained(args.dst_model_path, safe_serialization=False)
    os.system(f'cp {args.base_model_path}/*token* {args.dst_model_path}/')

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
