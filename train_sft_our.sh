#model=qwen15
#exp_name=qwen15_sft_our
#CUDA_VISIBLE_DEVICES=2,3,4,5 python -u train.py model=$model datasets=[alpaca_sample] loss=sft exp_name=$exp_name gradient_accumulation_steps=1 batch_size=16 eval_batch_size=5 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
#model=cerebras7b
#exp_name=cerebras7b_sft_norm
#CUDA_VISIBLE_DEVICES=2,3,4,5 python -u train.py model=$model datasets=[alpaca] loss=sft_norm exp_name=$exp_name gradient_accumulation_steps=1 batch_size=16 eval_batch_size=5 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
#model=pythia14
#exp_name=pythia14_sft_our
#CUDA_VISIBLE_DEVICES=2,3,4,5 python -u train.py model=$model datasets=[alpaca] loss=sft_norm exp_name=$exp_name gradient_accumulation_steps=1 batch_size=16 eval_batch_size=5 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16


#model=pythia28
#exp_name=pythia28b_sft_norm
#lr=5e-7
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py model=$model datasets=[alpaca] loss=sft_norm exp_name=$exp_name gradient_accumulation_steps=1 batch_size=16 eval_batch_size=5 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
#
model=pythia28
exp_name=pythia28b-sft-norm-hh
lr=2e-7
https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py model=$model datasets=[hh] loss=sft_norm lr=$lr exp_name=$exp_name gradient_accumulation_steps=1 batch_size=30 eval_batch_size=6 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16

prefix=".cache/root/"
tail="/LATEST/policy.pt"
pt_path="${prefix}${exp_name}${tail}"
python -u pt2bin.py model=$model model.archive=$pt_path datasets=[alpaca] loss=sft exp_name=$exp_name gradient_accumulation_steps=2 batch_size=10 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=float32

#rm -rf "${prefix}${exp_name}"
#model=pythia12b
#exp_name=pythia12b_sft_norm
#CUDA_VISIBLE_DEVICES=2,3,4,5,6 python -u train.py model=$model datasets=[alpaca] loss=sft_norm exp_name=$exp_name gradient_accumulation_steps=1 batch_size=10 eval_batch_size=5 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16




