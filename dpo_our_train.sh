model=pythia28
exp_name=pythia28b-dpo-our-safe-sftpku-2e7-01
lr=2e-7
beta=0.5
loss=dpo_our
n_epochs=1
model_sft=/private/home/liudianqing/tmp_model_path/pythias/pythia28b-sft-norm-mix
https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u train.py n_epochs=$n_epochs model=$model datasets=[pku] loss=$loss loss.beta=$beta lr=$lr model.archive=$model_sft exp_name=$exp_name gradient_accumulation_steps=1 batch_size=18 eval_batch_size=6 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
prefix=".cache/root/"
tail="/LATEST/policy.pt"
pt_path="${prefix}${exp_name}${tail}"
python -u pt2bin.py model=$model model.archive=$pt_path datasets=[alpaca] loss=sft exp_name=$exp_name gradient_accumulation_steps=1 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=float32

#rm -rf "${prefix}${exp_name}"






