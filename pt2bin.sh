#model=pythia12b
#exp_name=pythia12b_sft_summ_5e5
#pt_path=.cache/root/pythia12b_sft_summ_5e5/step-159990/policy.pt
#python -u pt2bin.py model=$model model.archive=$pt_path datasets=[alpaca] loss=sft exp_name=$exp_name gradient_accumulation_steps=2 batch_size=10 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=float32

# step-232000/  step-348000/  step-464000/  step-580000/
model=pythia28
exp_name=pythia28b-sft-norm-summ-6e6
pt_path=.cache/root/pythia28b-sft-norm-summ-6e6/step-464000/policy.pt
python -u pt2bin.py model=$model model.archive=$pt_path datasets=[alpaca] loss=sft exp_name=$exp_name gradient_accumulation_steps=2 batch_size=10 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=float32

