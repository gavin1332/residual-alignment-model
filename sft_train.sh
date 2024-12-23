EXP_NAME=pythia28b_hh_sft_b32
MODEL=pythia28b

https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py \
    exp_name=$EXP_NAME \
    model=$MODEL \
    datasets=[hh] \
    loss=sft \
    gradient_accumulation_steps=2 \
    batch_size=32 \
    lr=2e-7 \
    eval_batch_size=32 \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    n_epochs=1 \
    eval_every=80000

