MODEL=pythia28b
DATASET=hh
LOSS=dpo_our
BATCH_SIZE=32
LR=2e-7
N_EPOCHS=2
EXP_NAME=${MODEL}_${DATASET}_${LOSS}_b${BATCH_SIZE}_e${N_EPOCHS}
MODEL_SFT=.cache/root/hf/pythia28b_hh_sft_b32_e4/step-160000

https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py \
    exp_name=$EXP_NAME \
    model=$MODEL \
    datasets=[$DATASET] \
    loss=$LOSS \
    model.archive=$MODEL_SFT \
    gradient_accumulation_steps=1 \
    batch_size=$BATCH_SIZE \
    lr=$LR \
    eval_batch_size=32 \
    trainer=FSDPTrainer \
    sample_during_eval=false \
    model.fsdp_policy_mp=bfloat16 \
    n_epochs=${N_EPOCHS} \
    eval_every=80000

