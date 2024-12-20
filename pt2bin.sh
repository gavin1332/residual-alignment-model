#EXP_NAME=pythia28b_hh_dpo_our_b32_e2
#EXP_NAME=pythia12b_hh_sft_b32
EXP_NAME=pythia12b_hh_dpo_b32_e2
STEP=step-160000

python -u pt2bin.py --base_model_path /private/model/EleutherAI/pythia-12b-deduped \
    --src_model_path .cache/root/${EXP_NAME}/$STEP/policy.pt \
    --dst_model_path .cache/root/hf/${EXP_NAME}/$STEP \
    --dst_dtype bfloat16

