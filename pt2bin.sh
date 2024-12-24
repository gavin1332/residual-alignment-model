#!/bin/bash

set -eux

BASE_MODEL=EleutherAI/pythia-2.8b-deduped
EXP_NAME=pythia28b_hh_dpo_our_b32_e2

STEP_LIST=("step-80000" "step-160000")
for step in "${STEP_LIST[@]}"; do
    python -u pt2bin.py --base_model_path /private/model/$BASE_MODEL \
            --src_model_path .cache/root/${EXP_NAME}/$step/policy.pt \
            --dst_model_path .cache/root/hf/${EXP_NAME}/$step \
            --dst_dtype bfloat16
done
