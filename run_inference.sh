#!/usr/bin/env bash

set -ex

if [[ $# != 1 && $# != 3 ]]; then
    echo "Usage: $0 input_file|input_prefix [output_file] [device]"
    exit 1
fi

INPUT_FILE=$1

# mpirun
if [ ! -z $OMPI_COMM_WORLD_RANK ]; then
    INDEX=`printf "%03d" $OMPI_COMM_WORLD_RANK`
    export CUDA_VISIBLE_DEVICES=$(( OMPI_COMM_WORLD_RANK % 8 ))

    INPUT_FILE=$INPUT_FILE$INDEX
    OUTPUT_FILE=$INPUT_FILE.out
else
    OUTPUT_FILE=$2
    export CUDA_VISIBLE_DEVICES=$3
fi

if [ ! -f $INPUT_FILE ]; then
    echo "$INPUT_FILE not exists."
    exit 1
fi

TEMPLATE_NAME=claude
#MODEL=/private/home/liuyi/code/exp/mine/.cache/root/hf/pythia12b_hh_dpo_b32_e2/step-160000
#MODEL=/private/home/liuyi/code/exp/mine/.cache/root/hf/pythia12b_hh_sft_b32
MODEL=/private/home/liuyi/code/exp/mine/.cache/root/hf/pythia28b_hh_dpo_our_b32_e2/step-160000
BASE_MODEL=/private/home/liuyi/code/exp/mine/.cache/root/hf/pythia12b_hh_sft_b32

if [ -z $BASE_MODEL]; then
    python -u inference.py --model $MODEL \
                           --template_name $TEMPLATE_NAME \
                           --data_file $INPUT_FILE \
                           --output_file $OUTPUT_FILE \
                           --temperature 0.8 \
                           --top_p 0.9 \
                           --max_new_tokens 512 \
                           --repetition_penalty 1.02 \

else
    IS_SPAR_MODE=true
    TEMPERATURE=0.8
    TOP_P=0.95
    if [ "IS_SPAR_MODE" -eq "false" ]; then
        TEMPERATURE=1.2
        TOP_P=0.9
    fi
    python -u inference.py --model $MODEL \
                           --template_name $TEMPLATE_NAME \
                           --data_file $INPUT_FILE \
                           --output_file $OUTPUT_FILE \
                           --temperature $TEMPERATURE \
                           --top_p $TOP_P \
                           --max_new_tokens 512 \
                           --repetition_penalty 1.2 \
                           --base_model $BASE_MODEL \
                           --is_spar_mode $IS_SPAR_MODE \
                           --base_top_p 0.95 \
                           --base_temperature 1.0 \
                           --resize_emb \

fi
