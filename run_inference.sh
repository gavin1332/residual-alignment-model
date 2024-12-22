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
INPUT_KEY=instruction
OUTPUT_KEY=output

WORK_DIR=/private/home/liuyi/code/exp/mine
MODEL=$WORK_DIR/.cache/root/hf/pythia28b_hh_dpo_our_b32_e2/LATEST
BASE_MODEL=$WORK_DIR/.cache/root/hf/pythia12b_hh_sft_b32.1221/step-160000

if [ -z $BASE_MODEL]; then
    python -u inference.py --model $MODEL \
                           --template_name $TEMPLATE_NAME \
                           --data_file $INPUT_FILE \
                           --json_format \
                           --output_file $OUTPUT_FILE \
                           --input_key $INPUT_KEY \
                           --output_key $OUTPUT_KEY \
                           --temperature 0.8 \
                           --top_p 0.9 \
                           --max_new_tokens 512 \
                           --repetition_penalty 1.2

else
    SAMPLE_MODE=spar
    BASE_TEMPERATURE=1.0
    BASE_TOP_P=0.9
    TEMPERATURE=0.9
    TOP_P=0.9
    REPETITION_PENALTY=1.2
    if [ "$SAMPLE_MODE" = "mds" ]; then
        BASE_TEMPERATURE=0.8
        TEMPERATURE=1.1
        TOP_P=0.9
        REPETITION_PENALTY=1.2
    fi
    python -u inference.py --model $MODEL \
                           --template_name $TEMPLATE_NAME \
                           --data_file $INPUT_FILE \
                           --json_format \
                           --output_file $OUTPUT_FILE \
                           --input_key $INPUT_KEY \
                           --output_key $OUTPUT_KEY \
                           --temperature $TEMPERATURE \
                           --top_p $TOP_P \
                           --max_new_tokens 512 \
                           --repetition_penalty $REPETITION_PENALTY \
                           --base_model $BASE_MODEL \
                           --sample_mode $SAMPLE_MODE \
                           --base_top_k -1 \
                           --base_top_p $BASE_TOP_P \
                           --base_temperature $BASE_TEMPERATURE \
                           --resize_emb

fi
