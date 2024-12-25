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
MODEL=$WORK_DIR/.cache/root/hf/qwen3b_hh_dpo_our_b32_lr1e-6_e1/step-160000
BASE_MODEL=$WORK_DIR/.cache/root/hf/qwen14b_hh_sft_b32/step-160000
LEN_ARGS="--max_prompt_length 512 --max_new_tokens 512"

SAMPLE_ARGS="--model $MODEL \
             --template_name $TEMPLATE_NAME \
             --data_file $INPUT_FILE \
             --output_file $OUTPUT_FILE \
             --input_key $INPUT_KEY \
             --output_key $OUTPUT_KEY \
             $LEN_ARGS"

if [ -z $BASE_MODEL ]; then
    python -u inference.py --temperature 0.8 \
                           --top_p 0.9 \
                           --repetition_penalty 1.02 \
                           $SAMPLE_ARGS
else
    SAMPLE_MODE=spar_js
    if [[ $SAMPLE_MODE == spar* ]]; then
        SAMPLE_ARGS="$SAMPLE_ARGS --base_temperature 0.8 \
                                  --base_top_p 0.95 \
                                  --temperature 0.9 \
                                  --top_p 0.9 \
                                  --repetition_penalty 1.05"
    elif [ "$SAMPLE_MODE" = "mds" ]; then
        SAMPLE_ARGS="$SAMPLE_ARGS --base_temperature 1.1 \
                                  --expert_temperature 1.1 \
                                  --temperature 0.8 \
                                  --top_p 0.9 \
                                  --repetition_penalty 1.2"
    else
        echo "Unkown sample_mode: ${SAMPLE_MODE}"
        exit 1
    fi

    python -u inference.py --base_model $BASE_MODEL \
                           --sample_mode $SAMPLE_MODE \
                           --base_top_k -1 \
                           --resize_emb \
                           $SAMPLE_ARGS
fi
