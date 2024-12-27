#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i) MODEL_OUTPUT="$2"; shift 2 ;;
        -r) REFERENCE_OUTPUT="$2"; shift 2 ;;
        -m) MAX_INSTANCES="$2"; shift 2 ;;
        -o) OUTPUT_PATH="$2"; shift 2 ;;
        -c) CLIENT_CONFIG="$2"; shift 2 ;;
        *) echo "Usage: $0 -i model_output [-r reference_output] [-m max_indstances]" \
                " [-o output_path] [-c client_config]"; exit 1 ;;
    esac
done

if [[ -z $MODEL_OUTPUT ]]; then
    echo "Usage: $0 -i model_output [-r reference_output] [-m max_indstances]" \
         " [-o output_path] [-c client_config]"
    exit 1
fi

set -ex

CONF_PREFIX=/private/home/liuyi/code/exp/mine/evaluate/alpaca_eval/evaluators_configs
if [[ -z $CLIENT_CONFIG ]]; then
    export OPENAI_API_KEY=EMPTY
    ANNOTATOR=alpaca_eval_qwen25_72b_fn
    #ANNOTATOR=alpaca_eval_vllm_llama3_70b_fn
else
    export OPENAI_CLIENT_CONFIG_PATH=$CLIENT_CONFIG
    ANNOTATOR=alpaca_eval_gpt4_turbo_fn
fi

/bin/rm -f $CONF_PREFIX/$ANNOTATOR/annotations_seed0_configs.json

ARGS="--model_outputs $MODEL_OUTPUT --annotators_config $CONF_PREFIX/$ANNOTATOR"
if [[ ! -z $REFERENCE_OUTPUT ]]; then
    ARGS="$ARGS --reference_outputs $REFERENCE_OUTPUT"
fi

if [[ ! -z $MAX_INSTANCES ]]; then
    ARGS="$ARGS --max_instances $MAX_INSTANCES"
fi

if [[ ! -z $OUTPUT_PATH ]]; then
    ARGS="$ARGS --output_path $OUTPUT_PATH"
fi

https_proxy=10.211.30.6:8888 alpaca_eval $ARGS
