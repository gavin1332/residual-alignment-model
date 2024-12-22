#!/bin/bash

if [[ $# != 1 && $# != 2 ]]; then
    echo "Usage: bash $0 model_output [jsonl2json:0|1]"
    exit
fi

set -eux

MODEL_OUTPUT=$1

CONF_PREFIX=/private/home/liuyi/code/exp/mine/evaluate/alpaca_eval/evaluators_configs
ANNOTATOR=alpaca_eval_qwen25_72b_fn
#ANNOTATOR=alpaca_eval_vllm_llama3_70b_fn

/bin/rm -f $CONF_PREFIX/$ANNOTATOR/annotations_seed0_configs.json

if [[ $# == 2 && "$2" == "1" ]]; then
    TMP_FILE=$MODEL_OUTPUT.tmp.json
    python scripts/jsonl2json.py -i $MODEL_OUTPUT -o $TMP_FILE
    MODEL_OUTPUT=$TMP_FILE
fi

export OPENAI_API_KEY=EMPTY
export https_proxy=10.211.30.6:8888
alpaca_eval --model_outputs $MODEL_OUTPUT --annotators_config $CONF_PREFIX/$ANNOTATOR

