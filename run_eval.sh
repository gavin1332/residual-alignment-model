#!/bin/bash

if [ $# != 1 ]; then
    echo "Usage: bash $0 model_output"
    exit
fi

MODEL_OUTPUT=$1

CONF_PREFIX=/private/home/liuyi/code/exp/mine/evaluate/alpaca_eval/evaluators_configs
ANNOTATOR=alpaca_eval_qwen25_72b_fn  #alpaca_eval_vllm_llama3_70b_fn

if [ -f $CONF_PREFIX/$ANNOTATOR/annotations_seed0_configs.json ]; then
    /bin/rm -i $CONF_PREFIX/$ANNOTATOR/annotations_seed0_configs.json
fi

export OPENAI_API_KEY=EMPTY
export https_proxy=10.211.30.6:8888
alpaca_eval --model_outputs $MODEL_OUTPUT --annotators_config $CONF_PREFIX/$ANNOTATOR

