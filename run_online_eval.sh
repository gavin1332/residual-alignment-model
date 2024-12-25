#!/bin/bash

if [[ $# < 1 && $# > 3 ]]; then
    echo "Usage: bash $0 model_output max_instances [reference_outputs]"
    exit
fi

set -eux

MODEL_OUTPUT=$1
MAX_INSTANCES=$2

export OPENAI_API_KEY=EMPTY
export https_proxy=10.211.30.6:8888

ARGS="--model_outputs $MODEL_OUTPUT --max_instances $MAX_INSTANCES"
if [[ $# == 3 ]]; then
    ARGS="$ARGS --reference_outputs $2"
fi

alpaca_eval $ARGS
