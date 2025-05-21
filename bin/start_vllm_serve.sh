#!/bin/bash

set -eux

MODEL_PATH=/private/model/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4
#GPU_IDS=(6 7)
GPU_IDS=(4 5 6 7)
for DEV in ${GPU_IDS[@]}; do
    CUDA_VISIBLE_DEVICES=$DEV vllm serve $MODEL_PATH \
                                         --dtype auto \
                                         --gpu-memory-utilization 0.97 \
                                         --max-num-seqs 3 \
                                         --max-model-len 8192 \
                                         --uvicorn-log-level warning \
                                         --host 0.0.0.0 \
                                         --port 999$DEV &> _log/vllm.$DEV.log &
done

# edit bin/vllm_ngnx.conf and run the following scripts outside
if false; then
    NGINX_CONF=vllm_nginx.conf
    cp bin/$NGINX_CONF /etc/nginx/sites-available/
    ln -sf /etc/nginx/sites-available/$NGINX_CONF /etc/nginx/sites-enabled/
    nginx -t
    service nginx reload
fi

wait
