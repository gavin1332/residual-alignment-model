set -eux

BASE=qwen14b
MODEL=qwen3b
EXTRA_ARGS="--skip_sampling --skip_eval"

python scripts/sft_summ.py -m $BASE -t -s 2 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS

python scripts/sample_from_proposal.py \
    --model_path _output/model/wu_sft_summ-sft_${BASE}_b64_lr1e-06/LATEST \
    --input_file _data/train/openai_summarize_tldr.jsonl \
    --output_file _data/train/openai_summarize_tldr_sampling_${BASE}.jsonl

python scripts/ours_sft_summ.py -m $MODEL -t -s 2 -bs 64 -lr 2e-7 $EXTRA_ARGS

