set -eux

BASE=qwen14b
MODEL=qwen3b
EXTRA_ARGS="--skip_sampling --skip_eval"

python scripts/sft_uc.py -m $BASE -t -s 4 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS

python scripts/sample_from_proposal.py \
    --model_path _output/model/wu_sft_uc_${BASE}_b64_lr1e-06/LATEST \
    --input_file _data/train/ultrachat_round_one.json \
    --output_file _data/train/ultrachat_round_one_sampling_${BASE}.json

python scripts/ours_sft_uc.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
