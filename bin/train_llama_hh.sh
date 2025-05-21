set -eux

MODEL=llama1b
EXTRA_ARGS="--skip_sampling --skip_eval"

python scripts/sft_hh.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS

MODEL_ARCHIVE="_output/model/sft_hh_${MODEL}_b64_lr1e-06/LATEST"
python scripts/ours_sft_hh.py -m $MODEL --archive $MODEL_ARCHIVE -t -s 4 -bs 64 -lr 2e-7 $EXTRA_ARGS
