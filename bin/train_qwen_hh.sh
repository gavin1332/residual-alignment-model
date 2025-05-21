set -eux

BASE=qwen14b
MODEL=qwen3b
EXTRA_ARGS="--skip_sampling --skip_eval"

#python bin/sft_hh.py -m $BASE -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/sft_hh.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/aligner_hh.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS

BASE_ARCHIVE="_output/model/sft_hh_${BASE}_b64_lr1e-06/LATEST"
MODEL_ARCHIVE="_output/model/sft_hh_${MODEL}_b64_lr1e-06/LATEST"
ALIGNER_ARCHIVE="_output/model/wu_aligner_hh_${MODEL}_b64_lr1e-06/LATEST"

#python bin/aligner_hh.py -m $MODEL --archive $ALIGNER_ARCHIVE -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/ours_sft_hh.py -m $MODEL --archive $MODEL_ARCHIVE -t -s 4 -bs 64 -lr 2e-7 $EXTRA_ARGS
#python bin/dpo_hh.py -m $MODEL --archive $MODEL_ARCHIVE -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/ours_sft_hh.py -m $MODEL --archive $MODEL_ARCHIVE -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/dpo_hh.py -m $BASE --archive $BASE_ARCHIVE -t -s 8 -bs 64 -lr 1e-6 $EXTRA_ARGS

## alpha
#for alpha in 1e-5 1e-3 1e-2 1e-1; do
#    python bin/ours_sft_hh.py -m $MODEL --archive $MODEL_ARCHIVE -t -s 4 -bs 64 -lr 2e-7 --alpha $alpha $EXTRA_ARGS
#done
#
## size
#for M in qwen0.5b qwen1.5b; do
#    python bin/sft_hh.py -m $M -t -s 2 -bs 64 -lr 1e-6 $EXTRA_ARGS
#    python bin/ours_sft_hh.py -m $M --archive _output/model/sft_hh_${M}_b64_lr1e-06/LATEST -t -s 4 -bs 64 -lr 2e-7 $EXTRA_ARGS
#done
#python bin/sft_hh.py -m qwen7b -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/ours_sft_hh.py -m qwen7b --archive _output/model/sft_hh_qwen7b_b64_lr1e-06/LATEST -t -s 8 -bs 64 -lr 2e-7 $EXTRA_ARGS


python bin/ours_dpo_hh.py -m $MODEL --archive $MODEL_ARCHIVE -t -s 4 -bs 64 -lr 2e-7 $EXTRA_ARGS
