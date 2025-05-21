set -eux

BASE=qwen14b
MODEL=qwen3b
EXTRA_ARGS="--skip_sampling --skip_eval"

#python bin/sft_uc.py -m $BASE -t -s 4 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS
#
#python bin/sample_from_proposal.py \
#    --model_path _output/model/wu_sft_uc_${BASE}_b64_lr1e-06/LATEST \
#    --input_file _data/train/ultrachat_round_one.json \
#    --output_file _data/train/ultrachat_round_one_sampling_${BASE}.json
#
#python bin/aligner_uc.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS
#
#python bin/sft_uc.py -m $BASE -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/sft_uc.py -m $MODEL -t -s 2 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/sft_uc.py -m $BASE -t -s 4 -bs 64 -lr 1e-5 $EXTRA_ARGS
#python bin/aligner_uc.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/ours_sft_uc.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 $EXTRA_ARGS
#
#python bin/aligner_uc.py -m $MODEL -t -s 4 -bs 64 -lr 2e-7 $EXTRA_ARGS
#python bin/ours_sft_uc.py -m $MODEL -t -s 4 -bs 64 -lr 2e-7 $EXTRA_ARGS

## alpha
#for alpha in 2e-5 5e-5 2e-4 5e-4 1e-3; do
#    python bin/ours_sft_uc.py -m $MODEL -t -s 4 -bs 64 -lr 2e-7 --alpha $alpha $EXTRA_ARGS
#done
#
## size
#for M in qwen0.5b qwen1.5b qwen7b; do
#    python bin/ours_sft_uc.py -m $M -t -s 4 -bs 64 -lr 2e-7 $EXTRA_ARGS
#done

#python bin/sample_from_proposal.py \
#    --model_path _output/model/sft_uc_${BASE}_b64_lr1e-06/LATEST \
#    --input_file _data/train/ultrachat_round_one.json \
#    --output_file _data/train/ultrachat_round_one_sampling_${BASE}-sft.json

python bin/ours_sft_uc.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 --train_dataset uc_${BASE}-sft --exp_flag sft $EXTRA_ARGS
#python bin/aligner_uc.py -m $MODEL -t -s 4 -bs 64 -lr 1e-6 --train_dataset uc_${BASE}-sft_aligner_common --exp_flag sft $EXTRA_ARGS
