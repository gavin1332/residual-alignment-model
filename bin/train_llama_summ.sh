set -eux

BASE=llama8b
MODEL=llama3b
EXTRA_ARGS="--skip_sampling --skip_eval -g 0,1,2,3"

#python bin/sft_summ.py -m $BASE -t -s 2 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS
#
#python bin/sample_from_proposal.py \
#    --model_path _output/model/wu_sft_summ-sft_${BASE}_b64_lr1e-06/LATEST \
#    --input_file _data/train/openai_summarize_tldr.jsonl \
#    --output_file _data/train/openai_summarize_tldr_sampling_${BASE}.jsonl
#
#python bin/aligner_summ.py -m $MODEL -t -s 2 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS
#ALIGNER_ARCHIVE=_output/model/wu_aligner_summ-sft_${MODEL}_b64_lr1e-06/LATEST
#
#python bin/sft_summ.py -m $BASE -t -s 2 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/sft_summ.py -m $MODEL -t -s 1 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/aligner_summ.py -m $MODEL --archive $ALIGNER_ARCHIVE -t -s 2 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/ours_sft_summ.py -m $MODEL -t -s 2 -bs 64 -lr 1e-6 $EXTRA_ARGS
#
#python bin/aligner_summ.py -m $MODEL --archive $ALIGNER_ARCHIVE -t -s 2 -bs 64 -lr 2e-7 $EXTRA_ARGS
#python bin/ours_sft_summ.py -m $MODEL -t -s 2 -bs 64 -lr 2e-7 $EXTRA_ARGS

#python bin/ours_sft_summ.py -m llama1b -t -s 2 -bs 64 -lr 2e-7 $EXTRA_ARGS
#python bin/aligner_summ.py -m llama1b -t -s 2 -bs 64 -lr 1e-6 --warmup $EXTRA_ARGS
#ALIGNER_ARCHIVE=_output/model/wu_aligner_summ-sft_llama1b_b64_lr1e-06/LATEST
#python bin/aligner_summ.py -m llama1b --archive $ALIGNER_ARCHIVE -t -s 2 -bs 64 -lr 1e-6 $EXTRA_ARGS
#python bin/sft_summ.py -m llama1b -t -s 1 -bs 64 -lr 1e-6 $EXTRA_ARGS
#
#python bin/sample_from_proposal.py \
#    --model_path _output/model/sft_summ-sft_${BASE}_b64_lr1e-06/LATEST \
#    --input_file _data/train/openai_summarize_tldr.jsonl \
#    --output_file _data/train/openai_summarize_tldr_sampling_${BASE}-sft.jsonl
#
#python bin/ours_sft_summ.py -m llama1b -t -s 4 -bs 64 -lr 2e-7 --train_dataset summ-sft_${BASE}-sft --exp_flag sft $EXTRA_ARGS
python bin/aligner_summ.py -m llama1b -t -s 4 -bs 64 -lr 1e-6 --train_dataset summ-sft_${BASE}-sft_aligner_common --exp_flag sft $EXTRA_ARGS
