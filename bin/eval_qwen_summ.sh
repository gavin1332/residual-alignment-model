set -eux

MODELs=(qwen14b qwen3b)
LRs=(1e-6 2e-7)
TASKs=(summ)
PROPS="--proposal _output/model/wu_sft_summ-sft_qwen14b_b64_lr1e-06/LATEST"
ARGS="-g 0,1,2,3,4,5,6,7"

#python bin/sft_summ.py -m qwen14b -lr 1e-6 --eval_task summ --silent --temperature 0.3 --warmup $ARGS

#model=qwen3b
#for task in ${TASKs[@]}; do
#    python bin/aligner_summ.py -m $model -lr 1e-6 --eval_task $task --silent --pre_sampling --skip_sampling --skip_eval $PROPS $ARGS
#done
#
#for model in ${MODELs[@]}; do
#    for task in ${TASKs[@]}; do
#        for lr in ${LRs[@]}; do
#            for tp in 0.3 0.5 0.7; do
#                python bin/sft_summ.py -m $model -lr $lr --eval_task $task --silent --temperature $tp $ARGS
#            done
#        done
#    done
#done
#
#model=qwen3b
#for task in ${TASKs[@]}; do
#    for lr in ${LRs[@]}; do
#        for tp in 0.3 0.5 0.7; do
#            python bin/aligner_summ.py -m $model -lr $lr --eval_task $task --silent --temperature $tp $PROPS $ARGS
#            for btp in 0.3 0.5 0.7; do
#                python bin/ours_sft_summ.py -m $model -lr $lr --eval_task $task --silent --temperature $tp --base_temperature $btp --sample_mode par_kl $PROPS $ARGS
#            done
#        done
#    done
#done

PROPS="--proposal _output/model/sft_summ-sft_qwen14b_b64_lr1e-06/LATEST"
#for tp in 0.3; do
#    for btp in 0.5 0.7; do
#        python bin/ours_sft_summ.py -m qwen3b -lr 2e-7 --eval_task summ --silent --temperature $tp --base_temperature $btp $PROPS --eval_flag sft $ARGS
#    done
#done

python bin/aligner_summ.py -m qwen3b -lr 2e-7 --temperature 0.3 $PROPS --pre_sampling --eval_task summ --eval_ckpt LATEST --eval_flag sft -fc
