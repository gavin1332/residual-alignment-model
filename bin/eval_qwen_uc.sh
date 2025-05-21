set -eux

MODELs=(qwen14b qwen3b)
LRs=(1e-5 1e-6 2e-7)
TASKs=(helpful)
ARGS="-g 0,1,2,3,4,5,6,7"

#for model in ${MODELs[@]}; do
#    for task in ${TASKs[@]}; do
#        python bin/aligner_uc.py -m $model -lr 1e-6 --eval_task $task --silent --pre_sampling --skip_sampling --skip_eval $ARGS
#    done
#done
#
#for model in ${MODELs[@]}; do
#    for task in ${TASKs[@]}; do
#        for lr in ${LRs[@]}; do
#            for tp in 0.3 0.5 0.7; do
#                echo
#                python bin/sft_uc.py -m $model -lr $lr --eval_task $task --silent --temperature $tp $ARGS
#                python bin/aligner_uc.py -m $model -lr $lr --eval_task $task --silent --temperature $tp $ARGS
#            done
#        done
#    done
#done
#
#for lr in ${LRs[@]}; do
#    for model in ${MODELs[@]}; do
#        for task in ${TASKs[@]}; do
#            for tp in 0.3 0.5 0.7; do
#                for btp in 0.3 0.5 0.7; do
#                    python bin/ours_sft_uc.py -m $model -lr $lr --eval_task $task --silent --temperature $tp --base_temperature $btp $ARGS --skip_eval
#                done
#            done
#        done
#    done
#done

PROPS="--proposal _output/model/sft_uc_qwen14b_b64_lr1e-06/LATEST"
model=qwen3b
#for tp in 0.3 0.5; do
#    for btp in 0.5 0.7; do
#        python bin/ours_sft_uc.py -m $model -lr 2e-7 --silent --temperature $tp --base_temperature $btp $PROPS --eval_flag sft $ARGS
#    done
#done

#python bin/aligner_uc.py -m $model -lr 2e-7 --temperature 0.5 $PROPS --pre_sampling --eval_ckpt LATEST --eval_flag sft -fc

for tp in 0.5 0.7; do
    #python bin/aligner_uc.py -m $model -lr 1e-6 --temperature $tp $PROPS --pre_sampling --exp_flag sft --eval_flag sft $ARGS -fc
    for btp in 0.5 0.7; do
        python bin/ours_sft_uc.py -m $model -lr 1e-6 --temperature $tp --base_temperature $btp --exp_flag sft --eval_flag sft $PROPS $ARGS
    done
done
