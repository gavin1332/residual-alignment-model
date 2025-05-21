set -eux

MODELs=(llama3b llama8b)
LRs=(1e-6 2e-7)
TASKs=(helpful harmless)
ARGS="-g 1,2,3,4,5,6,7 --skip_sampling --eval_ckpts step-80384"
PROPS="--proposal _output/model/sft_hh_llama8b_b64_lr1e-06/LATEST"

#model=llama3b
#for task in ${TASKs[@]}; do
#    python bin/aligner_hh.py -m $model -lr 1e-6 --eval_task $task --silent --pre_sampling --skip_sampling --skip_eval $PROPS $ARGS
#done
#
#for model in ${MODELs[@]}; do
#    for task in ${TASKs[@]}; do
#        for lr in ${LRs[@]}; do
#            for tp in 0.3 0.5 0.7; do
#                python bin/sft_hh.py -m $model -lr $lr --eval_task $task --silent --temperature $tp $ARGS
#                python bin/dpo_hh.py -m $model -lr $lr --eval_task $task --silent --temperature $tp $ARGS
#            done
#        done
#    done
#done

#model=llama3b
#for task in ${TASKs[@]}; do
#    for lr in ${LRs[@]}; do
#        for tp in 0.3 0.5 0.7; do
#            for btp in 0.5 0.7; do
#                python bin/ours_sft_hh.py -m $model -lr $lr --eval_task $task --silent --temperature $tp --base_temperature $btp --sample_mode par_kl $PROPS $ARGS
#            done
#        done
#    done
#done

#for tp in 0.3; do
#    for btp in 0.3 0.5 0.7; do
#        python bin/ours_sft_hh.py -m llama3b -lr 2e-7 --eval_task harmless --silent --temperature $tp --base_temperature $btp --proposal _output/model/dpo_hh_llama8b_b64_lr1e-06/step-80384 --eval_flag dpo $ARGS
#    done
#done
#
#for tp in 0.5 0.7; do
#    for btp in 0.5 0.7; do
#        python bin/ours_sft_hh.py -m llama3b -lr 2e-7 --eval_task helpful --silent --temperature $tp --base_temperature $btp --proposal _output/model/dpo_hh_llama8b_b64_lr1e-06/step-80384 --eval_flag dpo $ARGS
#    done
#done

#PROPS="--proposal _output/model/sft_hh_llama8b_b64_lr1e-06/LATEST"
#ARGS="-g 0,1,2,3,4,5,6,7 --eval_flag alpha_v2"
#for alpha in 1e-5 1e-4 1e-3 1e-2 1e-1; do
#    python bin/ours_sft_hh.py -m llama3b -lr 2e-7 --alpha $alpha --eval_task helpful --silent --temperature 0.5 --base_temperature 0.7 $PROPS $ARGS
#    python bin/ours_sft_hh.py -m llama3b -lr 2e-7 --alpha $alpha --eval_task harmless --silent --temperature 0.3 --base_temperature 0.7 $PROPS $ARGS
#done
#
#ARGS="-g 0,1,2,3,4,5,6,7 --eval_flag size_v2"
#for M in llama1b llama3b llama8b; do
#    python bin/ours_sft_hh.py -m $M -lr 2e-7 --eval_task helpful --silent --temperature 0.5 --base_temperature 0.7 $PROPS $ARGS
#    python bin/ours_sft_hh.py -m $M -lr 2e-7 --eval_task harmless --silent --temperature 0.3 --base_temperature 0.7 $PROPS $ARGS
#done

#python bin/aligner_hh.py -m llama3b -lr 1e-6 --eval_task helpful --temperature 0.5 --eval_flag dpo --eval_ckpt LATEST $PROPS --pre_sampling -fc
#python bin/aligner_hh.py -m llama3b -lr 1e-6 --eval_task harmless --temperature 0.3 --eval_flag dpo --eval_ckpt LATEST $PROPS --pre_sampling -fc

#ARGS="-g 1,2,3,4,5,6,7 --skip_eval --eval_ckpts step-80384"
ARGS="-g 1,2,3,4,5,6,7"
PROPS="--proposal _output/model/sft_hh_llama8b_b64_lr1e-06/LATEST"
model=llama1b
#for task in ${TASKs[@]}; do
#    python bin/aligner_hh.py -m $model -lr 1e-6 --eval_task $task --silent --pre_sampling $PROPS $ARGS
#done
#
#python bin/dpo_hh.py -m $model -lr 1e-6 --eval_task helpful --silent --temperature 0.5 $ARGS
#python bin/dpo_hh.py -m $model -lr 1e-6 --eval_task harmless --silent --temperature 0.3 $ARGS
#python bin/ours_sft_hh.py -m $model -lr 2e-7 --eval_task helpful --silent --temperature 0.7 --base_temperature 0.5 --sample_mode par_kl $PROPS $ARGS
#python bin/ours_sft_hh.py -m $model -lr 2e-7 --eval_task harmless --silent --temperature 0.3 --base_temperature 0.5 --sample_mode par_kl $PROPS $ARGS
python bin/aligner_hh.py -m $model -lr 1e-6 --eval_task helpful --temperature 0.5 --pre_sampling $PROPS $ARGS -fc
python bin/aligner_hh.py -m $model -lr 1e-6 --eval_task harmless --temperature 0.3 --pre_sampling $PROPS $ARGS -fc

PROPS="--proposal _output/model/dpo_hh_llama8b_b64_lr1e-06/step-80384"
#python bin/ours_sft_hh.py -m $model -lr 2e-7 --eval_task helpful --silent --temperature 0.7 --base_temperature 0.5 --eval_flag dpo $PROPS $ARGS
#python bin/ours_sft_hh.py -m $model -lr 2e-7 --eval_task harmless --silent --temperature 0.3 --base_temperature 0.5 --eval_flag dpo $PROPS $ARGS
python bin/aligner_hh.py -m $model -lr 1e-6 --eval_task helpful --temperature 0.5 --pre_sampling --eval_flag dpo $PROPS $ARGS -fc
python bin/aligner_hh.py -m $model -lr 1e-6 --eval_task harmless --temperature 0.3 --pre_sampling --eval_flag dpo $PROPS $ARGS -fc
