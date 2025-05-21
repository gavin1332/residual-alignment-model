set -eux

model=llama1b
python scripts/ours_sft_uc.py -m $model -lr 2e-7 --temperature 0.7 --base_temperature 0.5 --eval_task helpful --proposal _output/model/wu_sft_uc_llama8b_b64_lr1e-06/LATEST
python scripts/ours_sft_summ.py -m $model -lr 2e-7 --temperature 0.3 --base_temperature 0.5 --eval_task summ --proposal _output/model/wu_sft_summ-sft_llama8b_b64_lr1e-06/LATEST
python scripts/ours_sft_hh.py -m $model -lr 2e-7 --temperature 0.5 --base_temperature 0.7 --eval_task helpful --proposal _output/model/sft_hh_llama8b_b64_lr1e-06/LATEST
python scripts/ours_sft_hh.py -m $model -lr 2e-7 --temperature 0.3 --base_temperature 0.7 --eval_task harmless --proposal _output/model/sft_hh_llama8b_b64_lr1e-06/LATEST


model=qwen3b
python scripts/ours_sft_uc.py -m $model -lr 2e-7 --temperature 0.3 --base_temperature 0.7 --eval_task helpful --proposal _output/model/wu_sft_uc_qwen14b_b64_lr1e-06/LATEST
python scripts/ours_sft_summ.py -m $model -lr 2e-7 --temperature 0.3 --base_temperature 0.5 --eval_task summ --proposal _output/model/wu_sft_summ-sft_qwen14b_b64_lr1e-06/LATEST
python scripts/ours_sft_hh.py -m $model -lr 2e-7 --temperature 0.7 --base_temperature 0.5 --eval_task helpful --proposal _output/model/sft_hh_qwen14b_b64_lr1e-06/LATEST
python scripts/ours_sft_hh.py -m $model -lr 2e-7 --temperature 0.3 --base_temperature 0.5 --eval_task harmless --proposal _output/model/sft_hh_qwen14b_b64_lr1e-06/LATEST
