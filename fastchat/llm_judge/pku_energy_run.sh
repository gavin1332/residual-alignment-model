model_id=Ours-DPO-MDS-pku

prefix="data/PKU/model_answer/"
tail=".jsonl"
pt_path="${prefix}${model_id}${tail}"
rm $pt_path

CUDA_VISIBLE_DEVICES=0 python gen_pku_energy.py \
	  --model-path /private/home/liudianqing/tmp_model_path/pythias/pythia12b_sft_norm_mix \
	  --model-id $model_id \
	  --energy_model_path /private/home/liudianqing/tmp_model_path/pythias/pythia28b-dpo-our-safe-sftpku-2e7-01


