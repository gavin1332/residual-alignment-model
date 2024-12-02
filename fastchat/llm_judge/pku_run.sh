model_id=pythia12b-dpo-norm-safe-sftpku-2e7-05

prefix="data/PKU/model_answer/"
tail=".jsonl"
pt_path="${prefix}${model_id}${tail}"
rm $pt_path

CUDA_VISIBLE_DEVICES=0 python gen_pku.py \
	  --model-path /private/home/liudianqing/tmp_model_path/pythias/pythia12b-dpo-norm-safe-sftpku-2e7-05 \
	  --model-id $model_id

