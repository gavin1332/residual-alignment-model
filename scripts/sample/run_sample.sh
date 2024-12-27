filepath=./qwen
file_prefix=qwen14b-sample-summ-result
gpu_num=7
https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=0 python sample_qwen.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 1 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=1 python sample_qwen.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 2 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=2 python sample_qwen.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 3 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=3 python sample_qwen.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 4 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=4 python sample_qwen.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 5 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=5 python sample_qwen.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 6 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=6 python sample_qwen.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 7

python merge.py --filepath $filepath --file_prefix $file_prefix


filepath=./pythia
file_prefix=pythia12b-sample-summ-result
gpu_num=7
https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=0 python sample_pythia.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 1 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=1 python sample_pythia.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 2 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=2 python sample_pythia.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 3 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=3 python sample_pythia.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 4 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=4 python sample_pythia.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 5 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=5 python sample_pythia.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 6 & https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=6 python sample_pythia.py --filepath $filepath --file_prefix $file_prefix --gpu_num $gpu_num --part 7

python merge.py --filepath $filepath --file_prefix $file_prefix