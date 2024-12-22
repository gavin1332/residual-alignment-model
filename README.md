# 基于hh数据和alpaca_eval进行训练和评测脚本（2024年12月22日更新）

```bash
# SFT training
bash sft_train.sh

# DPO training
bash dpo_train.sh

# convert model from pytorch to transformers
bash pt2bin.sh

# inference. The "some_id" is optional for distinguishing temporary files of concurrent runs.
## single GPU
bash run_inference.sh evaluate/alpaca_eval/alpaca_eval_ques.json output.json 0 [some_id]
## multiple GPU
bash localrun_inference.sh evaluate/alpaca_eval/alpaca_eval_ques.json output.json 0,1,2,3 [some_id]

# run alpaca_eval
## if json format is required
bash run_eval.sh output.json 1
## else use json-lines format
bash run_eval.sh output.json
```

# DPO: Direct Preference Optimization

##训练环境
source /private/home/liudianqing/dpo_env/bin/activate

## 配置模型路径
在config/model目录下配置不同种类pretrain model的路径等

## Running SFT

**SFT-norm:**

    bash train_sft_norm.sh
model为要训练的pretrain model对应的配置文件的文件名；exp_name为训练后的模型id；

loss=sft_norm,

datasets=[xx]，xx为要训练的数据集，见preference_datasets.py

训练完成后保存的只有policy.pt文件，通过pt2bin.py将policy.pt转换为pytorch_model.bin文件，词表文件需人工添加至转换后的目录

**SFT-our:**

    bash train_sft_our.sh
> Note: loss设置为sft


## Running DPO
**DPO-norm:**

    bash dpo-norm-train.sh

**DPO-norm:**

    bash dpo_our_train.sh

## 评估
**配置环境:**

    git clone https://github.com/lm-sys/FastChat.git
    cd FastChat
    pip install -e ".[model_worker,llm_judge]"

    source /private/home/liudianqing/lm_judge_env/bin/activate

删除FastChat/fastChat下的llm_judge目录，然后将本项目fastchat的目录下的文件按层级复制到FastChat/fastChat下

**生成测试结果:**

单模型结果:

    model_id=pythia12b-dpo-norm-safe-sftpku-2e7-05 # 测试结果以prefix+model_id命名
    # prefix="data/PKU/model_answer/",可在gen_pku.py中修改
    CUDA_VISIBLE_DEVICES=0 python gen_pku.py \
    	  --model-path pythia12b-dpo-norm-safe-sftpku-2e7-05
    	  --model-id $model_id
>采样参数在python gen_pku.py中设置

大小两个模型采样（model-path：大模型，energy_model_path：小模型）:

    CUDA_VISIBLE_DEVICES=0 python gen_pku_energy.py \
	  --model-path /private/home/liudianqing/tmp_model_path/pythias/pythia12b_sft_norm_mix \
	  --model-id $model_id \
	  --energy_model_path /private/home/liudianqing/tmp_model_path/pythias/pythia28b-dpo-our-safe-sftpku-2e7-01

>其中gen_pku_energy.py为Ours-DPO-sPAR采样，gen_pku_ip.py为Ours-DPO-MDS采样;采样参数须在models/llama/pythia_modeling_emb_tmp.py,models/llama/pythia_modeling_emb_tmp.py中的sample方法中更改

> 当transformers版本大于4.40时,上述两文件的 sample 方法需改名为 _sample

**Llama-Guard 3评估**

    CUDA_VISIBLE_DEVICES=0 python LlamaGuard.py --file data/PKU/model_answer/pythia28b-sft-norm-mix.jsonl

**BeaverReward评估**

    CUDA_VISIBLE_DEVICES=0 python LlamaBeaverReward.py --file data/PKU/model_answer/pythia28b-sft-norm-mix.jsonl

**AlpacaEval评估(依赖 lm_judge_env 环境)**

激活环境：source /private/home/liudianqing/lm_judge_env/bin/activate

1. 生成离线结果

单模型，以pythia为例，其他模型需更换代码内的CausalLM类型：


    model_id=pythia12-sft_norm-alpacaeval
    prefix="data/PKU/model_answer/"
    tail=".jsonl"
    pt_path="${prefix}${model_id}${tail}"
    rm $pt_path
    https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=0 python gen_alpacaeval.py \
          --model-path /private/home/liudianqing/tmp_model_path/pythias/pythia12-sft_norm-hh \
          --model-id $model_id \
          --bench-name PKU
    
生成结果保存在./data/PKU/model_answer/{model_id}.json


MDS：


    https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=2 python gen_alpacaeval_ip.py \
	  --model-path /private/home/liudianqing/tmp_model_path/pythias/pythia12-sft_norm-hh \
	  --model-id $model_id \
	  --energy_model_path /private/home/liudianqing/tmp_model_path/pythias/pythia28-dpo-our-hhsft-2e7 \
	  --bench-name PKU


sPAR：


    https_proxy=10.211.30.6:8888 CUDA_VISIBLE_DEVICES=2 python gen_alpacaeval_energy.py \ 
        ......


2. 生成评估结果（lm_judge_env环境内）


    https_proxy=10.211.30.6:8888 OPENAI_API_KEY=EMPTY alpaca_eval --model_outputs 'xxxxx.json'   --annotators_config 'alpaca_eval_qwen2.5_70b_fn'


3. 修改评测用模型接口

须修改alpaca_eval源代码，新增接口可参考/private/home/liudianqing/lm_judge_env/lib/python3.10/site-packages/alpaca_eval/evaluators_configs/alpaca_eval_qwen2.5_70b_fn目录下的配置


**大模型评估胜率**

    model_id=xxx
    # 删除历史评测数据
    prefix="data/PKU/model_answer/"
    tail=".jsonl"
    pt_path="${prefix}${model_id}${tail}"
    rm $pt_path
    python clean_judge_file.py --file=data/PKU/model_judgment/{大模型名}_hh_single.jsonl --model_id=$model_id
    # 开始本次评测
    python judgment_hh.py --judge-model Qwen2.5-72B-Instruct-GPTQ-Int4 --bench-name PKU --model_id $model_id 
    # 查看结果
    python show_result_hh.py --input-file=data/PKU/model_judgment/{评测用大模型}_hh_single.jsonl

>--bench-name对应测试结果所在目录，与前面的prefix对应,

> 大模型接口可在common_hh.py中400行左右更改设置

> 根据不同评估内容可在common_hh.py的run_judge_single修改评估的prompt
