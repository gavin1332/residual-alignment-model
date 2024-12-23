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
