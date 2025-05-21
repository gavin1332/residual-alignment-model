# RAM: Residual Alignment Model

## What is this repo?

This repo includes a reference implementation of the **Residual Alignment Model (RAM)** for training and inference, as described in the paper **Leveraging Importance Sampling to Detach Alignment
Modules from Large Language Models**.

## Training

Run our SFT for Llama-3.2-1B on Anthropic-HH data:

    bash bin/train_llama_hh.sh

Run our SFT for Llama-3.2-1B on TL;DR Summarization data:

    bash bin/train_llama_summ.sh

Run our SFT for Llama-3.2-1B on UltraChat data:

    bash bin/train_llama_uc.sh

Run our SFT for Qwen2.5-3B on Anthropic-HH data:

    bash bin/train_qwen_hh.sh

Run our SFT for Qwen2.5-3B on TL;DR Summarization data:

    bash bin/train_qwen_summ.sh

Run our SFT for Qwen2.5-3B on UltraChat data:

    bash bin/train_qwen_uc.sh

> Note: Due to the need to sample the original training set with the *Proposal Module* to synthesize training data for *Residual Aligner*, the original training set must be pre-downloaded to the `_data/train` directory. Please refer to `data_file` variable in `get_uc` and `get_summ_sft` functions in file `preference_datasets.py`

> Note: All models used in our experiment is pre-downloaded in to `/private/model` directory, please refer to `config/model/\*.yaml` for detailed information.

## Evaluation

Run evaluation:

    bash bin/eval_ours.sh

> Note: For the UltraChat task, we use AlpacaEval 2 to evaluate all 805 test examples, with the reference outputs from [gpt4_1106_preview](https://github.com/tatsu-lab/alpaca_eval/blob/main/results/gpt4_1106_preview/model_outputs.json). For other domain-specific tasks, we randomly sample 300 examples from their test sets respectively for evaluation. Please ensure that this test data is pre-stored in the `_data/test` directory. Refer to the `test_files` variable in the `pipeline.py` file for more details.
