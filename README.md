# RAM: Residual Alignment Model

## What is this repo?

This repo includes a reference implementation of the **Residual Alignment Model (RAM)** for training and inference, as described in the paper [Leveraging Importance Sampling to Detach Alignment
Modules from Large Language Models]().

The code here supports any causal HuggingFace model- look at our examples in `config/model` to add your own. Adding your own datasets is also easy. See [the README section](https://github.com/huggingface/peft) on adding datasets.

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

> Note: Due to the need to sample the original training set with the *Proposal Module* to synthesize training data for **RAM**, the original training set must be pre-downloaded to the `_data/train` directory. Please refer to `data_file` variable in `get_uc` and `get_summ_sft` in `preference_datasets.py`

## Evaluation

Run evaluation:

    bash bin/eval_ours.sh

> Note: In addition to the UltraChat task, which uses AlpacaEval 2 for evaluation with all 805 samples, other tasks will randomly sample 300 examples from the test set for evaluation. Please ensure that these test data are pre-stored in the `_data/test` directory. Please refer to `test_files` in `pipeline.py`

# Citing RAM

If RAM or this repository is useful in your own research, you can use the following BibTeX entry:

    @article{
        liu2023leveraging,
        title={Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models},
        author={},
        booktitle={},
        year={2025},
        url={https://arxiv.org/abs/??}
    }
