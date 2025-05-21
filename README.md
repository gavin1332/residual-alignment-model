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

## Evaluation

Run evaluation:

    bash bin/eval_ours.sh

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
