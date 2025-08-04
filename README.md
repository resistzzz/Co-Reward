<p align="center">
  <img src="figs/CoReward_logo.png" alt="Co-Reward Logo" width="300"/>
</p>

<h1 align="center"><b>Co-Reward: Self-Supervised Reinforcement Learning for Large Language Model Reasoning via Contrastive Agreement</b></h1>

<p align="center">
  <a href="./CoReward-paper.pdf"><img src="https://img.shields.io/badge/Paper-v1-pend.svg" alt="Paper"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="Liscence"></a>
  <img src="https://img.shields.io/github/stars/resistzzz/Co-Reward?color=yellow&label=Star" alt="Stars" >
</p>

Our current version can be found in [📄Paper](https://arxiv.org/abs/2508.00410).

![Pipeline](figs/CoReward_pipeline.png)

**Co-Reward** is a self-supervised reinforcement learning method for LLM reasoning, which leverages contrastive agreement between original and rephrased questions, enabling them to serve as reward signals for each other during training. It effectively mitigates the training collapse of existing self-reward reasoning methods, such as Entroy Minimization, Intuitor, and Majority-Voting.


![Performance](figs/performance.png)


### Install Environment

```bash
# 1. create a clean conda environment
conda create -y -n coreward python=3.10
conda activate coreward

# 2. clone the repository
git clone https://github.com/tmlr-group/Co-Reward.git
cd Co-Reward

# 3. install external dependencies
cd coreward
bash scripts/install_env.sh

# 4. add Coreward to PYTHONPATH in editable mode
pip install -e . --no-deps
```

### Training on MATH Dataset

Modify the WANDB_KEY in the `coreward/math_co_reward.sh` script to your own WANDB key, then run the following command:

```
cd coreward
bash math_co_reward.sh
```

### Preprocess the Training Data

First, download the MATH dataset and prepare it using the following Python script:

```
python examples/data_preprocess/math_dataset.py
```

Second, rephrase the training set using Qwen3-32B as follows:

```
python rewrite_questions.py \
  --input_path data/math/train.parquet \
  --output_jsonl data/math/train_rewrite_Qwen3-32B.jsonl \
  --output_parquet data/math/train_rewrite_Qwen3-32B.parquet \
  --output_original_parquet data/math/train_original.parquet \
  --model_path $YOUR_Qwen3-32B_MODEL_PATH \
  --tokenizer_path $YOUR_Qwen3-32B_TOKENIZER_PATH \
  --question_column prompt \
  --batch_size 128
```

Then, you can train your LLM using Co-Reward following above script.


### Dataset

We release our rephrased MATH training set on [TMLR-Group-HF/CoReward-RephrasedMATH](https://huggingface.co/datasets/TMLR-Group-HF/CoReward-RephrasedMATH).


### Checkpoints

We release all checkpoints trained by us, including our Co-Reward and baselines.

#### Checkpoints of Co-Reward
| Model Name | Model Size | Method | Hugging Face Link |
| --- | --- | --- | --- |
| TMLR-Group-HF/CoReward-Qwen2.5-3B | 3B | Co-Reward | [View Model](https://huggingface.co/TMLR-Group-HF/CoReward-Qwen2.5-3B) |
| TMLR-Group-HF/CoReward-Qwen2.5-7B | 7B | Co-Reward | To be updated |
| TMLR-Group-HF/CoReward-Qwen3-1.7B-Base | 1.7B | Co-Reward | [View Model](https://huggingface.co/TMLR-Group-HF/CoReward-Qwen3-1.7B-Base) |
| TMLR-Group-HF/Co-Reward-Qwen3-4B-Base | 4B | Co-Reward | [View Model](https://huggingface.co/TMLR-Group-HF/CoReward-Qwen3-4B-Base) |
| TMLR-Group-HF/Co-Reward-Qwen3-8B-Base | 8B | Co-Reward | To be updated |
| TMLR-Group-HF/CoReward-Llama-3.2-3B-Instruct | 3B | Co-Reward | [View Model](https://huggingface.co/TMLR-Group-HF/CoReward-Llama-3.2-3B-Instruct) |

#### Checkpoints of Co-Reward
Waiting to be uploaded.


### TODO
This is an initial version of the code. We will make the following updates in the future.
- [ ] [Models] Release all of our trained LLM checkpoints
- [ ] [Code] Update the evaluation code
- [ ] [Paper] Update the Arxiv paper link
- [x] [Environment] Update the runing environment file
- [ ] [Readme] Update the README


## 📄 Citation

If you use our code or data, please cite our paper 📄!
```
@article{zhang2025coreward,
  title={Co-Reward: Self-supervised Reinforcement Learning for Large Language Model Reasoning via Contrastive Agreement}, 
  author={Zizhuo Zhang and Jianing Zhu and Xinmu Ge and Zihua Zhao and Zhanke Zhou and Xuan Li and Xiao Feng and Jiangchao Yao and Bo Han},
  journal={arXiv preprint arXiv:2508.00410}
  year={2025},
}
```
Please give us a **Star**, thanks very much for your focus on our work!!



