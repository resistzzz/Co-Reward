<p align="center">
  <img src="figs/CoReward_logo.png" alt="Co-Reward Logo" width="300"/>
</p>

<h1 align="center"><b>Co-Reward: Self-Supervised Reinforcement Learning for Large Language Model Reasoning via Contrastive Agreement</b></h1>

<p align="center">
  <a href="./CoReward-paper.pdf"><img src="https://img.shields.io/badge/Paper-v1-pend.svg" alt="Paper"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-red.svg" alt="Liscence"></a>
  <img src="https://img.shields.io/github/stars/resistzzz/Co-Reward?color=yellow&label=Star" alt="Stars" >
</p>

Our current version can be found in [📄 Paper](./CoReward-paper.pdf).

![Pipeline](figs/CoReward_pipeline.png)

**Co-Reward** is a self-supervised reinforcement learning method for LLM reasoning, which leverages contrastive agreement between original and rephrased questions, enabling them to serve as reward signals for each other during training. It effectively mitigates the training collapse of existing self-reward reasoning methods, such as Entroy Minimization, Intuitor, and Majority-Voting.


![Performance](figs/performance.png)


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

Then, you can train your LLM using Co-Reward following above script.

### TODO
This is an initial version of the code. We will make the following updates in the future.
- [ ] [Models] Release all of our trained LLM checkpoints
- [ ] [Code] Update the evaluation code
- [ ] [Paper] Update the Arxiv paper link
- [ ] [Environment] Update the runing environment file
- [ ] [Readme] Update the README


## 📄 Citation

Our Paper is pending on the Arxiv processing, waiting to be online.



