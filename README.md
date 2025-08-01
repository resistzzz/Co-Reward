<p align="center">
  <img src="figs/CoReward_logo.png" alt="Co-Reward Logo" width="120"/>
</p>

## Co-Reward: Self-Supervised RL for LLM Reasoning via Contrastive Agreement
---
[📄 Paper](./CoReward-paper.pdf)

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
- [x] [Readme] Update the README


## 📄 Citation

Our Paper is pending on the Arxiv processing, waiting to be online.

