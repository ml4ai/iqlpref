---
hide:
  - toc        # Hide table of contents
---

# DT

## Overview

The Decision Transformer (DT) model casts offline reinforcement learning as a conditional sequence modeling problem. 

Unlike prior approaches to offline RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal 
actions by leveraging a causally masked Transformer. By conditioning an autoregressive model on the desired return
(reward-to-go), past states, and actions, Decision Transformer model can generate future actions that achieve the desired return. 

Original paper:

 * [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
 * [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039)
   (similar approach, came out at the same time)

Reference resources:

* :material-github: [Official codebase for Decision Transformer](https://github.com/kzl/decision-transformer)

!!! info
        Due to the simple supervised objective and transformer architecture, Decision Transformer is simple, stable and easy to implement as it
        has a minimum number of moving parts.

!!! warning
        Despite its simplicity and stability, DT has a number of drawbacks. It does not capable of stitching suboptimal 
        trajectories (that's why poor performance on AntMaze datasets), and can also [show](https://arxiv.org/abs/2205.15967) bad performance in stochastic environments. 
    
Possible extensions:

* [Online Decision Transformer](https://arxiv.org/abs/2202.05607)
* [Emergent Agentic Transformer from Chain of Hindsight Experience](https://arxiv.org/abs/2305.16554)
* [Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL](https://proceedings.mlr.press/v202/yamagata23a.html)

We'd be glad if someone would be interested in contributing them!

## Implemented Variants

| Variants Implemented                                                                                           | Description                                                              |
|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| :material-github: [`offline/dt.py`](https://github.com/corl-team/CORL/blob/main/algorithms/offline/dt.py#L498) | For continuous action spaces and offline RL without fine-tuning support. |


## Explanation of logged metrics

* `eval/{target_return}_return_mean`: mean undiscounted evaluation return when prompted with `config.target_return` value (there might be more than one)
* `eval/{target_return}_return_std`: standard deviation of the undiscounted evaluation return across `config.eval_episodes` episodes
* `eval/{target_return}_normalized_score_mean`: mean normalized score when prompted with `config.target_return` value (there might be more than one). 
  Should be between 0 and 100, where 100+ is the performance above expert for this environment. 
  Implemented by D4RL library [[:material-github: source](https://github.com/Farama-Foundation/D4RL/blob/71a9549f2091accff93eeff68f1f3ab2c0e0a288/d4rl/offline_env.py#L71)].
* `eval/{target_return}_normalized_score_std`: standard deviation of the normalized score return across `config.eval_episodes` episodes
* `train_loss`: current training loss, Mean squared error (MSE) for continuous action spaces
* `learning_rate`: current learning rate, helps monitor learning rate schedule

## Implementation details

1. Batch sampling weighted by trajectory length (:material-github: [algorithms/offline/dt.py#L171](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L171))
2. State normalization during training and inference (:material-github: [algorithms/offline/dt.py#L181](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L181))
3. Reward downscaling (:material-github: [algorithms/offline/dt.py#L182](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L182))
4. Positional embedding shared across one transition (:material-github: [algorithms/offline/dt.py#L323](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L323))
5. Prompting with multiple return-to-go's during evaluation, as DT can be sensitive to the prompt (:material-github: [algorithms/offline/dt.py#L498](https://github.com/corl-team/CORL/blob/e9768f90a95c809a5587dd888e203d0b76b07a39/algorithms/offline/dt.py#L498))

## Experimental results

For detailed scores on all benchmarked datasets see [benchmarks section](../benchmarks/offline.md). 
Reports visually compare our reproduction results with original paper scores to make sure our implementation is working properly.

<iframe src="https://wandb.ai/tlab/CORL/reports/-Offline-Decision-Transformer--VmlldzoyNzA2MTk3" style="width:100%; height:500px" title="Decision Transformer Report"></iframe>