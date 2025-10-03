# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf

# Implementation TODOs:
# 1. iql_deterministic is true only for 2 datasets. Can we remote it?
# 2. MLP class introduced bugs in the past. We should remove it.
# 3. Refactor IQL updating code to be more consistent in style
import contextlib
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import d4rl
import numpy as np
import pyrallis
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm.auto import trange
import pandas as pd

TensorBatch = List[torch.Tensor]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class EvalConfig:
    project: str = "IQL-eval"
    group: str = "IQL-eval"
    name: str = "eval"
    eval_csv: str = "~/iqlpref/task_reward_iql_results/pen_results.csv"
    actor_path: str = (
        "~/iqlpref/human_pen_models/iql-D4RL/pen/human-v2-0f6ecda0/best_model.pt"
    )
    iql_deterministic: bool = False  # Use deterministic actor
    actor_dropout: Optional[float] = 0.1  # Adroit uses dropout for policy network
    # training params
    env: str = "D4RL/pen/human-v2"  # Minari remote dataset name
    normalize_state: bool = True  # Normalize states
    eval_episodes: int = 1000  # How many episodes run during evaluation
    eval_seed: int = 10

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        # epsilon should be already added in std.
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state, env.observation_space)
    return env


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(
            self.max_action * action, -self.max_action, self.max_action
        )
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(
                self(state) * self.max_action, -self.max_action, self.max_action
            )
            .cpu()
            .data.numpy()
            .flatten()
        )


@torch.no_grad()
def evaluate(
    env: gym.Env, actor: nn.Module, num_episodes: int, seed: int, device: str
) -> np.ndarray:
    episode_rewards = []
    for i in trange(num_episodes):
        done = False
        state, info = env.reset(seed=seed + i)

        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)
    return np.asarray(episode_rewards)


@pyrallis.wrap()
def eval_actor(config: EvalConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )
    eval_env = gym.make(config.env)
    dataset = d4rl.qlearning_dataset(eval_env)

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])

    if config.normalize_state:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    if config.iql_deterministic:
        actor = DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        ).to(DEVICE)
    else:
        actor = GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        ).to(DEVICE)

    actor_path = os.path.expanduser(config.actor_path)
    actor_path_split = actor_path.split("/")
    model_id = actor_path_split[-2].split("-")[-1]
    checkpoint_id = actor_path_split[-1].split(".")[0].split("_")[-1]
    if DEVICE == "cuda":
        actor.load_state_dict(torch.load(actor_path, weights_only=False)["actor"])
        actor.to(DEVICE)
    else:
        actor.load_state_dict(
            torch.load(actor_path, map_location=DEVICE, weights_only=False)["actor"]
        )
    actor.eval()
    normalized_scores = None
    mean_n_s = None
    std_n_s = None
    eval_scores = evaluate(
        env=eval_env,
        actor=actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
    )
    median_score = np.median(eval_scores)

    eval_csv = os.path.expanduser(config.eval_csv)
    try:
        df = pd.read_csv(config.eval_csv)
        new_row = pd.DataFrame(
            {
                "dataset": [config.env],
                "model_id": [model_id],
                "checkpoint_id": [checkpoint_id],
                "median_score": [median_score],
                "num_episodes": [config.eval_episodes],
            }
        )
        df = pd.concat([df, new_row], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame(
            {
                "dataset": [config.env],
                "model_id": [model_id],
                "checkpoint_id": [checkpoint_id],
                "median_score": [median_score],
                "num_episodes": [config.eval_episodes],
            }
        )
    df.to_csv(config.eval_csv, index=False)


if __name__ == "__main__":
    eval_actor()
