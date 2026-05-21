# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../gp_reward-priors"))
from optbnn.bnn.nets.mlp import MLP as RewardMLP
from optbnn.bnn.nets.pref_trans import PT as RewardPT

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # wandb project name
    project: str = "IQL-pref"
    # wandb group name
    group: str = "IQL-D4RL"
    # wandb run name
    name: str = "IQL"
    # training dataset and evaluation environment
    env: str = "halfcheetah-medium-expert-v2"
    # discount factor
    discount: float = 0.99
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # actor update inverse temperature, similar to AWAC
    # small beta -> BC, big beta -> maximizing Q-value
    beta: float = 3.0
    # coefficient for asymmetric critic loss
    iql_tau: float = 0.7
    # whether to use deterministic actor
    iql_deterministic: bool = False
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 256
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL) (0 is none, other integers lead to different types of normalizations)
    normalize_reward: int = 0
    # V-critic function learning rate
    vf_lr: float = 3e-4
    # Q-critic learning rate
    qf_lr: float = 3e-4
    # actor learning rate
    actor_lr: float = 3e-4
    #  where to use dropout for policy network, optional
    actor_dropout: Optional[float] = None
    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    load_model: str = ""
    # file name for loading a reward model, optional
    reward_model_path: str = ""
    # required for when loading a reward model. If ==1 then MR, if >1 then PT
    query_length: int = 1
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.lerp_(sp.data, tau)


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
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = torch.randint(0, min(self._size, self._pointer), (batch_size,), device=self._device)
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._next_states[indices],
            self._dones[indices],
        ]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    # wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env,
    actor: nn.Module,
    device: str,
    n_episodes: int,
    discount: float,
    seed: int,
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        step = 0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward * (discount**step)
            step += 1
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    i = 0
    trj_lens = np.zeros(dataset["rewards"].shape[0])
    for j, (r, d) in enumerate(zip(dataset["rewards"], dataset["terminals"])):
        ep_ret += float(r)
        ep_len += 1
        trj_lens[i : j + 1] = ep_len
        if d or ep_len == max_episode_steps:
            i = j + 1
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns), trj_lens


def modify_reward(dataset, env_name, normalize_reward, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret, _ = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        if normalize_reward == 1:
            dataset["rewards"] -= 1.0
        elif normalize_reward == 2:
            min_ret, max_ret, _ = return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
        elif normalize_reward == 3:
            min_ret, max_ret, _ = return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
            dataset["rewards"] -= 1.0
        elif normalize_reward == 4:
            min_ret, max_ret, _ = return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] -= min_ret
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
        elif normalize_reward == 5:
            min_ret, max_ret, _ = return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] -= min_ret
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
            dataset["rewards"] -= 1.0
        elif normalize_reward == 6:
            min_ret, max_ret, trj_lens = return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] -= min_ret / trj_lens
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
        else:
            min_ret, max_ret, trj_lens = return_reward_range(dataset, max_episode_steps)
            dataset["rewards"] -= min_ret / trj_lens
            dataset["rewards"] /= max_ret - min_ret
            dataset["rewards"] *= max_episode_steps
            dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


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
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
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
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def qlearning_dataset_mr(env, r_model, dataset=None, terminate_on_end=False, **kwargs):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    use_timeouts = "timeouts" in dataset
    obs_all = dataset["observations"].astype(np.float32)
    act_all = dataset["actions"].astype(np.float32)

    # Single pass to build the keep mask and track episode steps.
    keep = np.ones(N - 1, dtype=bool)
    ep = 0
    for i in range(N - 1):
        done_bool = bool(dataset["terminals"][i])
        final = bool(dataset["timeouts"][i]) if use_timeouts else ep == env._max_episode_steps - 1
        if (not terminate_on_end) and final:
            keep[i] = False
            ep = 0
            continue
        if done_bool or final:
            ep = 0
        ep += 1

    # One batched forward pass for all N-1 transitions.
    device = next(r_model.parameters()).device
    obs_act = np.concatenate([obs_all[:-1], act_all[:-1]], axis=1)
    with torch.no_grad():
        all_rewards = r_model(
            torch.from_numpy(obs_act).to(device)
        ).squeeze(-1).cpu().numpy()

    return {
        "observations": obs_all[:-1][keep],
        "actions": act_all[:-1][keep],
        "next_observations": obs_all[1:][keep],
        "rewards": all_rewards[keep],
        "terminals": dataset["terminals"][:-1][keep],
    }


def qlearning_dataset_pt(
    env, r_model, query_length=100, dataset=None, terminate_on_end=False, **kwargs
):
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    use_timeouts = "timeouts" in dataset
    obs_all = dataset["observations"].astype(np.float32)
    act_all = dataset["actions"].astype(np.float32)
    s_dim, a_dim = obs_all.shape[1], act_all.shape[1]

    # Single pass: keep mask + per-transition episode step (used as window index).
    keep = np.ones(N - 1, dtype=bool)
    ep_steps = np.zeros(N - 1, dtype=np.int64)
    ep = 0
    for i in range(N - 1):
        ep_steps[i] = ep
        done_bool = bool(dataset["terminals"][i])
        final = bool(dataset["timeouts"][i]) if use_timeouts else ep == env._max_episode_steps - 1
        if (not terminate_on_end) and final:
            keep[i] = False
            ep = 0
            continue
        if done_bool or final:
            ep = 0
        ep += 1

    # Chunked batched inference.  All windows are left-padded to query_length so
    # we always read the reward from result["value"][:, 0, -1, 0] (last token).
    # Padded positions receive attn_mask=0 which the transformer masks to -1e4.
    device = next(r_model.parameters()).device
    CHUNK = 256
    all_rewards = np.zeros(N - 1, dtype=np.float32)
    ts_template = np.arange(query_length, dtype=np.int64)

    with torch.no_grad():
        for cs in range(0, N - 1, CHUNK):
            ce = min(cs + CHUNK, N - 1)
            B = ce - cs
            eps = ep_steps[cs:ce]

            sts_c = np.zeros((B, query_length, s_dim), dtype=np.float32)
            acts_c = np.zeros((B, query_length, a_dim), dtype=np.float32)
            ts_c = np.zeros((B, query_length), dtype=np.int64)
            am_c = np.zeros((B, query_length), dtype=np.float32)

            # Full-length windows — vectorised with advanced indexing.
            mask_full = eps >= query_length
            if mask_full.any():
                starts = eps[mask_full] - query_length + 1
                row_idx = starts[:, None] + ts_template[None, :]  # [M, query_length]
                sts_c[mask_full] = obs_all[row_idx]
                acts_c[mask_full] = act_all[row_idx]
                ts_c[mask_full] = ts_template
                am_c[mask_full] = 1.0

            # Short windows (only the first query_length steps of each episode).
            for j in np.where(~mask_full)[0]:
                ep_j = int(eps[j])
                seq_len = ep_j + 1
                pad = query_length - seq_len
                sts_c[j, pad:] = obs_all[:seq_len]
                acts_c[j, pad:] = act_all[:seq_len]
                ts_c[j, pad:] = ts_template[:seq_len]
                am_c[j, pad:] = 1.0

            result, _ = r_model(
                torch.from_numpy(sts_c).to(device),
                torch.from_numpy(acts_c).to(device),
                torch.from_numpy(ts_c).to(device),
                torch.from_numpy(am_c).to(device),
            )
            # value shape: [B, 1, query_length, 1] — take last token.
            all_rewards[cs:ce] = result["value"][:, 0, -1, 0].cpu().numpy()

    return {
        "observations": obs_all[:-1][keep],
        "actions": act_all[:-1][keep],
        "next_observations": obs_all[1:][keep],
        "rewards": all_rewards[keep],
        "terminals": dataset["terminals"][:-1][keep],
    }


def load_mlp_reward_model(model_dir: str, device: str = "cpu") -> nn.Module:
    with open(os.path.join(model_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device, weights_only=False)
    state = ckpt["net"]
    input_dim = state["layers.0.W"].shape[0]
    hidden_dims = [state["layers.0.W"].shape[1]]
    i = 1
    while f"layers.linear_{i}.W" in state:
        hidden_dims.append(state[f"layers.linear_{i}.W"].shape[1])
        i += 1
    model = RewardMLP(input_dim, 1, hidden_dims, cfg.get("activations", "relu")).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_pt_reward_model(model_dir: str, device: str = "cpu") -> nn.Module:
    with open(os.path.join(model_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"), map_location=device, weights_only=False)
    state = ckpt["net"]
    state_dim = state["state_linear.weight"].shape[1]
    action_dim = state["action_linear.weight"].shape[1]
    embd_dim = state["state_linear.weight"].shape[0]
    max_episode_steps = state["timestep_embed.weight"].shape[0] - 1
    pref_attn_embd_dim = (state["pref_linear.weight"].shape[0] - 1) // 2
    num_layers = 0
    while f"gpt.layers.{num_layers}.layer_norm_0.weight" in state:
        num_layers += 1
    max_pos = state["gpt.layers.0.attention.causal_bias"].shape[2]
    intermediate_dim = cfg.get("intermediate_dim") or (4 * embd_dim)
    model = RewardPT(
        state_dim=state_dim,
        action_dim=action_dim,
        max_episode_steps=max_episode_steps,
        embd_dim=embd_dim,
        pref_attn_embd_dim=pref_attn_embd_dim,
        num_heads=cfg.get("num_heads", 4),
        attn_dropout=cfg.get("attn_dropout", 0.1),
        resid_dropout=cfg.get("resid_dropout", 0.1),
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        embd_dropout=cfg.get("embd_dropout", 0.1),
        max_pos=max_pos,
        eps=cfg.get("model_eps", 1e-5),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@pyrallis.wrap()
def train(config: TrainConfig):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    if config.reward_model_path:
        reward_model_path = os.path.expanduser(config.reward_model_path)
        # Run reward inference on the training device, then free GPU memory.
        if config.query_length > 1:
            reward_model = load_pt_reward_model(reward_model_path, device=config.device)
            dataset = qlearning_dataset_pt(env, reward_model, config.query_length)
        else:
            reward_model = load_mlp_reward_model(reward_model_path, device=config.device)
            dataset = qlearning_dataset_mr(env, reward_model)
        del reward_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env, config.normalize_reward)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    use_fused = config.device.startswith("cuda") and torch.cuda.is_available()
    adam_kwargs = {"fused": True} if use_fused else {}
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr, **adam_kwargs)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr, **adam_kwargs)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr, **adam_kwargs)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file, weights_only=False))
        actor = trainer.actor

    # Compile networks for faster execution (PyTorch 2.0+).
    if hasattr(torch, "compile"):
        trainer.qf = torch.compile(trainer.qf)
        trainer.q_target = torch.compile(trainer.q_target)
        trainer.vf = torch.compile(trainer.vf)
        trainer.actor = torch.compile(trainer.actor)
        actor = trainer.actor

    wandb_init(asdict(config))

    for t in range(int(config.max_timesteps)):
        # Replay buffer already lives on config.device — no .to() needed.
        batch = replay_buffer.sample(config.batch_size)
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                discount=config.discount,
                seed=config.seed,
            )
            mean_eval_score = eval_scores.mean()
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            wandb.log(
                {
                    "mean_score": mean_eval_score,
                },
                step=trainer.total_it,
            )


if __name__ == "__main__":
    train()