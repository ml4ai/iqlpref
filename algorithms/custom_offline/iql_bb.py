# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf

# Implementation TODOs:
# 1. iql_deterministic is true only for 2 datasets. Can we remote it?
# 2. MLP class introduced bugs in the past. We should remove it.
# 3. Refactor IQL updating code to be more consistent in style
import contextlib
import copy
import os
import sys
import random
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flax import nnx
import orbax.checkpoint as ocp
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler, BatchSampler
from tqdm.auto import trange

sys.path.insert(0, os.path.abspath("../"))

from CORL.reward_models.pref_transformer import load_PT

import h5py

TensorBatch = List[torch.Tensor]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # wandb params
    project: str = "CORL"
    group: str = "IQL-BB"
    name: str = "iql"
    # model params
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # training params
    dataset_id: str = "bbway1"
    dataset_path: str = (
        "~/CORL/t0012/reward_data_1/bbway1_t0012.hdf5"  # Minari remote dataset name
    )
    reward_model_path: str = "~/CORL/t0012/pt_rewards_1/best_model.ckpt"
    move_stats_path: str = "~/CORL/t0012/cache/p_stats.npy"
    update_steps: int = int(1e6)  # Total training networks updates
    batch_size: int = 256  # Batch size for all networks
    normalize_reward: bool = False  # Normalize reward
    # evaluation params
    eval_every: int = int(5e3)  # How often (time steps) we evaluate
    eval_episodes: int = 10  # How many episodes run during evaluation
    # general params
    train_seed: int = 0
    eval_seed: int = 0
    checkpoints_path: Optional[str] = None  # Save path

    def __post_init__(self):
        self.name = f"{self.name}-{self.dataset_id}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def load_stats(load_file):
    load_file = os.path.expanduser(load_file)
    return tuple(np.load(load_file, fix_imports=False))


def set_seed(seed: int, deterministic_torch: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


# This is how reward normalization among all datasets is done in original IQL
def return_reward_range(
    dataset: Dict[str, np.ndarray], max_episode_steps: int
) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


class IQL_H5Dataset(torch.utils.data.Dataset):
    # The task rewards flag overwrite normalized_rewards flag
    # Some environment developers recommended adjusting the task reward function by some constant,
    # this can be set through reward_adjustment.
    def __init__(
        self,
        file_path,
        normalized_rewards=True,
        reward_adjustment=0.0,
        device: str = "cpu",
    ):
        super(IQL_H5Dataset, self).__init__()
        self.file_path = file_path
        self.normalized_rewards = normalized_rewards
        self.reward_adjustment = reward_adjustment
        self._device = device
        with h5py.File(self.file_path, "r") as f:
            self._sts_shape = f["states"].shape
            self._acts_shape = f["actions"].shape
            self._max_speed = np.percentile(f["actions"][:, 0], 99)
            self._min_speed = 0.0
            self._max_angle = 180.0
            self._min_angle = -180.0

    def open_hdf5(self):
        self.h5_file = h5py.File(self.file_path, "r")
        self.states = self.h5_file["states"]
        self.next_states = self.h5_file["next_states"]
        self.actions = self.h5_file["actions"]
        self.attn_mask = self.h5_file["attn_mask"]
        if self.normalized_rewards:
            self.rewards = self.h5_file["n_rewards"]
        else:
            self.rewards = self.h5_file["rewards"]

    def __getitem__(self, index):
        if not hasattr(self, "h5_file"):
            self.open_hdf5()
        return (
            torch.tensor(
                self.states[index, ...], dtype=torch.float32, device=self._device
            ),
            torch.tensor(
                self.actions[index, ...], dtype=torch.float32, device=self._device
            ),
            torch.tensor(
                self.rewards[index, ...] + self.reward_adjustment,
                dtype=torch.float32,
                device=self._device,
            ),
            torch.tensor(
                self.next_states[index, ...], dtype=torch.float32, device=self._device
            ),
            torch.tensor(
                self.attn_mask[index, ...], dtype=torch.float32, device=self._device
            ),
        )

    def __len__(self):
        return self._sts_shape[0]

    def shapes(self):
        return self._sts_shape, self._acts_shape

    def max_actions(self):
        return torch.tensor([self._max_speed, self._max_angle], device=self._device)

    def min_actions(self):
        return torch.tensor([self._min_speed, self._min_angle], device=self._device)

class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(int(self.n_batches) * self.batch_size, self.dataset_length)
            for index in idx:
                yield int(index)

def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset, batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(RandomBatchSampler(dataset, batch_size), batch_size=batch_size, drop_last=drop_last)
    )



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
        max_actions: torch.Tensor,
        min_actions: torch.Tensor,
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
        self.max_actions = max_actions
        self.min_actions = min_actions

    def forward(self, sts: torch.Tensor) -> Normal:
        mean = self.net(sts)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(action, self.min_actions, self.max_actions)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_actions: torch.Tensor,
        min_actions: torch.Tensor,
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
        self.max_actions = max_actions
        self.min_actions = min_actions

    def forward(self, sts: torch.Tensor) -> torch.Tensor:
        return self.net(sts)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state), self.min_actions, self.max_actions)
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
        max_actions: torch.Tensor,
        min_actions: torch.Tensor,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        actor_lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_actions = max_actions
        self.min_actions = min_actions
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_scheduler = actor_lr_scheduler
        self.iql_tau = iql_tau
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def _update_v(self, states, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(states, actions)

        v = self.vf(states)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        attn_mask: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + attn_mask.float() * self.gamma * next_v.detach()
        qs = self.qf.both(states, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(states)
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
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            states,
            actions,
            rewards,
            next_states,
            attn_mask,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_states)
        # Update value function
        adv = self._update_v(states, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        attn_mask = attn_mask.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, states, actions, rewards, attn_mask, log_dict)
        # Update actor
        self._update_policy(adv, states, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_lr_scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_scheduler.load_state_dict(state_dict["actor_lr_scheduler"])


# @torch.no_grad()
# def evaluate(
#     env: gym.Env, actor: nn.Module, num_episodes: int, seed: int, device: str
# ) -> np.ndarray:
#     actor.eval()
#     episode_rewards = []
#     for i in range(num_episodes):
#         done = False
#         state, info = env.reset(seed=seed + i)

#         episode_reward = 0.0
#         while not done:
#             action = actor.act(state, device)
#             state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             episode_reward += reward
#         episode_rewards.append(episode_reward)

#     actor.train()
#     return np.asarray(episode_rewards)


def rand_circle(R, N, C=(0, 0), rng=np.random.default_rng()):
    r = R * np.sqrt(rng.random(N))
    theta = rng.random(N) * 2 * np.pi
    return C[0] + r * np.cos(theta), C[1] + r * np.sin(theta)


def point_dist(vecX, vecY, px, py):
    return np.sqrt(((vecX - px) ** 2) + ((vecY - py) ** 2)) * 1


def cos_plus(degrees):
    res = np.cos(degrees * (np.pi / 180.0))
    res = np.where(np.isclose(degrees, 90), 0.0, res)
    res = np.where(np.isclose(degrees, 270), 0.0, res)
    return res * 1


def sin_plus(degrees):
    res = np.sin(degrees * (np.pi / 180.0))
    res = np.where(np.isclose(degrees, 360), 0.0, res)
    res = np.where(np.isclose(degrees, 180), 0.0, res)
    return res * 1


# Find closest point on line a-b to point p
def closest_point_on_line(ax, ay, bx, by, px, py, thres):
    apx = px - ax
    apy = py - ay
    abx = bx - ax
    aby = by - ay

    ab2 = (abx**2) + (aby**2)
    # Accounts for obstacles wrapping around map
    cond = ab2 < (thres**2)
    apab = apx * abx + apy * aby
    if isinstance(cond, bool):
        t = np.asarray(apab) / np.asarray(ab2)
        t = np.where(np.isnan(t), 0.0, t)
        t = np.where(t < 0, 0.0, t)
        t = np.where(t > 1, 1.0, t)
        return ax + abx * t, ay + aby * t

    t = apab[cond] / ab2[cond]
    t = np.where(np.isnan(t), 0.0, t)
    t = np.where(t < 0, 0.0, t)
    t = np.where(t > 1, 1.0, t)
    return (ax[cond] + abx[cond] * t) * 1, (ay[cond] + aby[cond] * t) * 1


def point_collide(x1, y1, x2, y2, radius_1, radius_2=None):
    if radius_2 is None:
        radius_2 = radius_1
    dist = ((x1 - x2) ** 2) + ((y1 - y2) ** 2)
    tol = (radius_1 + radius_2) ** 2
    l = dist < tol
    e = np.isclose(dist, tol)
    return l | e


def collision(
    old_x, old_y, new_x, new_y, px, py, radius_1=0.3, radius_2=None, thres=2.0
):
    cpx, cpy = closest_point_on_line(old_x, old_y, new_x, new_y, px, py, thres)
    return (
        np.any(point_collide(cpx, cpy, px, py, radius_1, radius_2)),
        cpx * 1,
        cpy * 1,
    )


def find_direction(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    degs = np.arctan2(y, x) * (180.0 / np.pi)
    degs = np.where(np.isclose(degs, 0.0), 360.0, degs)
    degs = np.where(degs < 0, degs + 360.0, degs)
    return degs * 1


def first_nth_argmins(arr, n):
    """
    Returns the indices of the 0 to nth minimum values in a NumPy array.

    Parameters:
    arr (numpy.ndarray): The input NumPy array.
    n (int): The number of minimum values to consider (inclusive of 0th minimum).

    Returns:
    numpy.ndarray: An array containing the indices of the 0 to nth minimum values.
                   Returns an empty array if n is negative or greater than or equal to the array size.
    """
    if n < 0 or n > arr.size:
        return np.array([])

    indices = np.argpartition(arr, np.arange(n))[:n]
    return indices


@torch.no_grad()
def bb_run_eval_IQL(
    actor,
    num_episodes,
    r_model,
    move_stats,
    max_horizon=500,
    n_min_obstacles=6,
    days=181,
    context_length=100,
    seed=4,
    device="cpu",
):
    actor.eval()
    episode_returns = []
    rng = np.random.default_rng(seed)
    for i in range(num_episodes):
        level = rng.choice([9, 10, 11])
        n_obstacles = 50 if level == 9 else 100 if level == 10 else 150
        ai = rng.choice([1, 2, 3, 4])
        attempt = rng.choice(4)
        day = rng.choice(days)

        O_posX, O_posY = rand_circle(50, n_obstacles, rng=rng)

        O_angle = rng.uniform(0.0, 360.0, n_obstacles)

        while True:
            p_samp = rand_circle(50, None, rng=rng)
            if np.all(
                ((p_samp[0] - O_posX[0]) ** 2) + ((p_samp[1] - O_posY[0]) ** 2) > 1
            ):
                break

        p_posX = float(p_samp[0])
        p_posY = float(p_samp[1])

        while True:
            g_h = rng.uniform(0.0, 360.0)
            g_r = rng.normal(30)
            g = (
                float(p_posX + g_r * cos_plus(g_h)),
                float(p_posY + g_r * sin_plus(g_h)),
            )
            if ((g[0] ** 2) + (g[1] ** 2)) <= 2500:
                break

        def create_new_state():
            s = [p_posX, p_posY]
            obs_distances = point_dist(
                O_posX,
                O_posY,
                p_posX,
                p_posY,
            )

            min_dist_obs = first_nth_argmins(obs_distances, n_min_obstacles)

            for i in range(n_min_obstacles):
                s += [
                    O_posX[min_dist_obs[i]],
                    O_posY[min_dist_obs[i]],
                    O_angle[min_dist_obs[i]],
                ]

            s += [g[0], g[1]]

            s.append(level * 1.0)
            s.append(ai * 1.0)
            s.append(attempt * 1.0)

            if days is not None:
                s.append(day * 1.0)

            return np.asarray(s)

        s = create_new_state().reshape(1, 1, -1)
        a = np.zeros((1, 0, 2))
        t = np.zeros((1, 1), dtype=np.int32)

        episode_return = 0.0
        for i in range(max_horizon):
            action = actor.act(s[-1, -1], device)
            a = np.concat([a, action.reshape(1, 1, -1)], axis=1)
            a = a[:, -context_length:, :]

            reward, _ = r_model(
                s,
                a,
                t,
                np.ones((1, t.shape[1]), dtype=np.float32),
                training=False,
            )
            reward = reward["value"][:, 0, -1]

            old_p_posX = p_posX
            old_p_posY = p_posY
            p_posX = float(p_posX + (action[0] * cos_plus(action[1])))
            p_posY = float(p_posY + (action[0] * sin_plus(action[1])))

            coll, _, _ = collision(
                old_p_posX,
                old_p_posY,
                p_posX,
                p_posY,
                O_posX,
                O_posY,
            )
            o_dists = rng.normal(move_stats[2], move_stats[3], n_obstacles)
            old_O_posX = O_posX
            old_O_posY = O_posY
            O_posX = O_posX + (o_dists * cos_plus(O_angle))
            O_posY = O_posY + (o_dists * sin_plus(O_angle))

            g_o_dist = np.sqrt((O_posX**2) + (O_posY**2))
            O_posX = np.where(g_o_dist > 50.0, -old_O_posX, O_posX)
            O_posY = np.where(g_o_dist > 50.0, -old_O_posY, O_posY)

            coll, _, _ = collision(
                old_O_posX,
                old_O_posY,
                O_posX,
                O_posY,
                p_posX,
                p_posY,
            )

            coll, _, _ = collision(
                old_p_posX,
                old_p_posY,
                p_posX,
                p_posY,
                g[0],
                g[1],
                radius_2=1.0,
            )

            s = np.concat([s, create_new_state().reshape(1, 1, -1)], axis=1)

            s = s[:, -context_length:, :]

            t = np.concat([t, (t[-1][-1] + 1).reshape(1, -1)], axis=1)
            t = t[:, -context_length:]

            episode_return += reward
            if coll:
                break
        episode_returns.append(episode_return)
    actor.train()
    return np.asarray(episode_returns)


@pyrallis.wrap()
def train(config: TrainConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True,
    )

    dataset_path = os.path.expanduser(config.dataset_path)
    reward_model_path = os.path.expanduser(config.reward_model_path)

    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler())
    reward_model = load_PT(
        reward_model_path, checkpointer, on_cpu=not torch.cuda.is_available()
    )
    reward_model = nnx.jit(reward_model, static_argnums=4)
    checkpointer.close()
    move_stats = load_stats(config.move_stats_path)

    data = IQL_H5Dataset(
        dataset_path, normalized_rewards=config.normalize_reward, device=DEVICE
    )

    training_data_loader = fast_loader(
        data,
        batch_size=config.batch_size,
    )
    interval = len(data) / config.batch_size
    if int(interval) < interval:
        interval = int(interval + 1)
    else:
        interval = int(interval)
    state_shape, action_shape = data.shapes()
    state_dim = state_shape[1]
    action_dim = action_shape[1]
    max_actions = data.max_actions()
    min_actions = data.min_actions()

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    set_seed(config.train_seed)

    q_network = TwinQ(state_dim, action_dim).to(DEVICE)
    v_network = ValueFunction(state_dim).to(DEVICE)
    if config.iql_deterministic:
        actor = DeterministicPolicy(
            state_dim,
            action_dim,
            max_actions,
            min_actions,
            dropout=config.actor_dropout,
        ).to(DEVICE)
    else:
        actor = GaussianPolicy(
            state_dim,
            action_dim,
            max_actions,
            min_actions,
            dropout=config.actor_dropout,
        ).to(DEVICE)

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor_lr_scheduler = CosineAnnealingLR(actor_optimizer, config.update_steps)

    trainer = ImplicitQLearning(
        max_actions=max_actions,
        min_actions=min_actions,
        actor=actor,
        actor_optimizer=actor_optimizer,
        actor_lr_scheduler=actor_lr_scheduler,
        q_network=q_network,
        q_optimizer=q_optimizer,
        v_network=v_network,
        v_optimizer=v_optimizer,
        iql_tau=config.iql_tau,
        beta=config.beta,
        gamma=config.gamma,
        tau=config.tau,
        device=DEVICE,
    )
    best_score = -np.inf
    # normalized_score = None
    best_step = 0
    for step in trange(config.update_steps):
        if step % interval == 0:
            tdl = iter(training_data_loader)
        batch = [b.to(DEVICE) for b in next(tdl)]
        log_dict = trainer.train(batch)

        wandb.log(log_dict, step=step)

        if (step + 1) % config.eval_every == 0:
            eval_scores = bb_run_eval_IQL(
                actor=actor,
                num_episodes=config.eval_episodes,
                r_model=reward_model,
                move_stats=move_stats,
                seed=config.eval_seed + step,
                device=DEVICE,
            )
            mean_eval = eval_scores.mean()
            wandb.log({"evaluation_return": mean_eval}, step=step)
            # optional normalized score logging, only if dataset has reference scores
            # with contextlib.suppress(ValueError):
            #     normalized_score = (
            #         minari.get_normalized_score(dataset, eval_scores).mean() * 100
            #     )
            #     wandb.log({"normalized_score": normalized_score}, step=step)

            # if normalized_score is not None:
            #     if normalized_score > best_score:
            #         best_score = normalized_score
            #         best_step = step
            #         if config.checkpoints_path is not None:
            #             torch.save(
            #                 trainer.state_dict(),
            #                 os.path.join(config.checkpoints_path, f"best_model.pt"),
            #             )
            # else:
            #     if mean_eval > best_score:
            #         best_score = mean_eval
            #         best_step = step
            #         if config.checkpoints_path is not None:
            #             torch.save(
            #                 trainer.state_dict(),
            #                 os.path.join(config.checkpoints_path, f"best_model.pt"),
            #             )

            if mean_eval > best_score:
                best_score = mean_eval
                best_step = step
                if config.checkpoints_path is not None:
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.checkpoints_path, f"best_model.pt"),
                    )
            wandb.log({"best_score_so_far": best_score}, step=step)
            wandb.log({"best_step_so_far": best_step}, step=step)
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{step}.pt"),
                )


if __name__ == "__main__":
    train()
