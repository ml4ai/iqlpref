"""Test eval_actor's final-position capture with a fake auto-resetting vector env."""
import ast
import sys
import types
from functools import partial

import numpy as np
import torch

src = open(sys.argv[1]).read()
tree = ast.parse(src)
code = next(ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "eval_actor")
code = code.replace("@torch.inference_mode()", "")

MEAN = np.array([10.0, 20.0, 0.0])
STD = np.array([2.0, 4.0, 1.0])


class FakeVec:
    """Each env walks in x; episode e of env i lasts 3 + i steps, then auto-resets.
    Returns NORMALIZED obs, like the wrapped eval envs.  info_mode picks the format."""
    mode = "tuple"

    def __init__(self, fns):
        self.n = len(fns)
        self.t = np.zeros(self.n, int)

    def _obs(self, i, t):
        raw = np.array([float(t), float(100 + i), 0.0])
        return (raw - MEAN) / STD

    def reset(self):
        return np.stack([self._obs(i, 0) for i in range(self.n)])

    def step(self, actions):
        self.t += 1
        obs, rew, done, infos = [], [], [], []
        for i in range(self.n):
            d = self.t[i] >= 3 + i
            info = {}
            if d:
                term = self._obs(i, self.t[i])
                if FakeVec.mode == "tuple":
                    info["terminal_observation"] = term
                self.t[i] = 0
            obs.append(self._obs(i, self.t[i]))
            rew.append(1.0 if (d and i % 2 == 0) else 0.0)     # even envs "succeed"
            done.append(d)
            infos.append(info)
        if FakeVec.mode == "dict":
            infos = {"final_observation": np.array([self._obs(i, 3 + i) if done[i] else None
                                                    for i in range(self.n)], dtype=object)}
        if FakeVec.mode == "none":
            infos = tuple({} for _ in range(self.n))
        return np.stack(obs), np.array(rew), np.array(done), infos

    def close(self):
        pass


class Actor(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], 1)


gym = types.SimpleNamespace(vector=types.SimpleNamespace(AsyncVectorEnv=FakeVec))
g = {"np": np, "torch": torch, "gym": gym, "partial": partial, "nn": torch.nn,
     "_make_eval_env": lambda *a: None}
exec("from typing import *\n" + code, g)
eval_actor = g["eval_actor"]

for mode, off_by_one in (("tuple", False), ("dict", False), ("none", True)):
    FakeVec.mode = mode
    scores, steps, finals = eval_actor("antmaze-medium-play-v2", Actor(), 1.0, MEAN, STD, "cpu",
                                       n_episodes=4, seed=0, n_envs=4, return_final=True)
    assert len(finals) == 4 and len(scores) == 4
    for (x, y, s, l) in finals:
        i = int(round(y)) - 100                     # env index from the raw y
        assert l == 3 + i
        want_x = (3 + i) - (1 if off_by_one else 0)  # the fallback is one step early
        assert abs(x - want_x) < 1e-9, (mode, x, want_x)
        assert s == int(i % 2 == 0)
    # with the option off: exactly the old two-value return
    out = eval_actor("antmaze-medium-play-v2", Actor(), 1.0, MEAN, STD, "cpu",
                     n_episodes=4, seed=0, n_envs=4)
    assert len(out) == 2
print("final positions OK (terminal_observation, final_observation, and the fallback)")
