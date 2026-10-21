"""Test the control code paths of iql_eval.py without d4rl: extract the functions and replay
the train() order (reward -> gauge -> modify_reward -> drop_terminals) on a toy dataset.

    python tests/iql_eval/test_controls.py algorithms/offline/iql_eval.py
"""
import ast
import sys

import numpy as np

src = open(sys.argv[1]).read()
tree = ast.parse(src)
want = {"heuristic_rewards", "gauge_reward", "modify_reward", "return_reward_range"}
code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name in want)
g = {"np": np}
exec("from typing import *\n" + code, g)
assert want <= set(g), set(g)

rng = np.random.default_rng(0)
N = 3000
obs = rng.normal(size=(N, 29))
nxt = obs.copy()
nxt[:, :2] += rng.normal(scale=0.1, size=(N, 2))
term = np.zeros(N, dtype=np.float32)
term[rng.choice(N, 40, replace=False)] = 1
base = {"observations": obs, "next_observations": nxt, "terminals": term.copy(),
        "timeouts": np.zeros(N), "rewards": rng.normal(size=N).astype(np.float32)}

# 1. heuristic: exactly the torso displacement
h = g["heuristic_rewards"](base, "path_length")
assert np.allclose(h, np.linalg.norm(nxt[:, :2] - obs[:, :2], axis=1), atol=1e-6)
try:
    g["heuristic_rewards"](base, "goal_progress")
    raise AssertionError("accepted an unknown heuristic")
except ValueError:
    pass


def pipeline(drop, idx=3, heuristic=False):
    d = {k: np.array(v, copy=True) for k, v in base.items()}
    if heuristic:
        d["rewards"] = g["heuristic_rewards"](d, "path_length")
    g["gauge_reward"](d, "max0")
    g["modify_reward"](d, "antmaze-medium-play-v2", idx)
    if drop:
        d["terminals"] = np.zeros_like(d["terminals"])
    return d


for heur in (False, True):
    for idx in range(1, 8):
        a, b = pipeline(False, idx, heur), pipeline(True, idx, heur)
        # the reward (and hence the normalization) is identical with and without the drop
        assert np.array_equal(a["rewards"], b["rewards"]), (heur, idx)
        assert a["terminals"].sum() == 40 and b["terminals"].sum() == 0
# gauge makes the heuristic <= 0 before normalization; index 3 then scales and shifts by -1
d = {k: np.array(v, copy=True) for k, v in base.items()}
d["rewards"] = g["heuristic_rewards"](d, "path_length")
g["gauge_reward"](d, "max0")
assert d["rewards"].max() == 0 and (d["rewards"] <= 0).all()
print("control paths OK")
