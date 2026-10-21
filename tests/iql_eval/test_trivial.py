"""Test iql_eval's trivial rewards without d4rl: the function, the train() order, and the config checks."""
import ast, dataclasses, os, sys, uuid
from typing import *
import numpy as np

src = open(sys.argv[1]).read()
tree = ast.parse(src)
fn = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
g = {"np": np}
exec("\n\n".join(fn[k] for k in ("trivial_rewards", "modify_reward", "return_reward_range")), g)

rng = np.random.default_rng(0)
N = 5000
r = (rng.random(N) < 0.01).astype(np.float32)          # a 0/1 task reward
d = {"rewards": r, "terminals": r.copy(), "timeouts": np.zeros(N)}
z, cz = g["trivial_rewards"](d, "zero")
m, cm = g["trivial_rewards"](d, "mean")
assert cz == 0.0 and (z == 0).all() and z.dtype == np.float32 and z.shape == r.shape
assert abs(cm - r.mean()) < 1e-7 and np.allclose(m, r.mean()) and m.dtype == np.float32
assert (d["rewards"] == r).all()                        # input untouched
try:
    g["trivial_rewards"](d, "one"); raise AssertionError("accepted 'one'")
except ValueError:
    pass

# the config checks: exec the TrainConfig class with its decorators' imports
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainConfig")
cg = {"dataclass": dataclasses.dataclass, "field": dataclasses.field, "os": os, "uuid": uuid,
      "Optional": Optional, "List": List, "Tuple": Tuple, "Union": Union, "Any": Any, "Dict": Dict}
exec(ast.get_source_segment(src, cls).replace("@dataclass", "@dataclass", 1), cg)
TC = dataclasses.dataclass(cg["TrainConfig"])   # the segment excludes the decorator
TC(trivial_reward="zero"); TC(trivial_reward="mean", drop_terminals=True)
for bad in (dict(trivial_reward="one"), dict(trivial_reward="zero", normalize_reward=3),
            dict(trivial_reward="mean", heuristic_reward="path_length"),
            dict(trivial_reward="zero", reward_model_root="~/x/antmaze_medium_play_mr_eval")):
    try:
        TC(**bad); raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
c = TC()
assert c.trivial_reward is None and not c.drop_terminals and c.heuristic_reward is None
# §14.10: a sequence reward model must declare the fixed labeller
TC(query_length=100, seq_labeller=2)
for bad in (dict(query_length=100), dict(query_length=100, seq_labeller=1)):
    try:
        TC(**bad); raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
TC(query_length=1)                                     # MR / BNN / task reward: unaffected
print("trivial rewards OK")
