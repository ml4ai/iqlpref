"""NMR end to end, without d4rl or pyrallis (handoff §14.2 item 1):
train the PyTorch NMR with PTTrainer on a real preference split, save it as the training
script does, reload it with iql_eval's loader, and label a toy dataset through the fixed
sequence labeller against a per-row own-episode reference.

    python tests/iql_eval/test_nmr.py algorithms/offline/iql_eval.py
"""
import ast
import dataclasses
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors"))
from optbnn.bnn.nets.nmr import NMR                      # noqa: E402
from optbnn.training.training import PTTrainer            # noqa: E402
from optbnn.utils import util                             # noqa: E402

src = open(sys.argv[1]).read()
tree = ast.parse(src)
fns = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
g = {"np": np, "torch": torch, "nn": nn, "os": os, "yaml": yaml, "RewardNMR": NMR,
     "Dict": Dict, "Any": Any}
exec("\n\n".join(fns[k] for k in ("_strip_compile_prefix", "load_nmr_reward_model",
                                   "qlearning_dataset_pt")), g)

torch.manual_seed(0)
# ---- 1. train on the real medium_play seed-1 split (a few epochs, CPU) ----
d = os.path.join(ROOT, "gp_reward-priors", "data", "antmaze", "antmaze-medium-play-v2", "eval", "seed_1")
train = util.Pref_H5Dataset(os.path.join(d, "antmaze-medium-play-v2_pref_train_1.hdf5"))
val = util.Pref_H5Dataset(os.path.join(d, "antmaze-medium-play-v2_pref_val_1.hdf5"))
(_, T, sdim), (_, _, adim) = train.shapes()
loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
vloader = torch.utils.data.DataLoader(val, batch_size=64)
net = NMR(sdim, adim, 32)
trainer = PTTrainer(net, opt=torch.optim.Adam(net.parameters(), lr=1e-3), device="cpu", bt_pool="mean")


def val_loss():
    return float(np.mean([trainer.evaluation(list(b))["eval_loss"] for b in vloader]))


before = val_loss()
for _ in range(3):
    for b in loader:
        trainer.train(list(b))
after = val_loss()
print(f"trained 3 epochs: val loss {before:.4f} -> {after:.4f}")
assert np.isfinite(after)

# ---- 2. save as the training script does; 3. reload with iql_eval's loader ----
with tempfile.TemporaryDirectory() as tmp:
    torch.save(trainer.state_dict(), os.path.join(tmp, "best_model.pt"))
    yaml.safe_dump({"embd_dim": 32, "embd_dropout": 0.1, "activations": "relu"},
                   open(os.path.join(tmp, "config.yaml"), "w"))
    loaded = g["load_nmr_reward_model"](tmp, sdim, adim)
    try:
        g["load_nmr_reward_model"](tmp, sdim + 1, adim)
        raise AssertionError("accepted a wrong state_dim")
    except ValueError:
        pass
net.eval()
b = next(iter(vloader))
with torch.no_grad():
    o1 = net(b[0], b[1])[0]["value"]
    o2 = loaded(b[0], b[1])[0]["value"]
assert torch.equal(o1, o2), "reloaded NMR differs"
assert not loaded.training
print("reload: identical outputs; eval mode")

# ---- 4. label through the fixed sequence labeller vs a per-row own-episode reference ----
rng = np.random.default_rng(0)
N, ql = 400, 25
ds = {"observations": rng.normal(size=(N, sdim)).astype(np.float32),
      "actions": rng.normal(size=(N, adim)).astype(np.float32),
      "rewards": np.zeros(N, np.float32), "terminals": np.zeros(N, bool), "timeouts": np.zeros(N, bool)}
ds["timeouts"][[149, 299]] = True
ds["terminals"][[60, 200]] = True
got = g["qlearning_dataset_pt"](None, loaded, query_length=ql, dataset=ds)["rewards"]
want, start = [], 0
with torch.no_grad():
    for i in range(N - 1):
        lo = max(start, i - ql + 1)
        L = i - lo + 1
        s = np.zeros((1, ql, sdim), np.float32)
        a = np.zeros((1, ql, adim), np.float32)
        s[0, ql - L:] = ds["observations"][lo:i + 1]
        a[0, ql - L:] = ds["actions"][lo:i + 1]
        if not ds["timeouts"][i]:
            want.append(float(loaded(torch.from_numpy(s), torch.from_numpy(a))[0]["value"][0, 0, -1, 0]))
        if ds["timeouts"][i] or ds["terminals"][i]:
            start = i + 1
want = np.array(want)
assert got.shape == want.shape and np.allclose(got, want, atol=1e-5), np.abs(got - want).max()
print(f"labelling: {len(got)} rows match the own-episode reference (max abs err {np.abs(got - want).max():.1e})")

# ---- 5. the config guards ----
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainConfig")
cg = {"dataclass": dataclasses.dataclass, "field": dataclasses.field, "os": os, "uuid": __import__("uuid"),
      "Optional": Optional, "List": List, "Tuple": Tuple, "Union": Union, "Any": Any, "Dict": Dict}
exec(ast.get_source_segment(src, cls), cg)
TC = dataclasses.dataclass(cg["TrainConfig"])
TC(nmr_reward_model=True, query_length=100, seq_labeller=2)
for bad in (dict(nmr_reward_model=True, query_length=1),
            dict(nmr_reward_model=True, query_length=100),                       # no seq_labeller
            dict(nmr_reward_model=True, query_length=100, seq_labeller=2, mr_ensemble=True)):
    try:
        TC(**bad)
        raise AssertionError(f"accepted {bad}")
    except ValueError:
        pass
print("NMR OK")
