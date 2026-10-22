"""load_pt_reward_model rebuilds the PT the training script trained -- including its
number of attention heads, which the training script derives in __post_init__
(embd_dim // head_dim) and so never writes to config.yaml (handoff 14.28).

    /opt/anaconda3/envs/irl/bin/python tests/iql_eval/test_pt_loader.py algorithms/offline/iql_eval.py
"""
import ast
import os
import sys
import tempfile
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors"))
from optbnn.bnn.nets.pref_trans import PT                  # noqa: E402

src = open(sys.argv[1]).read()
fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
g = {"np": np, "torch": torch, "nn": nn, "os": os, "yaml": yaml, "RewardPT": PT, "Dict": Dict, "Any": Any}
exec("\n\n".join(fns[k] for k in fns if k in ("_strip_compile_prefix", "load_pt_reward_model", "_pt_num_heads")), g)

# config.yaml as run_pt_training_antmaze_eval.py writes it: pyrallis.dump of the
# dataclass FIELDS after __post_init__ -- embd_dim / head_dim expanded, num_heads absent.
CASES = {   # variant: (embd exponent, head exponent, layers) -- the four PT winners
    "medium_play": (5, 3, 4), "medium_diverse": (3, 5, 2), "large_play": (5, 3, 2), "large_diverse": (5, 5, 3)}
torch.manual_seed(0)
s, a = torch.randn(3, 100, 29), torch.randn(3, 100, 8)
t, m = torch.arange(100).repeat(3, 1), torch.ones(3, 100)
bad = []
for v, (e, h, L) in CASES.items():
    embd, head = 2 ** e, min(2 ** h, 2 ** e)
    heads = embd // head
    net = PT(29, 8, 99, embd, embd, heads, 0.1, 0.1, 4 * embd, L, 0.1, 2048, 0.1).eval()
    cfg = {"embd_dim": embd, "head_dim": head, "pref_attn_embd_dim": embd, "intermediate_dim": 4 * embd,
           "num_layers": L, "attn_dropout": 0.1, "resid_dropout": 0.1, "embd_dropout": 0.1, "model_eps": 0.1}
    with tempfile.TemporaryDirectory() as d:
        torch.save({"net": net.state_dict()}, os.path.join(d, "best_model.pt"))
        yaml.safe_dump(cfg, open(os.path.join(d, "config.yaml"), "w"))
        got = g["load_pt_reward_model"](d)
    with torch.no_grad():
        same = torch.allclose(net(s, a, t, m)[0]["value"], got(s, a, t, m)[0]["value"], atol=1e-6)
    print(f"  {v:15s} embd {embd:3d} head {head:3d} -> {heads} head(s): reloaded outputs {'IDENTICAL' if same else 'DIFFER'}")
    if not same:
        bad.append(v)
assert not bad, f"load_pt_reward_model mis-builds: {bad}"
print("PT loader OK")
