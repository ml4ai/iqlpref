"""Fixed qlearning_dataset_pt vs a slow per-row reference: whole window, timesteps and mask."""
import ast, sys
import numpy as np, torch
src = open(sys.argv[1]).read()
tree = ast.parse(src)
code = next(ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "qlearning_dataset_pt")
g = {"np": np, "torch": torch}
exec(code, g)
rng = np.random.default_rng(0)
W = rng.normal(size=(256, 3))

class Fingerprint(torch.nn.Module):
    """A scalar that depends on every window entry, its position, timestep and mask."""
    def __init__(self):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
    def forward(self, s, a, t, m):
        T = s.shape[1]
        w = torch.tensor(W[:T], dtype=torch.float64)
        v = (s.double().sum(-1) * w[:, 0] + a.double().sum(-1) * w[:, 1]
             + t.double() * w[:, 2] + m.double() * 0.37).sum(1)
        out = torch.zeros(s.shape[0], 1, T, 1, dtype=torch.float64)
        out[:, 0, -1, 0] = v
        return {"value": out}, None

def reference(ds, ql, fp):
    """Per row: own-episode window (episodes end at terminal or timeout), PT-repo convention."""
    N = len(ds["rewards"])
    start, out, keep = 0, [], []
    for i in range(N - 1):
        lo = max(start, i - ql + 1)
        L = i - lo + 1
        s = np.zeros((1, ql, ds["observations"].shape[1]), np.float32)
        a = np.zeros((1, ql, ds["actions"].shape[1]), np.float32)
        t = np.zeros((1, ql), np.int64); m = np.zeros((1, ql), np.float32)
        s[0, ql - L:] = ds["observations"][lo:i + 1]; a[0, ql - L:] = ds["actions"][lo:i + 1]
        t[0, ql - L:] = np.arange(L); m[0, ql - L:] = 1
        r, _ = fp(*map(torch.from_numpy, (s, a, t, m)))
        out.append(float(r["value"][0, 0, -1, 0])); keep.append(not ds["timeouts"][i])
        if ds["timeouts"][i] or ds["terminals"][i]:
            start = i + 1
    return np.array(out)[keep]

N = 700
ds = {"observations": rng.normal(size=(N, 5)).astype(np.float32),
      "actions": rng.normal(size=(N, 2)).astype(np.float32),
      "rewards": np.zeros(N, np.float32), "terminals": np.zeros(N, bool), "timeouts": np.zeros(N, bool)}
ds["timeouts"][[199, 399, 599]] = True
ds["terminals"][[50, 51, 300, 455]] = True            # mid-episode terminals, two in a row
fp = Fingerprint()
for ql in (1, 3, 25, 100):
    got = g["qlearning_dataset_pt"](None, fp, query_length=ql, dataset=ds)["rewards"]
    want = reference(ds, ql, fp)
    assert got.shape == want.shape, (got.shape, want.shape)
    err = np.abs(got - want).max()
    assert err < 1e-3 * (1 + np.abs(want).max()), (ql, err)
    print(f"query_length {ql:3d}: {len(got)} rows match the reference (max abs err {err:.2e})")
print("PT windows OK")
