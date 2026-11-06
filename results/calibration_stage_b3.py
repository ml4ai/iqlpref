#!/usr/bin/env python
"""calibration_stage_b3.py -- is the BNN's 2x logit under-scaling a shrinkage of the WITHIN-CELL
(motion) part of its reward?  (gp_reward-priors HANDOFF §14.100)

§14.98-§14.99: the BNN's logits are 0.2-0.4 of MR's on the same pairs and seeds, in every chain,
nearly uniformly over the pairs' positional separation.  The map prior's kernel ties every input
in a maze cell together except for its nugget (amp2 * sig_n2), so variation WITHIN a cell -- body
state, velocity, the motion cue the human labels mostly follow (§14.94) -- has little prior room.
B3 tests whether that part, specifically, is what the BNN shrinks.

SCOPE: §14.92's exception (the BNN reward models' calibration; no IQL result).  POST HOC,
DESCRIPTIVE.  Box output: exp/calibration_b3/<variant>_<seed>.npz (gitignored); analysis text
belongs in gp_reward-priors/calibration_results/held_bnn_cvar/mechanism_stage_b3.txt.

measure (box), per variant x seed 1-10, on the seed's validation pairs:
    r_bnn   [2, B, T] the BNN posterior-mean reward per step (each draw centred by its mean
            over every valid step, as the gate centres; the mean over all draws)
    r_mr    [2, B, T] the MR anchor model's reward per step (MR's own input: state ++ action)
    cell    [2, B, T] each step's maze cell, from the LIVE layout (diagnose_sampling_tail._live_prior)
    speed   [2, B, T] the torso's step length |xy_{t+1} - xy_t| (the last valid step repeats)
    mask    [2, B, T] valid steps;  y1 [B]
analyse (here), per model, over the seed's valid validation steps:
    positional part   each cell's mean reward, on its steps;   within-cell part   the rest
    (1) sd of each part, BNN / MR
    (2) within-cell slope of reward on speed (both centred within cells), BNN / MR
    (3) each part's pair logits (mean-pooled, as training pools), median |d_BNN| / |d_MR|
    (4) each part's accuracy on the human labels, per model
    Cross-check first: the BNN's pooled per-step mean reproduces the gate's d_mean.

    cd ~/iqlpref && python results/calibration_stage_b3.py measure [--device cuda]
    /opt/anaconda3/envs/irl/bin/python results/calibration_stage_b3.py analyse \\
        > gp_reward-priors/calibration_results/held_bnn_cvar/mechanism_stage_b3.txt
    /opt/anaconda3/envs/irl/bin/python results/calibration_stage_b3.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPR = os.path.join(ROOT, "gp_reward-priors")
for p in (HERE, ROOT, GPR, os.path.join(GPR, "scripts_bnn")):
    sys.path.insert(0, p)

OUT = os.path.join(ROOT, "exp", "calibration_b3")
MEAS = os.path.join(GPR, "calibration_results", "measurements")
VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
SEEDS = tuple(range(1, 11))


# --------------------------------------------------------------------------
# Pure pieces (self-tested)
# --------------------------------------------------------------------------
def step_speed(xy, mask):
    """[.., T, 2] torso positions, [.., T] mask -> [.., T] step lengths; the last valid step repeats."""
    d = np.linalg.norm(np.diff(xy, axis=-2), axis=-1)
    sp = np.concatenate([d, d[..., -1:]], axis=-1)
    m = mask.astype(bool)
    nxt = np.concatenate([m[..., 1:], np.zeros_like(m[..., :1])], axis=-1)
    last = m & ~nxt                                           # last valid step: copy the one before
    prev = np.concatenate([sp[..., :1], sp[..., :-1]], axis=-1)
    return np.where(last, prev, sp)


def split(r, cell, mask):
    """Positional (cell-mean) and within-cell parts of r over the valid steps."""
    m = mask.astype(bool)
    pos = np.zeros_like(r, dtype=float)
    cv, rv = cell[m], r[m]
    means = {c: rv[cv == c].mean() for c in np.unique(cv)}
    pos[m] = np.array([means[c] for c in cv])
    within = np.where(m, r - pos, 0.0)
    return pos, within


def pooled_d(part, mask):
    """[2, B, T] -> [B] mean-pooled logit difference (segment 1 - segment 2)."""
    m = mask.astype(float)
    f = (part * m).sum(-1) / np.maximum(m.sum(-1), 1.0)
    return f[0] - f[1]


def within_slope(within, speed, cell, mask):
    m = mask.astype(bool)
    _, sw = split(speed, cell, mask)
    x, y = sw[m], within[m]
    return float((x @ y) / max(float(x @ x), 1e-12))


def seed_stats(rec):
    """-> dict of the four comparisons for one seed (BNN vs MR)."""
    mask, cell, y = rec["mask"], rec["cell"], np.asarray(rec["y1"]) > 0.5
    out = {}
    for k in ("bnn", "mr"):
        r = rec[f"r_{k}"]
        pos, within = split(r, cell, mask)
        m = mask.astype(bool)
        out[k] = {"sd_pos": float(pos[m].std()), "sd_within": float(within[m].std()),
                  "slope": within_slope(within, rec["speed"], cell, mask),
                  "d_pos": pooled_d(pos, mask), "d_within": pooled_d(within, mask), "d": pooled_d(r, mask)}
        out[k]["acc_pos"] = float(((out[k]["d_pos"] > 0) == y).mean())
        out[k]["acc_within"] = float(((out[k]["d_within"] > 0) == y).mean())
        out[k]["acc"] = float(((out[k]["d"] > 0) == y).mean())
    b, m_ = out["bnn"], out["mr"]
    med = lambda a, c: float(np.median(np.abs(a) / np.maximum(np.abs(c), 1e-9)))
    out["ratio"] = {"sd_pos": b["sd_pos"] / max(m_["sd_pos"], 1e-12),
                    "sd_within": b["sd_within"] / max(m_["sd_within"], 1e-12),
                    "slope": b["slope"] / m_["slope"] if abs(m_["slope"]) > 1e-12 else np.nan,
                    "d_pos": med(b["d_pos"], m_["d_pos"]), "d_within": med(b["d_within"], m_["d_within"]),
                    "d": med(b["d"], m_["d"])}
    return out


# --------------------------------------------------------------------------
# measure (box)
# --------------------------------------------------------------------------
def measure_one(v, s, device, models_root):
    import torch
    import diagnose_sampling_tail as D
    import rq1_accuracy as RA
    from optbnn.bnn.nets.mlp import MLP
    from optbnn.utils import util
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    run_dir = os.path.join(models_root, f"antmaze_{v}_bnn_eval_{s}")
    cfg = D._load_run_config(run_dir)
    (val, _) = D._resolve_dataset(cfg, "val")
    assert "/noise/" not in val and "/reduction/" not in val, val
    X, y = util.load_pref_data(val, training_ratio=1.0)
    B, _, T, dd = X.shape
    obs = dd - 1
    mask = np.stack([X[:, 0, :, obs], X[:, 1, :, obs]]) > 0                       # [2, B, T]
    inp = np.stack([X[:, 0, :, :obs], X[:, 1, :, :obs]])                          # [2, B, T, obs]
    flat = torch.from_numpy(inp.reshape(-1, obs).astype(np.float32)).to(device)
    mflat = mask.reshape(-1)
    net = MLP(input_dim=obs, output_dim=1, hidden_dims=[cfg["width"]] * cfg["depth"], activation_fn="relu").to(device)
    net.eval()
    acc, n = np.zeros(flat.shape[0]), 0
    for cid in D._resolve_chain_ids(run_dir, None, None, cfg["num_chains"]):
        for w in D._load_chain_weights(run_dir, cid, device):
            with torch.no_grad():
                for p, a in zip(net.parameters(), w):
                    p.copy_(torch.from_numpy(a).to(device))
                o = net(flat).cpu().numpy().reshape(-1).astype(np.float64)
            acc += o - o[mflat].mean()                                            # centred per draw
            n += 1
    r_bnn = (acc / n).reshape(2, B, T)
    prior, _, live = D._live_prior(cfg, quiet=True)
    if not live:
        raise RuntimeError(f"{run_dir}: the live layout did not load")
    cell = prior.cell_of(inp.reshape(-1, obs)).reshape(2, B, T)
    mr_dir = RA.model_dir(models_root, "mr", "anchor", v, s)
    mr = Q.load_mlp_reward_model(mr_dir, device=device)
    mr.eval()
    with torch.no_grad():
        r_mr = mr(flat).cpu().numpy().reshape(2, B, T).astype(np.float64)
    speed = step_speed(inp[..., :2], mask)
    return {"r_bnn": r_bnn, "r_mr": r_mr, "cell": cell, "speed": speed, "mask": mask,
            "y1": np.asarray(y)[:, 0], "n_draws": n, "bnn_dir": run_dir, "mr_dir": mr_dir}


def cmd_measure(a):
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    for v in VARIANTS:
        for s in SEEDS:
            path = os.path.join(a.out, f"{v}_{s}.npz")
            if os.path.exists(path) and not a.force:
                continue
            try:
                rec = measure_one(v, s, a.device, a.models_root)
            except Exception as e:                           # noqa: BLE001  one model must not stop the run
                print(f"  {v} {s}: FAILED ({type(e).__name__}: {str(e)[:160]})")
                continue
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **{k: np.asarray(x) for k, x in rec.items()})
            os.replace(tmp, path)
            print(f"  {v} {s}: {rec['y1'].size} pairs, {rec['n_draws']} draws")
    return 0


# --------------------------------------------------------------------------
# analyse (here)
# --------------------------------------------------------------------------
def cmd_analyse(a, out=print):
    out("BNN H3 CALIBRATION, MECHANISM STAGE B3 (handoff §14.100; scoped hold exception; post hoc, descriptive)")
    out("BNN / MR on the same validation steps: positional (cell-mean) and within-cell parts of the per-step reward")
    for v in VARIANTS:
        rows, gap, accs = [], 0.0, []
        for s in SEEDS:
            p = os.path.join(a.inp, f"{v}_{s}.npz")
            if not os.path.exists(p):
                continue
            rec = dict(np.load(p, allow_pickle=True))
            st = seed_stats(rec)
            g = json.load(open(os.path.join(MEAS, f"antmaze_{v}_bnn_eval_{s}.json")))
            gap = max(gap, float(np.abs(st["bnn"]["d"] - np.asarray(g["d_mean"])).max()))
            rows.append([st["ratio"][k] for k in ("sd_pos", "sd_within", "slope", "d_pos", "d_within", "d")])
            accs.append([st[m][k] for m in ("bnn", "mr") for k in ("acc", "acc_pos", "acc_within")])
        if len(rows) != len(SEEDS):
            out(f"\n=== {v}: {len(rows)} of {len(SEEDS)} seeds -- not read")
            continue
        out(f"\n=== {v}")
        out(f"  check: the BNN's pooled per-step mean vs the gate's d_mean, max |difference| {gap:.2e}"
            + ("" if gap < 1e-3 else "  !! NOT THE SAME MODELS OR PAIRS -- nothing below is read"))
        if gap >= 1e-3:
            continue
        R, A = np.nanmedian(np.asarray(rows, float), 0), np.mean(np.asarray(accs, float), 0)
        out(f"  BNN / MR (median over seeds): sd positional {R[0]:.2f}, sd within-cell {R[1]:.2f}, "
            f"within-cell slope on speed {R[2]:.2f}")
        out(f"  pair logits, median |d_BNN| / |d_MR|: positional part {R[3]:.2f}, within-cell part {R[4]:.2f}, "
            f"whole {R[5]:.2f}")
        out(f"  accuracy on the human labels  BNN: whole {A[0]:.3f}, positional part {A[1]:.3f}, within-cell part {A[2]:.3f}"
            f"  |  MR: whole {A[3]:.3f}, positional {A[4]:.3f}, within-cell {A[5]:.3f}")
    return 0


def selftest():
    rng = np.random.default_rng(0)
    B, T = 40, 20
    mask = rng.random((2, B, T)) < 0.9
    mask[..., 0] = True
    cell = rng.integers(0, 6, size=(2, B, T))
    speed = rng.gamma(2.0, 0.2, size=(2, B, T))
    cellval = rng.normal(0, 1, 6)
    pos = cellval[cell]
    r_mr = pos + 2.0 * (speed - speed.mean())
    r_bnn = pos + 0.3 * 2.0 * (speed - speed.mean())               # the within-cell (speed) part shrunk to 0.3
    y1 = (pooled_d(r_mr, mask) > 0).astype(float)
    st = seed_stats({"r_bnn": r_bnn, "r_mr": r_mr, "cell": cell, "speed": speed, "mask": mask, "y1": y1})
    assert abs(st["ratio"]["slope"] - 0.3) < 0.05, st["ratio"]
    assert abs(st["ratio"]["sd_pos"] - 1.0) < 0.1 and st["ratio"]["sd_within"] < 0.4, st["ratio"]
    assert st["ratio"]["d_within"] < 0.45 and 0.7 < st["ratio"]["d_pos"] < 1.4, st["ratio"]
    assert st["mr"]["acc"] == 1.0
    # the parts add up; the positional part is constant within a cell
    p, w = split(r_mr, cell, mask)
    assert np.allclose((p + w)[mask], r_mr[mask])
    assert np.allclose(pooled_d(p, mask) + pooled_d(w, mask), pooled_d(r_mr, mask))
    # step speed: the last valid step repeats the one before it
    xy = np.array([[[0, 0], [3, 4], [3, 4], [9, 12]]], float)
    sp = step_speed(xy, np.array([[1, 1, 1, 0]]))               # step lengths 5, 0, 10; step 2 is the last valid
    assert np.allclose(sp, [[5, 0, 0, 10]]), sp
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure")
    m.add_argument("--out", default=OUT)
    m.add_argument("--models-root", default=os.path.join(ROOT, "exp", "reward_learning"))
    m.add_argument("--device", default="cpu")
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--force", action="store_true")
    n = sub.add_parser("analyse")
    n.add_argument("--inp", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "measure":
        return cmd_measure(a)
    if a.cmd == "analyse":
        return cmd_analyse(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
