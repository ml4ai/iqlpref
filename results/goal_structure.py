#!/usr/bin/env python
"""goal_structure.py -- does the goal reach IQL through the reward, or through the data?
(handoff §13.14.10; DESCRIPTIVE, no verdict).  Run ON THE BOX from ~/iqlpref (it needs
D4RL, the reward models and the BNN label cache).

The goal is not in the antmaze observation (29 dims: torso x, y, orientation,
joints).  Without terminals, any goal-directedness must come from:

  1. REWARD MAPS.  Each learned reward, exactly as IQL is trained on it (iql_eval's
     own labelling functions; BNN labels from the cache), averaged over the D4RL
     dataset's transitions in each maze cell.  Spearman rho between that map and
     the shortest-path (BFS) distance from the cell to the evaluation goal's cell:
     NEGATIVE = the reward rises toward the goal.  Shift and positive scale do not
     change rho, so gauge and normalization are irrelevant here.  References: the
     task reward (0/1 at the goal) and the path-length heuristic.
  2. GOAL COVERAGE.  How often the data-collecting controller was commanded to
     the evaluation goal: infos/goal per transition, mapped to its cell.

    python results/goal_structure.py --variants "medium_play medium_diverse large_play large_diverse" --seeds "1 2 3" > exp/goal_structure.txt 2>&1
    python results/goal_structure.py --selftest
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors"))

MIN_PER_CELL = 50                  # a cell's mean is reported only with this many transitions
MODELS = os.path.join("exp", "reward_learning")


# --------------------------------------------------------------------------
# Maze geometry (pure; self-tested)
# --------------------------------------------------------------------------
def cells_of(xy, scaling, offset):
    row_off, col_off = offset
    col = np.rint(xy[:, 0] / scaling + col_off).astype(np.int64)
    row = np.rint(xy[:, 1] / scaling + row_off).astype(np.int64)
    return row, col


def bfs_distance(free_mask, start):
    """{(row, col): steps} over 4-connected free cells from start."""
    free_mask = np.asarray(free_mask, dtype=bool)
    dist = {tuple(start): 0}
    q = deque([tuple(start)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if (0 <= n[0] < free_mask.shape[0] and 0 <= n[1] < free_mask.shape[1]
                    and free_mask[n] and n not in dist):
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


def cell_means(values, rows, cols, min_n=MIN_PER_CELL):
    """{(row, col): (mean, n)} for cells with at least min_n entries."""
    out = {}
    key = rows * 10_000 + cols
    for k in np.unique(key):
        m = key == k
        if m.sum() >= min_n:
            out[(int(k // 10_000), int(k % 10_000))] = (float(values[m].mean()), int(m.sum()))
    return out


def goal_rho(means, dist):
    """Spearman rho between cell-mean reward and BFS distance to the goal, over cells
    that have both; and the goal cell's reward rank (1 = highest)."""
    from scipy.stats import spearmanr
    cells = [c for c in means if c in dist]
    if len(cells) < 5:
        return None, None, len(cells)
    r = np.array([means[c][0] for c in cells])
    d = np.array([dist[c] for c in cells])
    rho = float(spearmanr(r, d)[0])
    goal = [c for c in cells if dist[c] == 0]
    rank = None
    if goal:
        rank = int((r > means[goal[0]][0]).sum()) + 1
    return rho, rank, len(cells)


# --------------------------------------------------------------------------
# The box part
# --------------------------------------------------------------------------
def rewards_for(kind, env, variant, seed, device):
    """Per-transition reward exactly as iql_eval builds it (before gauge and
    normalization, which cannot change rho), and the matching dataset."""
    import iql_eval as Q                                   # algorithms/offline/iql_eval.py
    root = os.path.join(MODELS, f"antmaze_{variant}_{{}}_eval_{seed}")
    if kind == "MR":
        m = Q.load_mlp_reward_model(root.format("mr"), device=device)
        return Q.qlearning_dataset_mr(env, m)
    if kind == "PT":
        m = Q.load_pt_reward_model(root.format("pt"), device=device)
        return Q.qlearning_dataset_pt(env, m, 100)
    if kind in ("BNN mean", "BNN CVaR"):
        return Q.qlearning_dataset_bnn(env, root.format("bnn"), alpha=0.0 if kind == "BNN mean" else 0.95,
                                       n_samples=-1, device=device, centre_draws=True)
    raise ValueError(kind)


def run(a):
    import d4rl  # noqa: F401
    import gym
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    from optbnn.gp.maze_layouts import extract_maze_from_env
    for v in a.variants.split():
        name = f"antmaze-{v.replace('_', '-')}-v2"
        env = gym.make(name)
        free, scaling, offset = extract_maze_from_env(name)
        u = env.unwrapped
        goal_src = next((a_ for a_ in ("target_goal", "_target_goal", "_goal") if hasattr(u, a_)), None)
        if goal_src is not None:
            goal_xy = np.asarray(getattr(u, goal_src), dtype=float).reshape(1, -1)[:, :2]
            gr, gc = cells_of(goal_xy, scaling, offset)
        else:                                  # the maze map's goal marker
            mm = next(getattr(u, a_) for a_ in ("_maze_map", "maze_map") if hasattr(u, a_))
            hits = [(i, j) for i, row in enumerate(mm) for j, x in enumerate(row) if x == "g"]
            if len(hits) != 1:
                raise SystemExit(f"{name}: no target_goal attribute and {len(hits)} 'g' markers")
            goal_src, (gr, gc) = "maze map 'g'", (np.array([hits[0][0]]), np.array([hits[0][1]]))
            goal_xy = np.array([[(gc[0] - offset[1]) * scaling, (gr[0] - offset[0]) * scaling]])
        dist = bfs_distance(free, (int(gr[0]), int(gc[0])))
        if dist.get((int(gr[0]), int(gc[0]))) != 0 or not free[int(gr[0]), int(gc[0])]:
            raise SystemExit(f"{name}: the goal cell ({gr[0]}, {gc[0]}) is not a free cell")
        print(f"\n=== {name}: evaluation goal {goal_xy[0].round(2).tolist()} (from {goal_src}) -> cell "
              f"({gr[0]}, {gc[0]}); {int(free.sum())} free cells; max BFS distance {max(dist.values())}")

        # ---- 2. goal coverage -------------------------------------------------
        raw = env.get_dataset()
        g = np.asarray(raw["infos/goal"], dtype=float)
        rr, cc = cells_of(g, scaling, offset)
        gd = np.array([dist.get((int(r), int(c)), -1) for r, c in zip(rr, cc)])
        n = len(gd)
        print(f"  goal coverage: commanded goal in the evaluation goal's cell for "
              f"{(gd == 0).mean():.3f} of {n:,} transitions; within 1 cell {(np.isin(gd, (0, 1))).mean():.3f}; "
              f"{len(set(zip(rr.tolist(), cc.tolist())))} distinct commanded cells; "
              f"commanded-goal BFS distance to the evaluation goal: median {np.median(gd[gd >= 0]):.0f}")
        xr, xc = cells_of(np.asarray(raw["observations"])[:, :2], scaling, offset)
        occ = cell_means(np.ones(n), xr, xc, 1)
        print(f"  state coverage: {len(occ)} cells visited; transitions in the goal cell "
              f"{occ.get((int(gr[0]), int(gc[0])), (0, 0))[1]:,}")

        # ---- 1. reward maps ----------------------------------------------------
        refs = {"task reward (0/1)": np.asarray(raw["rewards"], dtype=float),
                "path-length heuristic": np.r_[np.linalg.norm(
                    np.diff(np.asarray(raw["observations"])[:, :2], axis=0), axis=1), 0.0]}
        for lab, val in refs.items():
            rho, rank, k = goal_rho(cell_means(val, xr, xc), dist)
            print(f"  {lab:24s}            rho {rho:+.3f}   goal cell's reward rank {rank} of {k}")
        for kind in a.kinds.split(","):
            kind = kind.strip()
            for s in [int(x) for x in a.seeds.split()]:
                try:
                    ds = rewards_for(kind, env, v, s, a.device)
                except Exception as e:                       # report, do not stop the sweep
                    print(f"  {kind:9s} seed {s:2d}  FAILED: {type(e).__name__}: {e}")
                    continue
                r_, c_ = cells_of(np.asarray(ds["observations"])[:, :2], scaling, offset)
                rho, rank, k = goal_rho(cell_means(np.asarray(ds["rewards"], dtype=float), r_, c_), dist)
                print(f"  {kind:9s} seed {s:2d}   {len(ds['rewards']):,} transitions   rho {rho:+.3f}   "
                      f"goal cell's reward rank {rank} of {k}")
    return 0


def selftest():
    from optbnn.gp.maze_layouts import ANTMAZE_LARGE_MAP, maze_map_to_arrays
    free, scaling, offset = maze_map_to_arrays(ANTMAZE_LARGE_MAP)
    goal = (7, 6)
    dist = bfs_distance(free, goal)
    assert dist[goal] == 0 and dist[(7, 5)] == 1 and dist[(7, 7)] == 1
    assert all(free[c] for c in dist) and len(dist) == int(free.sum())     # connected maze
    assert (6, 6) not in dist                                                # a wall
    # world <-> cell round trip at every free cell's centre
    row_off, col_off = offset
    rc = np.array(sorted(dist))
    xy = np.c_[(rc[:, 1] - col_off) * scaling, (rc[:, 0] - row_off) * scaling]
    r, c = cells_of(xy + np.random.default_rng(0).uniform(-1.9, 1.9, xy.shape), scaling, offset)
    assert (r == rc[:, 0]).all() and (c == rc[:, 1]).all()
    # a reward that falls with distance gives rho = -1 and puts the goal first
    rows, cols, vals = [], [], []
    for (ri, ci), d in dist.items():
        rows += [ri] * 60
        cols += [ci] * 60
        vals += [10.0 - d] * 60
    m = cell_means(np.array(vals), np.array(rows), np.array(cols))
    rho, rank, k = goal_rho(m, dist)
    assert abs(rho + 1) < 1e-12 and rank == 1 and k == len(dist)
    # sparse cells are dropped
    assert cell_means(np.ones(10), np.zeros(10, int), np.zeros(10, int)) == {}
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="medium_play medium_diverse large_play large_diverse")
    ap.add_argument("--seeds", default="1 2 3")
    ap.add_argument("--kinds", default="MR,PT,BNN mean,BNN CVaR")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    return selftest() if a.selftest else run(a)


if __name__ == "__main__":
    sys.exit(main())
