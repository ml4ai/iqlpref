#!/usr/bin/env python
"""final_positions.py -- where do evaluation episodes end?  (handoff §13.14.14; DESCRIPTIVE)

Reads final_positions.csv from every run of a sweep (iql_eval log_final_positions)
and reports, over each run's last 10 evaluations (the scoring window): the success
rate, and for FAILED episodes the final cell distribution and its BFS distance to
the evaluation goal.  The loitering hypothesis of §13.14.13 predicts that BNN CVaR's
failures without terminals end in cell (6, 5), one cell short of the goal.

    python results/final_positions.py <sweep_id> [<sweep_id> ...]
    python results/final_positions.py --selftest
"""

import collections
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "gp_reward-priors"))
sys.path.insert(0, HERE)

from goal_structure import bfs_distance, cells_of                  # noqa: E402

ENTITY, PROJECT = "champlin-university-of-arizona", "IQL-pref"
LAST = 10
# The hardcoded MEDIUM layout matches the box's live maze (26 free cells, goal cell
# (6, 6), checked 2026-10-19); the hardcoded LARGE layout does NOT (§13.14.11).
GOAL_CELL = {"medium": (6, 6)}


def layout(size):
    from optbnn.gp.maze_layouts import ANTMAZE_MEDIUM_MAP, maze_map_to_arrays
    if size != "medium":
        raise SystemExit("only the medium layout is checked against the live maze")
    return maze_map_to_arrays(ANTMAZE_MEDIUM_MAP)


def summarise(rows, free, scaling, offset, goal):
    """rows: dicts with eval_step, x, y, success.  Uses the last LAST eval steps."""
    steps = sorted({int(r["eval_step"]) for r in rows})[-LAST:]
    keep = [r for r in rows if int(r["eval_step"]) in steps]
    dist = bfs_distance(free, goal)
    succ = np.array([int(r["success"]) for r in keep])
    xy = np.array([[float(r["x"]), float(r["y"])] for r in keep])
    rr, cc = cells_of(xy, scaling, offset)
    fail = succ == 0
    cells = collections.Counter(zip(rr[fail].tolist(), cc[fail].tolist()))
    d = np.array([dist.get(c, -1) for c in zip(rr[fail].tolist(), cc[fail].tolist())])
    return {"n": int(len(keep)), "success": float(succ.mean()) if len(keep) else float("nan"),
            "n_fail": int(fail.sum()), "fail_cells": cells.most_common(5),
            "fail_dist": collections.Counter(d.tolist())}


def main():
    if "--selftest" in sys.argv:
        free, scaling, offset = layout("medium")
        row_off, col_off = offset
        c2xy = lambda r, c: ((c - col_off) * scaling, (r - row_off) * scaling)
        rows = []
        for step in range(1, 13):                         # 12 eval steps; the last 10 are read
            for e in range(10):
                if e < 4:
                    x, y, s = *c2xy(6, 6), 1              # successes at the goal
                elif e < 9:
                    x, y, s = *c2xy(6, 5), 0              # failures one cell short
                else:
                    x, y, s = *c2xy(1, 1), 0              # one failure back at the start
                rows.append({"eval_step": step * 5000, "x": x + 0.3, "y": y - 0.3, "success": s})
        out = summarise(rows, free, scaling, offset, GOAL_CELL["medium"])
        assert out["n"] == 100 and abs(out["success"] - 0.4) < 1e-12 and out["n_fail"] == 60
        assert out["fail_cells"][0] == ((6, 5), 50) and out["fail_dist"][1] == 50
        print("selftest OK")
        return 0
    import wandb
    api = wandb.Api(timeout=120)
    free, scaling, offset = layout("medium")
    for sid in sys.argv[1:]:
        sw = api.sweep(f"{ENTITY}/{PROJECT}/{sid}")
        allrows, n_runs = [], 0
        print(f"=== {sid}")
        for r in sorted(sw.runs, key=lambda r: r.config.get("seed", 0)):
            try:
                f = r.file("final_positions.csv").download(replace=True, root=f"/tmp/fp_{sid}_{r.id}")
                rows = list(csv.DictReader(open(f.name)))
            except Exception as e:
                print(f"  seed {r.config.get('seed')}: no final_positions.csv ({type(e).__name__})")
                continue
            n_runs += 1
            allrows += [dict(x, seed=r.config.get("seed")) for x in rows]
            s = summarise(rows, free, scaling, offset, GOAL_CELL["medium"])
            print(f"  seed {r.config.get('seed'):2d}  success {s['success']:.3f}  failed {s['n_fail']:4d}  "
                  f"failures end in {s['fail_cells'][:3]}")
        if n_runs:
            # pooled over runs: each run's own last-10 window, then concatenated
            pooled = {"n": 0, "n_fail": 0, "cells": collections.Counter(), "dist": collections.Counter(), "succ": []}
            for seed in sorted({x["seed"] for x in allrows}):
                s = summarise([x for x in allrows if x["seed"] == seed], free, scaling, offset, GOAL_CELL["medium"])
                pooled["n"] += s["n"]; pooled["n_fail"] += s["n_fail"]
                pooled["cells"].update(dict(s["fail_cells"])); pooled["dist"].update(s["fail_dist"])
                pooled["succ"].append(s["success"])
            fd = pooled["dist"]
            tot = sum(fd.values()) or 1
            print(f"  POOLED over {n_runs} runs: success {np.mean(pooled['succ']):.3f}; failed episodes "
                  f"{pooled['n_fail']} of {pooled['n']}; BFS distance of failure ends: "
                  + ", ".join(f"{k}: {v / tot:.2f}" for k, v in sorted(fd.items()))
                  + f"; top cells {pooled['cells'].most_common(4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
