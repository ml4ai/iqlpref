#!/usr/bin/env python
"""extract_prelim.py -- the proposal's Preliminary Results data, frozen (handoff §14.135).

The ONLY step of the paper pipeline that touches W&B (read-only).  It runs the registered
collectors, unchanged, and writes one snapshot that render_prelim.py reads with the network
off:
    standard   exp_readout.collect_exp(..., "sparsity", "standard")   the ladders with terminals
    candidate  exp_readout.collect_exp(..., "sparsity", "candidate")  the ladders without
    anchor     eval_readout.collect                                     full data, with terminals
    ctrl       control_readout.collect_controls                         every control cell
    accuracy   exp/rq1_accuracy.json, through rq1_readout.load_accuracy with its cross-check
               against the training runs' logged eval_acc_reloaded (sparsity levels and the
               anchor only)
These are exactly rq1_readout's inputs on the sparsity ladder, plus the anchor's other cells
(the task reward at index 1 on medium-play is task_reward_r1).  The BNN/CVaR hold applies:
the collectors do not fetch the held families (results/hold.py).

One row per run (family, dataset, protocol, level, seed, normalization index, trajectory
cut, W&B ids, last-10 / last-20 / max score), so every summary can be recomputed; each cell's
audit and completeness are kept beside them.  The file records the extraction time, the W&B
project, and the commits of iqlpref and gp_reward-priors.  Commit it in iqlpref.

    python results/paper/extract_prelim.py [--out results/paper/snapshot_prelim.json]
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
ROOT = os.path.dirname(RESULTS)
sys.path.insert(0, RESULTS)
sys.path.insert(0, ROOT)

import control_readout as C                                # noqa: E402
import eval_readout as R                                   # noqa: E402
import exp_readout as X                                    # noqa: E402
import hold                                                # noqa: E402
import rq1_readout as Q                                    # noqa: E402

SNAPSHOT = os.path.join(HERE, "snapshot_prelim.json")
AXIS = "sparsity"
ACC_LEVELS = set(X.LEVELS) | {X.FULL}


def _git(path):
    sha = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return {"commit": sha, "dirty": dirty}


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def cell_record(group, key, cell):
    """The cell's own fields (no scores: those are in the run rows)."""
    return {"group": group, "key": list(key), "sweep": cell["sweep"], "index": cell["index"],
            "name": cell.get("name"), "audit": cell["audit"], "incomplete": cell["incomplete"]}


def run_rows(group, key, cell):
    """One row per run of the cell, with its scores when it has them."""
    rows = []
    for s, m in sorted(cell.get("runs", {}).items(), key=lambda kv: (kv[0] is None, kv[0])):
        rows.append({"group": group, "key": list(key), "seed": s, "sweep": cell["sweep"], "run": m["id"],
                     "state": m["state"], "index": m["normalize_reward"], "cut": m["return_range_split"],
                     "drop_terminals": m["drop_terminals"], "score": cell["scores"].get(s),
                     "robust": cell["robust"].get(s), "max": cell["max"].get(s)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=SNAPSHOT)
    hold.parse()
    a = ap.parse_args()
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    winners_path = os.path.join(ROOT, "phase2_winners.json")
    gwin_path = os.path.join(ROOT, P.GRID_WINNERS)
    winners, gwin = json.load(open(winners_path)), json.load(open(gwin_path))
    api = wandb.Api(timeout=120)
    groups, notes = {}, []
    groups["standard"], n = X.collect_exp(api, P, winners, AXIS, "standard")
    notes += n
    groups["candidate"], n = X.collect_exp(api, P, gwin, AXIS, "candidate")
    notes += n
    groups["anchor"], n = R.collect(api, P, winners)
    notes += n
    groups["ctrl"], n = C.collect_controls(api, P, winners, gwin)
    notes += n
    acc, anotes = Q.load_accuracy(Q.ACC_PATH, Q.logged_accuracies())
    raw = json.load(open(Q.ACC_PATH))
    inv = {v: k for k, v in Q.ACC_FAM.items()}
    accuracy = []
    for k, rec in sorted(raw.items()):
        fam, lv, vu, s = k.split("|")
        key = (inv[fam], vu.replace("_", "-"), X.FULL if lv == "anchor" else lv, int(s))
        if key[2] not in ACC_LEVELS:
            continue
        val = acc.get(key)
        accuracy.append({"family": key[0], "dataset": key[1], "level": key[2], "seed": key[3],
                         "correct": rec["correct"], "n": rec["n"],
                         "value": val if isinstance(val, float) else None,
                         "reason": None if isinstance(val, float) else (val or "not measured")})
    cells, runs = [], []
    for g, cs in groups.items():
        for key, cell in sorted(cs.items(), key=lambda kv: str(kv[0])):
            cells.append(cell_record(g, key, cell))
            runs += run_rows(g, key, cell)
    keep = [x for x in notes + anotes if hold.INCLUDE or not any(t in x for t in ("BNN", "ENS_CVAR", "bnn"))]
    snap = {"meta": {"what": "proposal Preliminary Results floats (handoff §14.135): RQ1 sparsity inputs, "
                             "anchor, controls, accuracy",
                     "extracted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                     "wandb": f"{P.ENTITY}/{P.IQL_PROJECT}", "axis": AXIS,
                     "iqlpref": _git(ROOT), "gp_reward_priors": _git(os.path.join(ROOT, "gp_reward-priors")),
                     "inputs": {os.path.relpath(p, ROOT): _sha256(p) for p in (winners_path, gwin_path, Q.ACC_PATH)},
                     "hold_include": hold.INCLUDE, "seeds": list(R.SEEDS), "notes": keep},
            "winners": winners, "grid_winners": gwin, "cells": cells, "runs": runs, "accuracy": accuracy}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(snap, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"wrote {os.path.relpath(a.out, ROOT)}: {len(cells)} cells, {len(runs)} runs, {len(accuracy)} accuracy "
          f"records ({sum(r['value'] is not None for r in accuracy)} read)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
