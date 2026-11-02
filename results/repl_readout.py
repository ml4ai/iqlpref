#!/usr/bin/env python
"""repl_readout.py -- the replication's readouts (gp_reward-priors HANDOFF §14.83).

  status   every replication grid and cell: grids' run states and their winner (if scored
           into results/repl_grid_winners.json by `replication.py winners`); standard-protocol
           cells' own mean ± sd when complete; candidate-protocol cells' run STATES ONLY --
           §14.5 reads a family's candidate arm only in its registered reading, as in Study 1.
  rq1      §14.5's REGISTERED reading, on the replication: rq1_readout's own functions (units,
           tests, verdicts, report) run unchanged on the replication's cells and accuracies.

The units: rq1_readout pairs by seed over SEEDS = 1-10.  The replication's seeds are 101-110,
so each cell's scores (and each accuracy) are re-keyed seed s -> unit s - 100 at collection,
which changes no pairing: unit k is seed 100 + k in every cell and every accuracy alike.  A run
at any other seed is a config-audit failure, so the cell is not read.  What rq1_readout reports
"beside, no test" (the standard-arm trivial check, the motion reward) is not re-run (§14.83 decision
5) and prints as NOT READ.

    python results/repl_readout.py status
    python results/repl_readout.py rq1 [--axis sparsity|noise] [--acc exp/rq1_accuracy_repl.json]
    python results/repl_readout.py --selftest
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import eval_readout as R                                   # noqa: E402
import hold                                                # noqa: E402
import rq1_readout as Q                                    # noqa: E402

OFFSET = 100
ACC_PATH = os.path.join(ROOT, "exp", "rq1_accuracy_repl.json")


def _vd(v):
    return v.replace("_", "-")


def build_cell(sw, name, entry, index, runs):
    """One cell, scores keyed by UNIT (seed - 100).  runs: [(seed, state, config, curve or Exception)]."""
    import replication as Y
    cell = {"sweep": sw, "index": index, "scores": {}, "robust": {}, "max": {},
            "incomplete": [], "audit": [], "name": name}
    oracle = entry["fam"] not in Y.MODEL_FAM
    seen = set()
    for s, state, cfg, curve in runs:
        if s not in Y.EVAL_SEEDS:
            cell["audit"].append(f"seed {s} outside 101-110")
            continue
        if s in seen:
            cell["audit"].append(f"seed {s} appears twice")
        seen.add(s)
        cell["audit"] += [f"seed {s}: {x}" for x in R.audit_run(cfg, name, s, entry["fixed"], index, oracle)]
        if state != "finished":
            cell["incomplete"].append(f"seed {s} {state}")
            continue
        if isinstance(curve, Exception):
            cell["incomplete"].append(f"seed {s}: {curve}")
            continue
        u = s - OFFSET
        cell["scores"][u], cell["robust"][u], cell["max"][u] = curve
    missing = [s for s in Y.EVAL_SEEDS if s - OFFSET not in cell["scores"] and
               not any(x.startswith(f"seed {s}") for x in cell["incomplete"])]
    if missing:
        cell["incomplete"].append(f"seeds not started: {missing}")
    return cell


def collect(api, P, Y):
    """({cell name: cell}, notes) for every replication cell sweep at its expected index."""
    import iql_score
    sp, E = P.spec(), P._conditions()
    path = os.path.join(ROOT, Y.WINNERS)
    gwin = json.load(open(path)) if os.path.exists(path) else {}
    cs = Y.cell_spec(sp, E)
    fixed_of = {n: c["fixed"] for n, c in cs.items()}
    cells, notes = {}, []
    for sw in P.iql_sweeps(api):
        params = sw.config.get("parameters")
        hits = P.match_entry(params, fixed_of)
        if len(hits) != 1:
            continue
        name, flat = hits[0], P.flat(params)
        c = cs[name]
        if flat.get("seed") != Y.EVAL_SEEDS:
            continue                                   # a grid (seed 100), or the leak control (1-10)
        want = c["index"] if c["grid"] is None else gwin.get(c["grid"])
        if flat.get("normalize_reward") != want:
            notes.append(f"{sw.id}: {name} at index {flat.get('normalize_reward')}, expected {want} -- ignored")
            continue
        if c["fam"] in Y.METHOD and not hold.shown(Y.METHOD[c["fam"]]):
            continue
        runs = []
        for r in sw.runs:
            cfg = r.config or {}
            curve = None
            if r.state == "finished":
                try:
                    _, cv = iql_score.run_curve(r)
                    curve = (iql_score.score_curve(cv, n=iql_score.LAST_N),
                             iql_score.score_curve(cv, n=iql_score.ROBUST_N),
                             iql_score.score_curve(cv, statistic="max"))
                except (iql_score.IncompleteRun, ValueError) as e:
                    curve = e
            runs.append((cfg.get("seed"), r.state, cfg, curve))
        cell = build_cell(sw.id, name, c, want, runs)
        if sw.config.get("program") != P.DEFAULT_PROGRAM:
            cell["audit"].append(f"program {sw.config.get('program')!r}")
        if name in cells:
            notes.append(f"{name}: two sweeps ({cells[name]['sweep']}, {sw.id}) -- cell NOT read")
            cells[name]["audit"].append(f"second sweep {sw.id}")
            continue
        cells[name] = cell
    return cells, notes


def arrange(cells, Y):
    """rq1_readout's shapes: std / cand {(variant, METHOD, level)}, full {(variant, m)}, ctrl."""
    std, cand, full, ctrl = {}, {}, {}, {}
    for v in Y.P.VARIANTS:
        vd = _vd(v)
        for fam, m in Y.METHOD.items():
            for lv in ("full",) + tuple(Q.X.LEVELS) + tuple(Q.X.NOISE_LEVELS):
                key = (vd, m, Q.FULL if lv == "full" else lv)
                std[key] = cells.get(f"{Y.REPL_DIR}/std_{lv}_antmaze_{v}_{fam}")
                cand[key] = cells.get(f"{Y.REPL_DIR}/cand_{lv}_antmaze_{v}_{fam}")
            full[(vd, m)] = cand[(vd, m, Q.FULL)]
        full[(vd, "task_reward")] = cells.get(f"{Y.REPL_DIR}/cand_full_antmaze_{v}_task")
        ctrl[(vd, "trivial_zero_noterm")] = cells.get(f"{Y.REPL_DIR}/cand_zero_antmaze_{v}")
        ctrl[(vd, "trivial_mean_noterm")] = cells.get(f"{Y.REPL_DIR}/cand_mean_antmaze_{v}")
    return std, cand, full, ctrl


def rekey_accuracy(acc):
    """rq1_readout.load_accuracy's {(FAM, variant, level, seed): value} -> units; other seeds dropped."""
    return {(f, v, lv, s - OFFSET): x for (f, v, lv, s), x in acc.items() if 101 <= s <= 110}


def cmd_rq1(a, P, Y):
    import wandb
    cells, notes = collect(wandb.Api(timeout=120), P, Y)
    std, cand, full, ctrl = arrange(cells, Y)
    acc, anotes = Q.load_accuracy(a.acc, Q.logged_accuracies() if os.path.exists(a.acc) else None)
    res, use, uwhy, beside = Q.run(a.axis, std, cand, full, ctrl, rekey_accuracy(acc))
    print("#" * 100)
    print("# THE REPLICATION (gp_reward-priors HANDOFF §14.83): new reward models (selected and trained at seeds")
    print("# 100-110), return_range_split 'length' in both protocols, seeds 101-110 shown as units 1-10.")
    print("#" * 100)
    keep = [n for n in notes + anotes if hold.INCLUDE or not any(x in n for x in ("BNN", "ENS_CVAR", "bnn"))]
    keep.append("BESIDE items are not re-run in the replication (§14.83 decision 5): they print as NOT READ")
    Q.report(a.axis, res, use, uwhy, beside, keep)
    return 0


def cmd_status(P, Y):
    import collections
    import wandb
    api = wandb.Api(timeout=120)
    sp, E = P.spec(), P._conditions()
    path = os.path.join(ROOT, Y.WINNERS)
    gwin = json.load(open(path)) if os.path.exists(path) else {}
    gs = Y.grid_spec(sp, E)
    grids = Y.find_grids(api, sp, E)
    print("=" * 100)
    print("REPLICATION STATUS (HANDOFF §14.83).  Candidate cells: run states only (§14.5).")
    print("=" * 100)
    print("\nGRIDS (seed 100)")
    for name in gs:
        sws = grids.get(name, [])
        st = ", ".join(f"{s.id} {s.state} " + str(dict(collections.Counter(r.state for r in s.runs))) for s in sws)
        print(f"  {name:48s} {st or 'no sweep'}" + (f"   winner {gwin[name]}" if name in gwin else ""))
    cells, notes = collect(api, P, Y)
    for prot in Y.PROTOCOLS:
        print(f"\n{'STANDARD: own mean ± sd when complete' if prot == 'std' else 'CANDIDATE: states only'}")
        for name, c in Y.cell_spec(sp, E).items():
            if c["prot"] != prot or name not in cells:
                continue
            cell = cells[name]
            why = R.cell_ready(cell)
            if prot == "std" and not why:
                x = list(cell["scores"].values())
                print(f"  {name:56s} idx {cell['index']}  {statistics.mean(x):.3f} ± {statistics.stdev(x):.3f}")
            else:
                print(f"  {name:56s} idx {cell['index']}  " +
                      (why or f"COMPLETE ({len(cell['scores'])}/10)"))
    for n in notes:
        print(f"  note: {n}")
    return 0


def selftest():
    import replication as Y
    entry = {"fam": "mr", "fixed": {"config_path": "c", "reward_model_root": "/m/antmaze_x_mr_eval",
                                    "query_length": 1, "return_range_split": "length", "drop_terminals": True}}
    cfg = lambda s: dict(entry["fixed"], seed=s, normalize_reward=5, reward_model_path=f"/m/antmaze_x_mr_eval_{s}",
                         gauge_reward="max0")
    runs = [(s, "finished", cfg(s), (0.1 * (s - 100), 0.0, 0.0)) for s in Y.EVAL_SEEDS]
    c = build_cell("sw", "n", entry, 5, runs)
    assert R.cell_ready(c) is None and sorted(c["scores"]) == list(range(1, 11)), c
    assert abs(c["scores"][3] - 0.3) < 1e-12                       # unit 3 is seed 103
    # a seed-1 run (Study 1's lineage) or a model from the wrong seed is an audit failure
    bad = build_cell("sw", "n", entry, 5, runs[:-1] + [(1, "finished", cfg(1), (0.5, 0, 0))])
    assert R.cell_ready(bad).startswith("CONFIG AUDIT FAILED"), R.cell_ready(bad)
    wrong = dict(cfg(104), reward_model_path="/m/antmaze_x_mr_eval_4")
    bad = build_cell("sw", "n", entry, 5, runs[:3] + [(104, "finished", wrong, (0.4, 0, 0))] + runs[4:])
    assert "reward_model_path" in R.cell_ready(bad), R.cell_ready(bad)
    inc = build_cell("sw", "n", entry, 5, runs[:9] + [(110, "running", cfg(110), None)])
    assert R.cell_ready(inc).startswith("INCOMPLETE"), R.cell_ready(inc)
    # accuracies: seeds 101-110 -> units 1-10; Study 1's seeds are dropped
    acc = {("MR", "large-play", "r16", 103): 0.7, ("MR", "large-play", "r16", 3): 0.9}
    assert rekey_accuracy(acc) == {("MR", "large-play", "r16", 3): 0.7}
    # arrange: every key rq1_readout.run reads exists, and the full level is shared by name
    cells = {f"{Y.REPL_DIR}/cand_full_antmaze_large_play_pt": c, f"{Y.REPL_DIR}/cand_zero_antmaze_large_play": c}
    std, cand, full, ctrl = arrange(cells, Y)
    assert len(std) == len(cand) == 4 * 4 * 10 and cand[("large-play", "PT", Q.FULL)] is c
    assert full[("large-play", "PT")] is c and ctrl[("large-play", "trivial_zero_noterm")] is c
    assert full[("large-play", "MR")] is None and ("medium-play", "task_reward") in full
    # the registered functions run end to end on re-keyed cells (a synthetic, fully sensitive family)
    import random
    rng = random.Random(0)
    def mk(vals):
        return {"scores": dict(zip(range(1, 11), vals)), "robust": {}, "max": {}, "incomplete": [], "audit": [],
                "sweep": "s", "index": 3, "name": "n"}
    std, cand, full, ctrl, acc = {}, {}, {}, {}, {}
    lv_all = (Q.FULL,) + tuple(Q.X.LEVELS)
    for v in Q.VARIANTS:
        for i, lv in enumerate(lv_all):                      # full is the best level
            drop = 0.08 * i
            for m in Q.FAMILIES:
                cand[(v, m, lv)] = mk([0.6 - drop + rng.uniform(-.01, .01) for _ in range(10)])
                std[(v, m, lv)] = mk([0.6 + rng.uniform(-.01, .01) for _ in range(10)])
                for u in range(1, 11):
                    acc[(m, v, lv, u)] = 0.9 - 0.05 * i + rng.uniform(-.005, .005)
        for m in Q.FAMILIES:
            full[(v, m)] = cand[(v, m, Q.FULL)]
        full[(v, "task_reward")] = mk([0.7] * 10)
        ctrl[(v, "trivial_zero_noterm")] = mk([0.0] * 10)
        ctrl[(v, "trivial_mean_noterm")] = mk([0.01] * 10)
    res, use, uwhy, beside = Q.run("sparsity", std, cand, full, ctrl, acc)
    assert uwhy is None and all(use[v]["usable"] for v in Q.VARIANTS)
    shown = [f for f in Q.FAMILIES if hold.shown(f)]
    assert shown and all(res[f]["verdict"] == "SENSITIVE" for f in shown), {f: res[f].get("verdict") for f in res}
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("status", "rq1"))
    ap.add_argument("--axis", choices=tuple(Q.AXES), default="sparsity")
    ap.add_argument("--acc", default=ACC_PATH)
    ap.add_argument("--selftest", action="store_true")
    hold.parse()
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    import replication as Y
    sys.argv = _argv
    if a.cmd == "status":
        return cmd_status(P, Y)
    if a.cmd == "rq1":
        return cmd_rq1(a, P, Y)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
