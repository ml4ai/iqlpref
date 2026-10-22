#!/usr/bin/env python
"""norm_inventory.py -- the normalization inventory (handoff §14.2 item 5, §14.23).  DESCRIPTIVE.

For every uncovered family and variant: the seed-0 normalization grid WITH the
dataset's terminals (stage 4) and WITHOUT them (the no-terminals grids, §13.14.5),
each scored by the stage-4 rule (iql_score.select_normalization: last-10, argmax,
ties to the lowest index), and its status: done, running (k/n scored), or not run.
Then, per family, whether the selected index moves when the terminals go.

Families: the task reward (indices 0-3), MR, PT, NMR, the ensemble's mean, and the
motion reward (no with-terminals grid: its with-terminals index was fixed at 3 in
advance, §13.14).  PT's pre-fix grids (§14.10) are listed apart, as the time-only
reward.  Covered families (§14.1) are never fetched.

    python results/norm_inventory.py [--selftest]
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import hold                                                # noqa: E402

VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
FAMILIES = ("task_reward", "MR", "PT", "NMR", "ENS_MEAN", "heuristic")


def grid_status(runs, grid):
    """(status, winner or None, {index: score}) for one grid's score_run dicts."""
    import iql_score
    by = {r["normalize_reward"]: r["score"] for r in runs if r["score"] is not None}
    try:
        win, _ = iql_score.select_normalization(runs, grid)
        return "done", win, by
    except ValueError:
        return f"running ({len(by)}/{len(grid)} scored)", None, by


def fmt_row(grid, by, win):
    return " ".join((f"[{by[i]:.3f}]" if i == win else f" {by[i]:.3f} ") if i in by else "   --  " for i in grid)


def collect(api, P):
    """{(family, variant, protocol): {"sweep", "grid", "runs"}}; protocol in with / without / prefix."""
    import iql_score
    sp = P.spec()
    want = {n: f for n, f in sp.items() if hold.shown(P.method_of(n))}
    prefix = {n: {k: v for k, v in f.items() if k != "seq_labeller"}
              for n, f in sp.items() if n.startswith("pt_sweeps/")}           # §14.10
    gspec = P.control_grid_spec(sp)
    gfixed = {n: f for n, (a, f) in gspec.items()}
    out = {}
    for sw in api.project(P.IQL_PROJECT, entity=P.ENTITY).sweeps():
        params = sw.config.get("parameters")
        flat = P.flat(params)
        for table, proto in ((want, "with"), (prefix, "prefix"), (gfixed, "without")):
            hits = P.match_entry(params, table)
            if len(hits) != 1:
                continue
            name = hits[0]
            if proto == "without":
                anchor = gspec[name][0]
                fam = "heuristic" if P.is_heuristic_anchor(anchor) else P.method_of(anchor)
                v = name.split("s4noterm_antmaze_")[1]
                v = next(x for x in VARIANTS if v.startswith(x))
                grid = list(P.grid_of(anchor))
            else:
                fam, v, grid = P.method_of(name), P.variant_of_name(name), list(P.grid_of(name))
            if not hold.shown(fam) or P.lineage_of(flat, grid) != "stage4":
                continue
            key = (fam, v, proto)
            if key in out:
                out[key]["note"] = f"second grid sweep {sw.id} ignored"
                continue
            out[key] = {"sweep": sw.id, "grid": grid, "runs": [iql_score.score_run(r) for r in sw.runs]}
    return out


def report(grids, out=print):
    out("=" * 110)
    out("NORMALIZATION INVENTORY (handoff §14.2 item 5).  Seed 0; stage-4 rule; [winner].  DESCRIPTIVE.")
    if not hold.INCLUDE:
        out("BNN/CVaR HOLD (§14.1): BNN CVaR, BNN mean and ensemble CVaR are not fetched.")
    out("=" * 110)
    moves = []
    for fam in FAMILIES:
        out(f"\n{fam}")
        for v in VARIANTS:
            wins = {}
            for proto in ("with", "without") + (("prefix",) if fam == "PT" else ()):
                g = grids.get((fam, v, proto))
                label = {"with": "with terminals", "without": "no terminals",
                         "prefix": "pre-fix (time-only reward, §14.10)"}[proto]
                if g is None:
                    if fam == "heuristic" and proto == "with":
                        out(f"  {v:15s} {label:36s} none: index 3 fixed in advance (§13.14)")
                        wins[proto] = 3
                    else:
                        out(f"  {v:15s} {label:36s} not run")
                    continue
                status, win, by = grid_status(g["runs"], g["grid"])
                wins[proto] = win
                out(f"  {v:15s} {label:36s} {g['sweep']}  {status:20s} {fmt_row(g['grid'], by, win)}"
                    + (f"   note: {g['note']}" if g.get("note") else ""))
            if wins.get("with") is not None and wins.get("without") is not None:
                moved = wins["with"] != wins["without"]
                moves.append((fam, v, wins["with"], wins["without"], moved))
                out(f"  {'':15s} {'index with -> without terminals':36s} {wins['with']} -> {wins['without']}"
                    f"{'   MOVES' if moved else '   same'}")
    out(f"\nSUMMARY: the selected index moves when the terminals go in {sum(m[4] for m in moves)} of "
        f"{len(moves)} (family, variant) pairs with both grids complete.")
    return moves


def selftest():
    def runs(scores, grid=range(8)):
        return [{"id": f"r{i}", "normalize_reward": i, "score": s, "error": None if s is not None else "running"}
                for i, s in zip(grid, scores)]
    st, w, by = grid_status(runs([.1, .5, .2, .5, 0, 0, .3, .1]), list(range(8)))
    assert (st, w) == ("done", 1) and by[3] == .5                       # tie 1/3 -> lowest index
    st, w, _ = grid_status(runs([.1, .5, None, .5, 0, 0, .3, .1]), list(range(8)))
    assert st == "running (7/8 scored)" and w is None
    hold.INCLUDE = False
    g = {("MR", "medium_play", "with"): {"sweep": "a", "grid": list(range(8)), "runs": runs([.6, .5, .4, .3, 0, 0, .2, .1])},
         ("MR", "medium_play", "without"): {"sweep": "b", "grid": list(range(8)), "runs": runs([.5, .7, .4, .3, 0, 0, .2, .1])},
         ("task_reward", "large_play", "with"): {"sweep": "c", "grid": list(range(4)), "runs": runs([.1, .4, .3, .2], range(4))},
         ("heuristic", "large_play", "without"): {"sweep": "d", "grid": list(range(8)), "runs": runs([.1, .1, .1, .5, 0, 0, 0, 0])}}
    lines = []
    moves = report(g, lines.append)
    txt = "\n".join(lines)
    assert ("MR", "medium_play", 0, 1, True) in moves and ("heuristic", "large_play", 3, 3, False) in moves, moves
    body = [x for x in lines if "HOLD" not in x]
    assert "[0.400]" in txt and "index 3 fixed in advance" in txt and not any("BNN" in x for x in body)
    assert "1 of 2 (family, variant) pairs" in txt, txt
    print("selftest OK")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    hold.parse()
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    report(collect(wandb.Api(timeout=120), P))
    return 0


if __name__ == "__main__":
    sys.exit(main())
