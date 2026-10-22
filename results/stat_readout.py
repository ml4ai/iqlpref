#!/usr/bin/env python
"""stat_readout.py -- max versus last-10 (handoff §14.2 item 4).  DESCRIPTIVE: no verdict.

The Preference Transformer paper reports the MAXIMUM of a run's evaluation curve;
this campaign reports the mean of its LAST TEN points.  For every finished cell of
an uncovered family this prints both statistics, and for every comparison within a
group (paired by seed, exact bootstrap over seeds, as §12.5) whether moving from
last-10 to max changes:
  - the RANKING of the group's cells by mean;
  - the SIGN of a paired difference;
  - its SEPARATION: whether the 95% exact interval excludes zero.

Groups, each read separately:
  A. the anchor, with terminals, per variant (task reward at both indices, MR, PT,
     ensemble mean);
  B. the controls, per variant (no-terminals cells, re-runs, motion reward; from
     control_readout's collector);
  C. each family's degradation ladder, per variant and axis (the full-data anchor
     and every finished experiment level).

A cell is read only when complete and audit-clean (§12.5).  Covered families (the
BNN/CVaR hold, §14.1) are never fetched; --include-held overrides, only once the
author has lifted the hold.

    python results/stat_readout.py
    python results/stat_readout.py --selftest
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import eval_readout as R                                   # noqa: E402
import hold                                                # noqa: E402

VARIANTS = R.VARIANTS
SEEDS = R.SEEDS
STATS = (("scores", "last-10"), ("max", "max"))
ANCHOR_METHODS = ("task_reward", "task_reward_r1", "MR", "PT", "NMR", "ENS_MEAN",
                  "PT_TIMEONLY")              # the pre-fix PT cells as a control (§14.10, §14.24)
LADDER_FAMILIES = ("MR", "PT", "NMR", "ENS_MEAN")


def ready(cell):
    """None if both statistics may be read, else the reason not."""
    r = R.cell_ready(cell)
    if r:
        return r
    if sorted(cell.get("max", {})) != SEEDS:
        return "no max statistic for every seed"
    return None


def separated(c):
    return c["lower"] > 0 or c["upper"] < 0


def read_group(title, members, out):
    """members: [(label, cell)].  Prints the group; returns its counts."""
    out(f"\n  {title}")
    ok = [(lab, c) for lab, c in members if ready(c) is None]
    for lab, c in members:
        r = ready(c)
        if r:
            out(f"    {lab:24s} not read -- {r}")
    n = {"groups": 0, "rank_changes": 0, "pairs": 0, "sign_flips": 0, "sep_changes": 0}
    if len(ok) < 2:
        if ok:
            lab, c = ok[0]
            out(f"    {lab:24s} last-10 {np.mean([c['scores'][s] for s in SEEDS]):.3f}   "
                f"max {np.mean([c['max'][s] for s in SEEDS]):.3f}   (nothing to compare)")
        return n
    n["groups"] = 1
    means = {}
    for key, name in STATS:
        means[key] = {lab: float(np.mean([c[key][s] for s in SEEDS])) for lab, c in ok}
    for lab, c in ok:
        sd = {key: float(np.std([c[key][s] for s in SEEDS], ddof=1)) for key, _ in STATS}
        out(f"    {lab:24s} last-10 {means['scores'][lab]:.3f} ± {sd['scores']:.3f}   "
            f"max {means['max'][lab]:.3f} ± {sd['max']:.3f}   "
            f"max - last-10 {means['max'][lab] - means['scores'][lab]:+.3f}")
    rank = {key: sorted(means[key], key=lambda lab: -means[key][lab]) for key, _ in STATS}
    changed = rank["scores"] != rank["max"]
    n["rank_changes"] += int(changed)
    out(f"    ranking  last-10: {' > '.join(rank['scores'])}")
    out(f"             max:     {' > '.join(rank['max'])}"
        f"{'   ** RANKING CHANGES' if changed else '   (same)'}")
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            (la, a), (lb, b) = ok[i], ok[j]
            c10, cmx = R.compare(a, b, "scores"), R.compare(a, b, "max")
            flip = np.sign(c10["mean"]) != np.sign(cmx["mean"])
            sep = separated(c10) != separated(cmx)
            n["pairs"] += 1
            n["sign_flips"] += int(flip)
            n["sep_changes"] += int(sep)
            flags = ", ".join(x for x, f in (("SIGN FLIPS", flip), ("SEPARATION CHANGES", sep)) if f)
            out(f"    {la} - {lb}:  last-10 {c10['mean']:+.3f} [{c10['lower']:+.3f}, {c10['upper']:+.3f}]   "
                f"max {cmx['mean']:+.3f} [{cmx['lower']:+.3f}, {cmx['upper']:+.3f}]"
                + (f"   ** {flags}" if flags else ""))
    return n


def groups(anchor, ctrl, exp):
    """[(section, title, [(label, cell)])] in reading order."""
    out = []
    for v in VARIANTS:
        ds = f"antmaze-{v}-v2"
        mem = [(m, anchor[(ds, m)]) for m in hold.show(ANCHOR_METHODS) if (ds, m) in anchor]
        out.append(("A", f"{v}: the anchor, with terminals", mem))
    for v in VARIANTS:
        mem = [(k, c) for (vv, k), c in sorted(ctrl.items()) if vv == v and hold.shown_kind(k)]
        out.append(("B", f"{v}: the controls", mem))
    for axis, levels in exp.items():
        fams = sorted({m for (_, m, _) in levels} - {None})
        for v in VARIANTS:
            for m in fams:
                if not hold.shown(m):
                    continue
                cells = {lev: c for (vv, mm, lev), c in levels.items() if vv == v and mm == m}
                if len(cells) < 2:
                    continue
                order = sorted(cells, key=lambda lev: (lev != "full", len(lev), lev))   # r16 < r128
                out.append(("C", f"{v}: {m}, {axis} ladder", [(lev, cells[lev]) for lev in order]))
    return out


def report(gs, notes, out=print):
    out("=" * 100)
    out("MAX vs LAST-10 (handoff §14.2 item 4).  DESCRIPTIVE.  Paired by seed; 95% exact bootstrap.")
    out("Flags: ranking of cell means changes; a paired difference changes sign; its interval")
    out("changes between excluding and including zero.")
    if not hold.INCLUDE:
        out("BNN/CVaR HOLD (HANDOFF §14.1): BNN CVaR, BNN mean and ensemble CVaR are not fetched.")
    out("=" * 100)
    for x in notes:
        out(f"  note: {x}")
    tot = {"groups": 0, "rank_changes": 0, "pairs": 0, "sign_flips": 0, "sep_changes": 0}
    sec = None
    for s, title, mem in gs:
        if s != sec:
            out({"A": "\nA. THE ANCHOR, WITH TERMINALS", "B": "\nB. THE CONTROLS",
                 "C": "\nC. DEGRADATION LADDERS"}[s])
            sec = s
        for k, v in read_group(title, mem, out).items():
            tot[k] += v
    out(f"\nSUMMARY: {tot['groups']} groups read; ranking changes in {tot['rank_changes']}; "
        f"{tot['pairs']} paired comparisons, sign flips {tot['sign_flips']}, "
        f"separation changes {tot['sep_changes']}.")
    return tot


def selftest():
    rng = np.random.default_rng(1)

    def cell(l10, mx):
        return {"scores": dict(zip(SEEDS, l10)), "robust": dict(zip(SEEDS, l10)),
                "max": dict(zip(SEEDS, mx)), "incomplete": [], "audit": [], "sweep": "s",
                "index": 0, "name": "n"}
    base = rng.uniform(0.3, 0.5, 10)
    a = cell(base + 0.10, base + 0.20)        # ahead on last-10 by 0.1, behind on max by 0.1
    b = cell(base, base + 0.30)
    c = cell(base - 0.20, base + 0.25)
    lines = []
    tot = read_group("g", [("a", a), ("b", b), ("c", c)], lines.append)
    txt = "\n".join(lines)
    assert "a > b > c" in txt and "b > c > a" in txt and "RANKING CHANGES" in txt, txt
    assert tot == {"groups": 1, "rank_changes": 1, "pairs": 3, "sign_flips": 2, "sep_changes": 0}, tot
    # an incomplete cell is listed and not read; the max must be complete too
    d = cell(base, base)
    del d["max"][3]
    lines = []
    tot = read_group("g", [("a", a), ("d", d)], lines.append)
    assert tot["groups"] == 0 and "no max statistic" in "\n".join(lines)
    # groups(): the hold drops covered methods; ladders need two levels
    hold.INCLUDE = False
    anc = {("antmaze-medium-play-v2", m): a for m in ("MR", "PT", "BNN_CVAR", "ENS_CVAR", "ENS_MEAN")}
    ctrl = {("medium-play", "noterm_MR"): a, ("medium-play", "noterm_BNN_CVAR"): b}
    exp = {"sparsity": {("medium-play", "MR", "full"): a, ("medium-play", "MR", "r16"): b,
                        ("medium-play", "PT", "full"): a,
                        ("medium-play", "BNN_MEAN", "full"): a, ("medium-play", "BNN_MEAN", "r16"): b}}
    gs = groups(anc, ctrl, exp)
    flat = " ".join(f"{t} {[l for l, _ in m]}" for _, t, m in gs)
    assert "BNN" not in flat and "ENS_CVAR" not in flat, flat
    ladders = [(t, [l for l, _ in m]) for s, t, m in gs if s == "C"]
    assert ladders == [("medium-play: MR, sparsity ladder", ["full", "r16"])], ladders
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
    import control_readout as C
    import exp_readout as X
    winners = json.load(open(os.path.join(ROOT, "phase2_winners.json")))
    api = wandb.Api(timeout=120)
    anchor, notes = R.collect(api, P, winners)
    t_cells, t_notes = R.collect_timeonly(api, P)
    anchor.update(t_cells)
    notes += t_notes
    gpath = os.path.join(ROOT, P.GRID_WINNERS)
    gwin = json.load(open(gpath)) if os.path.exists(gpath) else {}
    ctrl, cnotes = C.collect_controls(api, P, winners, gwin)
    exp = {}
    for axis in X.AXES:
        cells, xnotes = X.collect_exp(api, P, winners, axis)
        notes += xnotes
        for (dataset, m), c in anchor.items():          # the anchor is every ladder's full level
            if m in LADDER_FAMILIES:
                cells[(dataset.replace("antmaze-", "").replace("-v2", ""), m, "full")] = c
        exp[axis] = cells
    report(groups(anchor, ctrl, exp), notes + cnotes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
