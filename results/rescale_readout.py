#!/usr/bin/env python
"""rescale_readout.py -- the rescaling companion arms against their frozen cells (handoff §14.78).
DESCRIPTIVE; not part of §14.5's test, and no verdict.

Every candidate (no-terminals) cell whose frozen index does not rescale (0: raw; 1: r - 1)
was re-run, full data and every ladder level, at the family's best rescaling index
(phase2_sweeps.RESCALE_ARM).  Same reward models, same everything else.  So per level:

    frozen     the candidate cell §14.5 reads (at the frozen no-terminals index)
    companion  the same cell at the rescaling index
    companion - frozen, paired by seed, exact 95% bootstrap (eval_readout.exact_bootstrap)

and per arm the degradation slope (as rq1_readout.dslope: OLS over all of the ladder's points,
the anchor included; positive = falls as the data degrades), per seed, with its mean and
exact 10-seed interval; and companion slope - frozen slope, paired by seed.

THE GATE.  A family's frozen candidate cells are part of §14.5's test and may not be read
before that family's registered reading (§14.5's do-NOTs).  Its companions would reveal
the same arm, so they wait too.  REGISTERED_READ lists the (ladder, family) readings done;
add to it, by hand and in the handoff, as each registered reading is made.  Beyond that,
every cell shown is complete and audit-clean.

    python results/rescale_readout.py [--axis sparsity|noise]
    python results/rescale_readout.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import eval_readout as R                                   # noqa: E402
import rq1_readout as Q                                    # noqa: E402

SEEDS = R.SEEDS
REGISTERED_READ = {"sparsity": {"MR", "PT", "ENS_MEAN", "NMR"}, "noise": set()}     # §14.70 MR; §14.87 PT; §14.88 ENS_MEAN; §14.101 NMR
LEVEL_ORDER = ("full",)                                     # then the axis's levels, coarse to fine


def companion_cells(api, P, sp, gwin, axis):
    """{(variant, method, level): cell} for the companion sweeps of the axis (exact match,
    index = the companion index), level "full" for the full-data companions."""
    import iql_score
    E = P._conditions()
    spec = {n: v for n, v in P.rescale_arm_spec(sp, gwin, E).items()
            if v[3] is None or v[3] in Q.AXES[axis]["levels"]}
    fixed_of = {n: f for n, (_, f, _, _) in spec.items()}
    cells, notes = {}, []
    for sw in P.iql_sweeps(api):
        hits = P.match_entry(sw.config.get("parameters"), fixed_of)
        # a companion shares its frozen cell's fixed params; only the index tells them apart
        flat = P.flat(sw.config.get("parameters") or {})
        hits = [h for h in hits if flat.get("normalize_reward") == spec[h][2] and flat.get("seed") == P.EVAL_SEEDS]
        if len(hits) != 1:
            continue
        name = hits[0]
        anchor, fixed, k, cond = spec[name]
        key = (Q.VARIANTS[[v.replace("-", "_") for v in Q.VARIANTS].index(P.variant_of_name(anchor))],
               P.method_of(anchor), cond or Q.FULL)
        cell = {"sweep": sw.id, "index": k, "scores": {}, "incomplete": [], "audit": [], "name": name}
        for r in sw.runs:
            s = (r.config or {}).get("seed")
            cell["audit"] += [f"run {r.id}: {x}" for x in R.audit_run(r.config or {}, name, s, fixed, k, False)]
            if r.state != "finished":
                cell["incomplete"].append(f"seed {s} {r.state}")
                continue
            try:
                _, curve = iql_score.run_curve(r)
                cell["scores"][s] = iql_score.score_curve(curve, n=iql_score.LAST_N)
            except (iql_score.IncompleteRun, ValueError) as e:
                cell["incomplete"].append(f"seed {s}: {e}")
        if key in cells:
            notes.append(f"{key}: two companion sweeps ({cells[key]['sweep']}, {sw.id}) -- not read")
            cells[key]["audit"].append(f"second sweep {sw.id}")
            continue
        cells[key] = cell
    return cells, notes


def compare(frozen, comp, axis, v, fam):
    """The per-level and slope comparison for one (variant, family), or the reason it is not read."""
    levels = (Q.FULL,) + tuple(Q.AXES[axis]["levels"])
    why = [f"{arm} {lv}: {R.cell_ready(c)}" for lv in levels
           for arm, c in (("frozen", frozen.get((v, fam, lv))), ("companion", comp.get((v, fam, lv))))
           if R.cell_ready(c)]
    if why:
        return None, why
    rows = []
    for lv in levels:
        a = [comp[(v, fam, lv)]["scores"][s] for s in SEEDS]
        b = [frozen[(v, fam, lv)]["scores"][s] for s in SEEDS]
        rows.append((lv, float(np.mean(b)), float(np.std(b, ddof=1)), float(np.mean(a)), float(np.std(a, ddof=1)),
                     R.exact_bootstrap(np.array(a) - np.array(b))))
    xs = [Q.AXES[axis]["x"](v, lv) for lv in levels]
    sl = {}
    for arm, cells in (("frozen", frozen), ("companion", comp)):
        sl[arm] = [Q.dslope(xs, [cells[(v, fam, lv)]["scores"][s] for lv in levels], Q.AXES[axis]["sign"])
                   for s in SEEDS]
    slopes = {arm: R.exact_bootstrap(sl[arm]) for arm in sl}
    slopes["companion - frozen"] = R.exact_bootstrap(np.array(sl["companion"]) - np.array(sl["frozen"]))
    return {"rows": rows, "slopes": slopes}, None


def _iv(e):
    return f"{e['mean']:+.4f} [{e['lower']:+.4f}, {e['upper']:+.4f}]"


def report(results, axis, frozen_idx, comp_idx, out=print):
    out("=" * 100)
    out(f"RESCALING COMPANION ARMS (HANDOFF §14.78), {axis} ladder, WITHOUT terminals.  DESCRIPTIVE; not §14.5's test.")
    out("frozen = the candidate cell at its frozen index; companion = the same cell at the family's best rescaling index")
    out("=" * 100)
    for (v, fam), (res, why) in results.items():
        out(f"\n{fam}, {v}: frozen index {frozen_idx[(v, fam)]}, companion index {comp_idx[(v, fam)]}")
        if res is None:
            out("  NOT READ -- " + "; ".join(why[:4]) + (f" (+{len(why) - 4} more)" if len(why) > 4 else ""))
            continue
        for lv, fm, fs, cm, cs, d in res["rows"]:
            out(f"  {lv:5s} frozen {fm:.3f} ± {fs:.3f}   companion {cm:.3f} ± {cs:.3f}   companion - frozen {_iv(d)}")
        for arm, e in res["slopes"].items():
            out(f"  degradation slope, {arm:18s} {_iv(e)}")


def run(axis, frozen, comp, sp_anchor_idx):
    """{(variant, family): (result or None, reasons)} for every companion cell of the axis."""
    out = {}
    for (v, fam) in sp_anchor_idx:
        if fam not in REGISTERED_READ[axis]:
            out[(v, fam)] = (None, [f"{fam}'s registered {axis} reading has not been made "
                                    f"(§14.5: its candidate arm is not read before then)"])
            continue
        out[(v, fam)] = compare(frozen, comp, axis, v, fam)
    return out


def selftest():
    rng = np.random.default_rng(3)

    def cell(xs):
        return {"scores": {s: float(round(x, 3)) for s, x in zip(SEEDS, xs)}, "audit": [], "incomplete": []}
    axis = "sparsity"
    levels = (Q.FULL,) + tuple(Q.AXES[axis]["levels"])
    v = "medium-play"
    xs = np.array([Q.AXES[axis]["x"](v, lv) for lv in levels])
    frozen, comp = {}, {}
    for i, lv in enumerate(levels):
        dx = (xs[i] - xs.mean()) / (xs.max() - xs.min())
        frozen[(v, "MR", lv)] = cell(0.4 + 0.4 * dx + rng.normal(0, 0.01, 10))      # declines
        comp[(v, "MR", lv)] = cell(0.4 + rng.normal(0, 0.01, 10))                  # flat
        frozen[(v, "NMR", lv)] = cell(0.4 + rng.normal(0, 0.01, 10))
        comp[(v, "NMR", lv)] = cell(0.4 + rng.normal(0, 0.01, 10))
    res = run(axis, frozen, comp, [(v, "MR")])
    res.update(run("noise", frozen, comp, [(v, "NMR")]))      # the gate: nothing is read on the noise ladder yet
    r, why = res[(v, "MR")]
    assert why is None and r["slopes"]["frozen"]["lower"] > 0 and r["slopes"]["companion"]["upper"] < 0.05
    assert r["slopes"]["companion - frozen"]["upper"] < 0                         # the companion is flatter
    assert res[(v, "NMR")][0] is None and "registered noise reading" in res[(v, "NMR")][1][0]   # the gate
    comp[(v, "MR", "r32")]["incomplete"] = ["seed 2 running"]
    assert run(axis, frozen, comp, [(v, "MR")])[(v, "MR")][0] is None              # completeness
    lines = []
    report(res, axis, {(v, "MR"): 1, (v, "NMR"): 1}, {(v, "MR"): 2, (v, "NMR"): 3}, out=lines.append)
    assert any("companion - frozen" in x for x in lines) and any("NOT READ" in x for x in lines)
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", choices=tuple(Q.AXES), default="sparsity")
    ap.add_argument("--selftest", action="store_true")
    import hold
    hold.parse()
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    winners = json.load(open(os.path.join(ROOT, "phase2_winners.json")))
    gwin = json.load(open(os.path.join(ROOT, P.GRID_WINNERS)))
    api = wandb.Api(timeout=120)
    sp = P.spec()
    pairs, fidx, cidx = [], {}, {}
    for anchor, k in P.RESCALE_ARM.items():
        v = Q.VARIANTS[[x.replace("-", "_") for x in Q.VARIANTS].index(P.variant_of_name(anchor))]
        fam = P.method_of(anchor)
        pairs.append((v, fam))
        fidx[(v, fam)], cidx[(v, fam)] = gwin[anchor], k
    readable = {f for f in REGISTERED_READ[a.axis]}
    frozen, comp, notes = {}, {}, []
    if readable:                       # nothing of a family's candidate arm is fetched before its reading
        _, cand, full, _, n1 = Q.collect(api, P, winners, gwin, a.axis)
        frozen = {k: c for k, c in cand.items() if k[1] in readable}
        comp, n2 = companion_cells(api, P, sp, gwin, a.axis)
        comp = {k: c for k, c in comp.items() if k[1] in readable}
        notes = n2
    for n in notes:
        print("note:", n)
    report(run(a.axis, frozen, comp, pairs), a.axis, fidx, cidx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
