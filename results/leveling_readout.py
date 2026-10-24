#!/usr/bin/env python
"""leveling_readout.py -- the leveling ablation and its secondary ties arm (handoff §14.40).
DESCRIPTIVE: exact paired 95% intervals over seeds 1-10, no test.

medium_diverse; families MR, PT, NMR, ensemble mean.  The cells (§14.40's table):
    A1  terminals,  ties removed, leveled      the anchor evaluation cell
    A2  terminals,  ties removed, NOT leveled  (same index as A1)
    A3  no terms,   ties removed, leveled      the no-terminals cell at its winner
    A4  no terms,   ties removed, NOT leveled  (same index as A3)
    R   terminals,  ties removed, NOT leveled, index 7   (the PT pipeline's setup)
    B1 = A3, B2 = A4
    B3  no terms,   ties KEPT,    leveled      the ties ablation's cell (ensemble: lev_B3)
    B4  no terms,   ties KEPT,    NOT leveled
Comparisons: A2 - A1 (P1), A4 - A3 (P2), R - A1, B3 - B1 and B4 - B2 (P3), and the
ties-by-leveling interaction (B4 - B2) - (B3 - B1), per seed.  Each cell's sd is shown,
since P1 predicts a wider spread without leveling.

    python results/leveling_readout.py [--selftest]
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
import control_readout as CR                               # noqa: E402

V = "medium-diverse"
FAMILIES = ("MR", "PT", "NMR", "ENS_MEAN")
SEEDS = R.SEEDS


def cells_for(anchor, ctrl, gwin, winners, m):
    """{cell name: cell or None} for one family."""
    a3 = CR.at_winner(ctrl, gwin, winners, V, m) if m == "MR" else ctrl.get((V, f"atwin_{m}"))
    b3 = ctrl.get((V, f"tiesnt_{m}")) if m != "ENS_MEAN" else ctrl.get((V, "levB3_ENS_MEAN"))
    a4 = ctrl.get((V, f"levA4_{m}"))
    return {"A1": anchor.get((f"antmaze-{V}-v2", m)), "A2": ctrl.get((V, f"levA2_{m}")),
            "A3": a3, "A4": a4, "R": ctrl.get((V, f"levR_{m}")),
            "B1": a3, "B2": a4, "B3": b3, "B4": ctrl.get((V, f"levB4_{m}"))}


def describe(c):
    r = R.cell_ready(c)
    if r:
        return f"{'no sweep' if c is None else c['sweep']}  {r}"
    x = np.array([c["scores"][s] for s in SEEDS])
    return f"{c['sweep']}  idx {c['index']}  {x.mean():.3f} ± {x.std(ddof=1):.3f}"


def interaction(b1, b2, b3, b4):
    d = np.array([(b4["scores"][s] - b2["scores"][s]) - (b3["scores"][s] - b1["scores"][s]) for s in SEEDS])
    bs = R.exact_bootstrap(d)
    return d.mean(), bs["lower"], bs["upper"]


def report(by_family, out=print):
    out("=" * 100)
    out("LEVELING ABLATION, medium_diverse (handoff §14.40).  DESCRIPTIVE.  Paired by seed; 95% exact bootstrap.")
    out("=" * 100)
    for m, C in by_family.items():
        out(f"\n{m}")
        for k in ("A1", "A2", "A3", "A4", "R", "B3", "B4"):
            out(f"  {k:3s} {describe(C[k])}")
        for tag, a, b in (("A2 - A1  (P1: leveling, with terminals)", "A2", "A1"),
                          ("A4 - A3  (P2: leveling, without terminals)", "A4", "A3"),
                          ("R  - A1  (PT pipeline vs anchor)", "R", "A1"),
                          ("B3 - B1  (ties, leveled, no terminals)", "B3", "B1"),
                          ("B4 - B2  (ties, not leveled, no terminals)", "B4", "B2")):
            CR.line(C[a], C[b], out, f"  {tag:44s}")
        if all(R.cell_ready(C[k]) is None for k in ("B1", "B2", "B3", "B4")):
            mean, lo, hi = interaction(C["B1"], C["B2"], C["B3"], C["B4"])
            out(f"  {'(B4 - B2) - (B3 - B1)  (P3: ties x leveling)':44s} {mean:+.3f}   95% exact bootstrap [{lo:+.3f}, {hi:+.3f}]")
        else:
            out(f"  {'(B4 - B2) - (B3 - B1)  (P3: ties x leveling)':44s} NOT READ -- a B cell is not complete")


def selftest():
    rng = np.random.default_rng(0)

    def cell(vals, sweep="s", idx=7):
        return {"scores": dict(zip(SEEDS, vals)), "robust": dict(zip(SEEDS, vals)), "max": dict(zip(SEEDS, vals)),
                "incomplete": [], "audit": [], "sweep": sweep, "index": idx, "name": "n"}
    b = rng.uniform(0.2, 0.4, 10)
    C = {"A1": cell(b + 0.3), "A2": cell(b + 0.1), "A3": cell(b), "A4": cell(b - 0.05), "R": cell(b + 0.2),
         "B1": None, "B2": None, "B3": cell(b + 0.01), "B4": cell(b - 0.15)}
    C["B1"], C["B2"] = C["A3"], C["A4"]
    lines = []
    report({"MR": C}, lines.append)
    txt = "\n".join(lines)
    assert "A2 - A1" in txt and "diff -0.200" in txt                       # P1
    assert "diff -0.050" in txt                                            # A4 - A3
    m, lo, hi = interaction(C["B1"], C["B2"], C["B3"], C["B4"])
    assert abs(m - (-0.11)) < 1e-9 and lo <= m <= hi, (m, lo, hi)          # (-0.10) - (0.01)
    C2 = dict(C, B4=None)
    lines = []
    report({"MR": C2}, lines.append)
    assert "NOT READ -- a B cell is not complete" in "\n".join(lines)
    # cells_for picks the right keys
    ctrl = {(V, "levA2_PT"): 1, (V, "levA4_PT"): 2, (V, "levR_PT"): 3, (V, "levB4_PT"): 4, (V, "tiesnt_PT"): 5,
            (V, "atwin_PT"): 6, (V, "levB3_ENS_MEAN"): 7, (V, "atwin_ENS_MEAN"): 8}
    cp = cells_for({(f"antmaze-{V}-v2", "PT"): 0}, ctrl, {}, {}, "PT")
    assert cp == {"A1": 0, "A2": 1, "A3": 6, "A4": 2, "R": 3, "B1": 6, "B2": 2, "B3": 5, "B4": 4}, cp
    ce = cells_for({}, ctrl, {}, {}, "ENS_MEAN")
    assert ce["B3"] == 7 and ce["A3"] == 8
    print("selftest OK")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    import hold
    hold.parse()
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    winners = json.load(open(os.path.join(ROOT, "phase2_winners.json")))
    gpath = os.path.join(ROOT, P.GRID_WINNERS)
    gwin = json.load(open(gpath)) if os.path.exists(gpath) else {}
    api = wandb.Api(timeout=120)
    anchor, _ = R.collect(api, P, winners)
    ctrl, _ = CR.collect_controls(api, P, winners, gwin)
    report({m: cells_for(anchor, ctrl, gwin, winners, m) for m in FAMILIES})
    return 0


if __name__ == "__main__":
    sys.exit(main())
