#!/usr/bin/env python
"""render_prelim.py -- the proposal's four Preliminary Results floats, from the snapshot only
(handoff §14.135).  Never touches W&B: it reads results/paper/snapshot_prelim.json (written
by extract_prelim.py), rebuilds the collectors' cells, and reruns the registered reading
offline with rq1_readout's own code (assemble, run), so the verdicts table cannot drift
from the readout.

Outputs (no titles or captions baked in; the proposal keeps each float's wrapper):
    fig_prelim_mr_ladder.pdf        MR on the sparsity ladder: (a) held-out accuracy,
                                    (b) policy score with and without terminals
    tab_prelim_trivial.tex          with terminals, full data: zero, constant, task reward
                                    at index 1, MR
    tab_prelim_verdicts.tex         RQ1's registered sparsity reading, every family read
    tab_prelim_verdicts_mr_pt.tex   the same, MR and PT only (the proposal's current layout)
    tab_prelim_leak.tex             large-diverse without terminals: as run and leak-free
Tables are the tabular environment only (booktabs).  Scores and accuracies x100; slopes
per halving of the training pairs x100 (rq1_readout's degradation slope: positive = falls).
Each table also gets a comment line naming the snapshot it came from.

    python results/paper/render_prelim.py [--out DIR] [--snapshot PATH] [--no-copy]
--out defaults to ~/Research/loren/proposal/figures/generated/; a copy also goes to
results/paper/out/ unless --no-copy.  The script prints every number it renders.
"""

import argparse
import decimal
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.dirname(HERE)
sys.path.insert(0, RESULTS)

import control_readout as C                                # noqa: E402
import eval_readout as R                                   # noqa: E402
import rq1_readout as Q                                    # noqa: E402

SNAPSHOT = os.path.join(HERE, "snapshot_prelim.json")
OUT = os.path.expanduser("~/Research/loren/proposal/figures/generated")
COPY = os.path.join(HERE, "out")
VARIANTS = Q.VARIANTS
LEVELS = (Q.FULL, "r128", "r64", "r32", "r16")              # left to right: full data, then fewer pairs
TICK = {Q.FULL: "full", "r128": "128", "r64": "64", "r32": "32", "r16": "16"}
TEAL, CORAL, PURPLE, GRAY = "#0F6E56", "#D85A30", "#534AB7", "#5F5E5A"   # the proposal's palette
DATASET_STYLE = dict(zip(VARIANTS, ((TEAL, "o"), (CORAL, "s"), (PURPLE, "^"), (GRAY, "D"))))
FAMILY_LABEL = {"MR": "MR", "PT": "PT", "NMR": "NMR", "ENS_MEAN": "Ensemble mean"}
LEAK_VARIANT = "large-diverse"


# --------------------------------------------------------------------------
# The snapshot
# --------------------------------------------------------------------------
def rebuild(snap):
    """{group: {key: cell}} as the collectors returned them, and A as rq1_readout reads it."""
    groups = {}
    for c in snap["cells"]:
        groups.setdefault(c["group"], {})[tuple(c["key"])] = {
            "sweep": c["sweep"], "index": c["index"], "name": c["name"], "audit": list(c["audit"]),
            "incomplete": list(c["incomplete"]), "scores": {}, "robust": {}, "max": {}, "runs": {}}
    for r in snap["runs"]:
        cell = groups[r["group"]][tuple(r["key"])]
        cell["runs"][r["seed"]] = r
        for k in ("score", "robust", "max"):
            if r[k] is not None:
                cell["scores" if k == "score" else k][r["seed"]] = r[k]
    acc = {(a["family"], a["dataset"], a["level"], a["seed"]): a["value"] if a["value"] is not None else a["reason"]
           for a in snap["accuracy"]}
    return groups, acc


def r1(x):
    """One decimal, half up, on the value with float noise removed: scores sit on a 0.001
    lattice, so a mean x100 is an exact two-decimal number, and 46.95 must not print as 46.9
    because the float is 46.9499999...  Used for every number in every output."""
    q = decimal.Decimal(repr(round(float(x), 9))).quantize(decimal.Decimal("0.1"), decimal.ROUND_HALF_UP)
    return f"{q:.1f}"


def sr1(x):
    """r1 with an explicit sign (+3.1, -0.0)."""
    t = r1(x)
    return t if t.startswith("-") else "+" + t


def stats(cell):
    """(mean, sd) x100 over seeds 1-10 of a ready cell, else None."""
    if R.cell_ready(cell):
        return None
    x = 100 * np.array([cell["scores"][s] for s in R.SEEDS])
    return float(x.mean()), float(x.std(ddof=1))


def pm(cell):
    s = stats(cell)
    return "n/a" if s is None else f"${r1(s[0])} \\pm {r1(s[1])}$"


def acc_mean(acc, fam, v, lv):
    x = [acc.get((fam, v, lv, s)) for s in R.SEEDS]
    return 100 * float(np.mean(x)) if all(isinstance(a, float) for a in x) else None


# --------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------
def style():
    import logging

    import matplotlib
    logging.getLogger("fontTools").setLevel(logging.ERROR)    # cmr10's header dates warn when subset; harmless
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["CMU Serif", "Latin Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4,
        "axes.grid": True, "grid.color": "#D9D9D6", "grid.linewidth": 0.5, "axes.axisbelow": True})
    return plt


def mr_ladder(std, cand, acc, gwin, path, log):
    """fig_prelim_mr_ladder.pdf: (a) MR's held-out accuracy per dataset; (b) its policy score
    per dataset, with terminals (coral, dashed, open) and without (teal, solid, filled).
    Means over seeds 1-10, no bands (both panels stay readable)."""
    plt = style()
    fig = plt.figure(figsize=(6.5, 4.5), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.05, 1], hspace=0.04)
    gs = top.add_gridspec(1, 3, width_ratios=[0.19, 0.62, 0.19])
    ax = top.add_subplot(gs[0, 1])
    x = np.arange(len(LEVELS))
    for v in VARIANTS:
        col, mk = DATASET_STYLE[v]
        y = [acc_mean(acc, "MR", v, lv) for lv in LEVELS]
        log(f"  (a) {v:15s} " + "  ".join(f"{TICK[lv]} {'n/a' if a is None else r1(a)}" for lv, a in zip(LEVELS, y)))
        ax.plot(x, [np.nan if a is None else a for a in y], color=col, marker=mk, label=v)
    ax.set_xticks(x, [TICK[lv] for lv in LEVELS])
    ax.set_ylim(70, 97)
    ax.set_xlabel("Training pairs")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Held-out preference accuracy")
    ax.legend(loc="lower left", frameon=False, handlelength=1.8)
    axes = bot.subplots(1, 4, sharey=True)
    for k, (axb, v) in enumerate(zip(axes, VARIANTS)):
        idx = gwin[f"mr_sweeps/sweep_antmaze_{v.replace('-', '_')}"]
        for arm, cells, col, ls, mfc, lab in (("with", std, CORAL, "--", "white", "with terminals"),
                                             ("without", cand, TEAL, "-", TEAL, "without terminals")):
            s = [stats(cells.get((v, "MR", lv))) for lv in LEVELS]
            log(f"  (b) {v:15s} {arm:7s} " + "  ".join(f"{TICK[lv]} {'n/a' if m is None else r1(m[0])}"
                                                         for lv, m in zip(LEVELS, s)))
            axb.plot(x, [np.nan if m is None else m[0] for m in s], color=col, ls=ls, marker="o", mfc=mfc,
                     mec=col, mew=1.0, label=lab)
        cand_idx = {cand[(v, "MR", lv)]["index"] for lv in LEVELS[1:] if (v, "MR", lv) in cand}
        if cand_idx - {idx}:
            log(f"  WARNING: {v} MR candidate cells at index {sorted(cand_idx)}, grid winner {idx}")
        axb.set_title(f"{v} (index {idx})", fontsize=8.5)
        axb.set_xticks(x, [TICK[lv] for lv in LEVELS])
        axb.set_ylim(0, 75)
        if k == 0:
            axb.set_ylabel("Score")
    axes[-1].legend(loc="upper right", frameon=False, handlelength=2.2)
    bot.suptitle("(b) Policy score (% of episodes reaching the goal)", fontsize=9)
    fig.savefig(path, metadata={"Creator": "iqlpref results/paper/render_prelim.py", "CreationDate": None})
    plt.close(fig)


# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------
def header(snap):
    m = snap["meta"]
    return (f"% Generated by iqlpref results/paper/render_prelim.py from snapshot_prelim.json (extracted "
            f"{m['extracted_utc']}, iqlpref {m['iqlpref']['commit'][:7]}). Do not edit by hand.\n")


def trivial_table(anchor, ctrl, log):
    """With terminals, full data: the trivial rewards (control_readout H), the task reward at
    index 1 and MR (the anchor, control_readout A)."""
    rows = []
    for label, get in (("Zero everywhere", lambda v: ctrl.get((v, "trivial_zero"))),
                       ("Dataset-average constant", lambda v: ctrl.get((v, "trivial_mean"))),
                       ("Task reward, $-1$ per step", lambda v: task_r1(anchor, v, log)),
                       ("MR", lambda v: anchor.get((f"antmaze-{v}-v2", "MR")))):
        cells = [get(v) for v in VARIANTS]
        log(f"  {label:27s} " + "  ".join(pm(c) for c in cells))
        rows.append(f"    {label} & " + " & ".join(pm(c) for c in cells) + r" \\")
    return ("\\begin{tabular}{@{}l r r r r@{}}\n  \\toprule\n  Reward & " + " & ".join(VARIANTS) +
            " \\\\\n  \\midrule\n" + "\n".join(rows) + "\n  \\bottomrule\n\\end{tabular}\n")


def task_r1(anchor, v, log):
    """The task reward at index 1 (r - 1: -1 per step): the anchor where stage 4 chose index 1,
    else the oracle's index-1 cell (eval_readout's task_reward_r1)."""
    ds = f"antmaze-{v}-v2"
    c = anchor.get((ds, "task_reward"))
    if c is not None and c["index"] == 1:
        return c
    r1 = anchor.get((ds, "task_reward_r1"))
    log(f"  note: {v} task reward: stage 4 chose index {None if c is None else c['index']}; the table uses the "
        f"index-1 cell {None if r1 is None else r1['sweep']}")
    return r1


def leak_table(ctrl, gwin, winners, log):
    """large-diverse without terminals, full data (control_readout I): each reward at its
    selected index as run (trajectories cut at the terminals) and leak-free (cut every
    1,000 steps only)."""
    v = LEAK_VARIANT
    rows = []
    for label, orig, lf in (("Movement heuristic", ctrl.get((v, "atwin_heuristic")), ctrl.get((v, "leakfree_heuristic"))),
                            ("MR", C.at_winner(ctrl, gwin, winners, v, "MR"), ctrl.get((v, "leakfree_MR")))):
        idx = orig["index"] if orig else None
        if lf is not None and lf["index"] != idx:
            log(f"  WARNING: {label}: as-run index {idx}, leak-free index {lf['index']}")
        for c, want in ((orig, "terminals"), (lf, "length")):
            cuts = {r["cut"] for r in (c or {}).get("runs", {}).values()}
            if c is not None and cuts != {want}:
                log(f"  WARNING: {label} {c['sweep']}: trajectory cut {sorted(cuts)}, expected {want!r}")
        log(f"  {label:20s} index {idx}  as run {pm(orig)}  leak-free {pm(lf)}")
        rows.append(f"    {label} & {idx} & {pm(orig)} & {pm(lf)} \\\\")
    return ("\\begin{tabular}{@{}l c r r@{}}\n  \\toprule\n  Reward & Index & As run & Leak-free \\\\\n"
            "  \\midrule\n" + "\n".join(rows) + "\n  \\bottomrule\n\\end{tabular}\n")


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:#.2g}"


def fmt_test(t, stacked):
    """The pooled effect x100 with its exact stratified 95% interval and the one-sided exact
    Wilcoxon p; bold when the condition did not reject."""
    e = t["e"]
    eff = f"{sr1(100 * e['mean'])}\\ [{sr1(100 * e['lower'])}, {sr1(100 * e['upper'])}]"
    p = fmt_p(t["t"]["p"])
    if not t["rejects"]:
        eff, p = f"\\bm{{{eff}}}", f"\\bm{{{p}}}"
    if stacked:
        return f"\\begin{{tabular}}[t]{{@{{}}l@{{}}}}${eff}$\\\\ ${p}$\\end{{tabular}}"
    return f"${eff}$, ${p}$"


def verdict_text(r):
    """not sensitive / sensitive / flagged, with the failed conditions."""
    if r["verdict"].startswith("FLAGGED"):
        return "flagged (manipulation check)"
    if "a" not in r:
        return "not sensitive, no usable dataset"
    failed = [tag for tag, ok in (("(a)", r["a"]["rejects"]), ("(b)", r["b"]["rejects"]), ("(c)", r["c"]["rejects"]))
              if not ok]
    return "sensitive" if not failed else "not sensitive, " + ", ".join(failed)


VERDICT_ROWS = (("Accuracy falls (manipulation check)", lambda r: r["manip"]),
                ("(a) Score falls without terminals", lambda r: r.get("a")),
                ("(b) Falls more than with terminals", lambda r: r.get("b")),
                ("(c) Beats zero reward", lambda r: (r.get("c") or {}).get("zero")),
                ("(c) Beats constant reward", lambda r: (r.get("c") or {}).get("mean")))


def verdicts_table(res, fams, log, stacked):
    if stacked:
        # four families: the row labels wrap, ragged right, and the gaps narrow, so the table
        # fits 6.5 in at \\small.  \\raggedright redefines \\\\ inside a p column, so rows end
        # with \\tabularnewline (no array package needed)
        spec = "@{}p{2.8cm}" + "".join("@{\\hspace{5pt}}l" for _ in fams) + "@{}"
        lab, end = "\\raggedright ", " \\tabularnewline"
    else:
        spec = "@{}l " + " ".join("l" for _ in fams) + "@{}"
        lab, end = "", r" \\"
    lines = [f"\\begin{{tabular}}{{{spec}}}", "  \\toprule",
             f"  {lab}Check or condition & " + " & ".join(FAMILY_LABEL[f] for f in fams) + end, "  \\midrule"]
    for label, get in VERDICT_ROWS:
        cells = []
        for f in fams:
            t = get(res[f])
            cells.append("n/a" if t is None else fmt_test(t, stacked))
        lines.append(f"  {lab}{label} & " + " & ".join(cells) + end + ("[2pt]" if stacked else ""))
    lines += ["  \\midrule", f"  {lab}Verdict & " + " & ".join(verdict_text(res[f]) for f in fams) + end,
              "  \\bottomrule", "\\end{tabular}"]
    for label, get in VERDICT_ROWS:
        log(f"  {label:37s} " + " | ".join(
            f"{f} " + ("n/a" if get(res[f]) is None else
                       f"{sr1(100 * get(res[f])['e']['mean'])} [{sr1(100 * get(res[f])['e']['lower'])}, "
                       f"{sr1(100 * get(res[f])['e']['upper'])}] {fmt_p(get(res[f])['t']['p'])}"
                       f"{'' if get(res[f])['rejects'] else ' (FAILED)'}") for f in fams))
    log("  Verdict " + " | ".join(f"{f}: {verdict_text(res[f])}" for f in fams))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default=SNAPSHOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-copy", action="store_true", help="do not also write results/paper/out/")
    a = ap.parse_args()
    snap = json.load(open(a.snapshot))
    m = snap["meta"]
    print(f"snapshot {os.path.relpath(a.snapshot)}: extracted {m['extracted_utc']} from {m['wandb']}, "
          f"iqlpref {m['iqlpref']['commit'][:7]}{' (dirty)' if m['iqlpref']['dirty'] else ''}")
    groups, acc = rebuild(snap)
    winners, gwin = snap["winners"], snap["grid_winners"]
    std, cand, full = Q.assemble(groups["standard"], groups["candidate"], groups["anchor"], groups["ctrl"],
                                 winners, gwin)
    res, use, uwhy, _ = Q.run(m["axis"], std, cand, full, groups["ctrl"], acc)
    if uwhy:
        raise SystemExit("usability check not read: " + "; ".join(uwhy))
    read = [f for f in Q.FAMILIES if (res.get(f) or {}).get("status") == "READ"]
    os.makedirs(a.out, exist_ok=True)
    out = {}

    def log(s):
        print(s)

    print("\nfig_prelim_mr_ladder.pdf")
    mr_ladder(std, cand, acc, gwin, os.path.join(a.out, "fig_prelim_mr_ladder.pdf"), log)
    print("\ntab_prelim_trivial.tex")
    out["tab_prelim_trivial.tex"] = trivial_table(groups["anchor"], groups["ctrl"], log)
    print(f"\ntab_prelim_verdicts.tex  (families read: {', '.join(read)})")
    out["tab_prelim_verdicts.tex"] = verdicts_table(res, read, log, stacked=len(read) > 2)
    out["tab_prelim_verdicts_mr_pt.tex"] = verdicts_table(res, [f for f in ("MR", "PT") if f in read],
                                                          lambda s: None, stacked=False)
    print(f"\ntab_prelim_leak.tex  ({LEAK_VARIANT})")
    out["tab_prelim_leak.tex"] = leak_table(groups["ctrl"], gwin, winners, log)
    for name, body in out.items():
        with open(os.path.join(a.out, name), "w") as f:
            f.write(header(snap) + body)
    names = ["fig_prelim_mr_ladder.pdf"] + list(out)
    if not a.no_copy and os.path.abspath(a.out) != os.path.abspath(COPY):
        os.makedirs(COPY, exist_ok=True)
        for n in names:
            shutil.copy2(os.path.join(a.out, n), os.path.join(COPY, n))
    print(f"\nwrote {', '.join(names)} to {a.out}" + ("" if a.no_copy else f" (copy in {os.path.relpath(COPY)})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
