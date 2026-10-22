#!/usr/bin/env python
"""control_readout.py -- the reading of the IQL controls (handoff §13.13-§13.14).

Written 2026-10-16, before any control run had finished.  Every number is the
§12.5 machinery unchanged: last-10 statistic, completeness gate (ten seeds x 200
points), config audit, and the exact paired bootstrap over seeds with the
paired-t interval beside it.  DESCRIPTIVE: no margin and no verdict.  What the
controls imply for the research direction is the user's call (§13.14).

THE QUESTIONS, fixed before the results
  A. What do the terminals carry?  Within a family, paired by seed:
         anchor (with terminals) - no terminals            MR, BNN CVaR
  B. Do the reward models separate once the terminals are gone?
         BNN CVaR - MR, without terminals, beside the same difference WITH
         terminals (the §12 cell)
  B2. Without terminals, does a learned reward carry more goal information
         than the task reward?  learned - task reward, both without terminals
         (the task-reward control was added 2026-10-17, still before any
         control result; A covers it too)
  C. Do the learned rewards beat motion shaping?  With terminals, paired by seed:
         each anchor method - the path-length heuristic
     The task reward is in C too, at its stage-4 winning index.
  D. The cells: each one's own mean ± sd.
  E. The no-terminals normalization grids (added 2026-10-17, §13.14.5, before
     any grid run existed): seed 0, every index, scored and selected by the
     stage-4 rule (iql_score.select_normalization: last-10, argmax, ties to the
     lowest index), beside the index the seeds 1-10 control used.  Where they
     differ, the seeds 1-10 cell is re-run at the grid's winner.

THE HOLD (HANDOFF §14.1, 2026-10-20): covered families (BNN CVaR, BNN mean,
ensemble CVaR) are dropped at collection and from every section; section B,
which is BNN CVaR - MR throughout, is not printed.  --include-held restores
them, only once the author has lifted the hold.

Run from the repo root:
    python results/control_readout.py
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
ANCHOR_METHODS = ("task_reward", "MR", "PT", "NMR", "ENS_MEAN", "ENS_CVAR", "BNN_MEAN", "BNN_CVAR")


def _short(dataset):
    return dataset.replace("antmaze-", "").replace("-v2", "")


def collect_controls(api, P, winners, gwin=None):
    """{(variant, kind): cell}, kind in noterm_MR, noterm_BNN_CVAR, heuristic."""
    import iql_score
    sp = P.spec()
    cspec = P.control_spec(sp)
    atw = P.control_atwin_spec(sp, gwin or {})          # §14.16: grid-only families' cells
    atw.update(P.control_ties_spec(sp, gwin or {}))     # §14.30: the ties arm without terminals
    fixed_of = {n: f for n, (_, f, _) in cspec.items()}
    fixed_of.update({n: f for n, (_, f, _, _, _) in atw.items()})
    cells, notes = {}, []
    for sw in api.project(P.IQL_PROJECT, entity=P.ENTITY).sweeps():
        params = sw.config.get("parameters")
        hits = P.match_entry(params, fixed_of)
        if len(hits) != 1:
            continue
        name = hits[0]
        flat = P.flat(params)
        if flat.get("seed") == 0 and isinstance(flat.get("normalize_reward"), list):
            continue                                   # a no-terminals grid: section E reads it
        if name in atw:                                # PT / ensemble mean / motion at the winner
            anchor, fixed, index, vv, fam = atw[name]
            if flat.get("seed") != P.EVAL_SEEDS or flat.get("normalize_reward") != index:
                notes.append(f"{sw.id}: {name} at index {flat.get('normalize_reward')} "
                             f"(winner {index}) -- ignored")
                continue
            v = vv.replace("_", "-")
            pre = "tiesnt_" if "/ties_noterm_" in name else "atwin_"
            kind = pre + ("heuristic" if fam == "heuristic" else P.method_of(anchor))
            no_model = fam == "heuristic"
        else:
            anchor, fixed, index = cspec[name]
            no_model = anchor is None or anchor.startswith("tr_")   # heuristic, task reward
        index = winners.get(anchor) if (anchor and name not in atw) else index
        prefix = "noterm_"
        if name in atw:
            pass
        elif (anchor and flat.get("seed") == P.EVAL_SEEDS and gwin and gwin.get(anchor) is not None
                and flat.get("normalize_reward") == gwin[anchor] != index):
            prefix, index = "rerun_", gwin[anchor]      # the re-run at the grid winner (§13.14.7)
        elif flat.get("seed") != P.EVAL_SEEDS or flat.get("normalize_reward") != index:
            notes.append(f"{sw.id}: {name} at seed {flat.get('seed')} index "
                         f"{flat.get('normalize_reward')} (expected {index}) -- ignored")
            continue
        if name not in atw:
            v = _short(P.dataset_of(anchor)) if anchor else name.split("heuristic_antmaze_")[1].replace("_", "-")
            kind = (prefix + P.method_of(anchor)) if anchor else "heuristic"
        if not hold.shown_kind(kind):
            continue                                   # covered by the hold (§14.1): not fetched
        cell = {"sweep": sw.id, "index": index, "scores": {}, "robust": {}, "max": {},
                "incomplete": [], "audit": [], "name": name}
        if sw.config.get("program") != P.DEFAULT_PROGRAM:
            cell["audit"].append(f"program {sw.config.get('program')!r}")
        seen = set()
        for r in sw.runs:
            s = (r.config or {}).get("seed")
            if s in seen:
                cell["audit"].append(f"seed {s} appears twice")
            seen.add(s)
            cell["audit"] += [f"run {r.id}: {x}" for x in
                              R.audit_run(r.config or {}, name, s, fixed, index, no_model)]
            if r.state != "finished":
                cell["incomplete"].append(f"seed {s} {r.state}")
                continue
            try:
                _, curve = iql_score.run_curve(r)
                cell["scores"][s] = iql_score.score_curve(curve, n=iql_score.LAST_N)
                cell["robust"][s] = iql_score.score_curve(curve, n=iql_score.ROBUST_N)
                cell["max"][s] = iql_score.score_curve(curve, statistic="max")   # §14.2 item 4
            except (iql_score.IncompleteRun, ValueError) as e:
                cell["incomplete"].append(f"seed {s}: {e}")
        missing = [s for s in SEEDS if s not in cell["scores"] and
                   not any(x.startswith(f"seed {s}") for x in cell["incomplete"])]
        if missing:
            cell["incomplete"].append(f"seeds not started: {missing}")
        key = (v, kind)
        if key in cells:
            notes.append(f"{key}: two sweeps ({cells[key]['sweep']}, {sw.id}) -- cell NOT read")
            cells[key]["audit"].append(f"second sweep {sw.id}")
            continue
        cells[key] = cell
    return cells, notes


def collect_grids(api, P):
    """{(variant, kind): {"sweep", "anchor", "grid", "runs": [score_run dicts], "notes"}}."""
    import iql_score
    sp = P.spec()
    gspec = P.control_grid_spec(sp)
    fixed_of = {n: f for n, (_, f) in gspec.items()}
    grids = {}
    for sw in api.project(P.IQL_PROJECT, entity=P.ENTITY).sweeps():
        params = sw.config.get("parameters")
        hits = P.match_entry(params, fixed_of)
        if len(hits) != 1:
            continue
        flat = P.flat(params)
        anchor, fixed = gspec[hits[0]]
        if flat.get("seed") != 0 or flat.get("normalize_reward") != list(P.grid_of(anchor)):
            continue                                   # a seeds 1-10 control, not a grid
        heur = P.is_heuristic_anchor(anchor)
        key = ((anchor.split("heuristic_antmaze_")[1].replace("_", "-"), "noterm_heuristic") if heur
               else (_short(P.dataset_of(anchor)), "noterm_" + P.method_of(anchor)))
        if not hold.shown_kind(key[1]):
            continue                                   # covered by the hold (§14.1)
        g = {"sweep": sw.id, "anchor": anchor, "grid": list(P.grid_of(anchor)), "runs": [],
             "notes": []}
        if heur:
            g["used"] = P.HEURISTIC_INDEX              # its with-terminals index, fixed in advance
        for r in sw.runs:
            bad = R.audit_run(r.config or {}, hits[0], 0, fixed, (r.config or {}).get("normalize_reward"),
                              anchor.startswith("tr_") or heur)
            if bad:
                g["notes"].append(f"run {r.id}: AUDIT {bad[0]}")
            g["runs"].append(iql_score.score_run(r))
        if key in grids:
            grids[key]["notes"].append(f"second grid sweep {sw.id} -- NOT read")
            continue
        grids[key] = g
    return grids


# Grids added 2026-10-21 (§14.2 item 5) with no anchor-index no-terminals cell:
# their seeds 1-10 cell runs at the grid winner directly.
GRID_ONLY = ("noterm_PT", "noterm_NMR", "noterm_ENS_MEAN", "noterm_heuristic")   # heuristic: §14.7; NMR: §14.17


def grid_section(grids, winners, ctrl, out):
    """Prints section E; returns {anchor: winner} for every COMPLETE grid."""
    import iql_score
    found = {}
    out("\nE. NO-TERMINALS NORMALIZATION GRIDS  (seed 0; stage-4 rule; the index the seeds 1-10 control used)")
    for v in VARIANTS:
        for kind in [k for k in ("noterm_task_reward", "noterm_MR", "noterm_PT", "noterm_NMR", "noterm_ENS_MEAN",
                                 "noterm_heuristic", "noterm_BNN_MEAN", "noterm_BNN_CVAR")
                     if hold.shown_kind(k)]:
            g = grids.get((v, kind))
            tag = f"  {v:15s} {kind:19s}"
            if g is None:
                out(f"{tag} no grid sweep")
                continue
            used = g.get("used", winners.get(g["anchor"]))
            if g["notes"]:
                out(f"{tag} {g['sweep']}  NOT READ -- {g['notes'][0]}")
                continue
            try:
                win, table = iql_score.select_normalization(g["runs"], g["grid"])
            except ValueError as e:
                done = sum(1 for s in g["runs"] if s.get("score") is not None)
                out(f"{tag} {g['sweep']}  INCOMPLETE ({done}/{len(g['grid'])} scored): {e}")
                continue
            found[g["anchor"]] = win
            by = {s["normalize_reward"]: s["score"] for s in table}
            row = " ".join(f"{by[i]:.3f}" for i in g["grid"])
            if kind in GRID_ONLY:                      # no seeds 1-10 cell yet: it runs at the winner
                verdict = f"anchor index {used}; seeds 1-10 run at {win}"
            else:
                verdict = "same as used" if win == used else f"DIFFERS from {used}: re-run seeds 1-10 at {win}"
            out(f"{tag} {g['sweep']}  idx {g['grid'][0]}-{g['grid'][-1]}: {row}   winner {win}  "
                f"({verdict})")
    return found


_ANCHOR = {"MR": "mr_sweeps/sweep_antmaze_{u}", "BNN_CVAR": "bnn_sweeps/sweep_antmaze_{u}_cvar",
           "BNN_MEAN": "bnn_sweeps/sweep_antmaze_{u}_mean", "task_reward": "tr_sweeps/sweep_antmaze_{u}"}


def at_winner(ctrl, gwin, winners, v, m):
    """The no-terminals cell at its grid winner: the re-run where the winner moved,
    the original control where it did not, None while the grid is not complete."""
    a = _ANCHOR[m].format(u=v.replace("-", "_"))
    if a not in gwin:
        return None
    return ctrl.get((v, f"noterm_{m}")) if gwin[a] == winners.get(a) else ctrl.get((v, f"rerun_{m}"))


def rerun_section(ctrl, A, gwin, winners, out):
    out("\nF. AT THE NO-TERMINALS WINNER  (each reward at its own no-terminals grid winner; §13.14.7)")
    for v in VARIANTS:
        for m in hold.show(("task_reward", "MR", "BNN_MEAN", "BNN_CVAR")):
            line(A[v][m], at_winner(ctrl, gwin, winners, v, m), out,
                 f"  {v:15s} {m:9s} with terminals - without, at its winner")
        # §14.16: grid-only families, whose first no-terminals cell is at the winner
        for m in ("PT", "NMR", "ENS_MEAN"):
            if (v, f"atwin_{m}") in ctrl:
                line(A[v].get(m), ctrl[(v, f"atwin_{m}")], out,
                     f"  {v:15s} {m:9s} with terminals - without, at its winner")
        if (v, "atwin_heuristic") in ctrl:
            line(ctrl.get((v, "heuristic")), ctrl[(v, "atwin_heuristic")], out,
                 f"  {v:15s} motion    with terminals - without, at its winner")
        if not hold.shown("BNN_CVAR"):
            continue
        line(at_winner(ctrl, gwin, winners, v, "BNN_CVAR"), at_winner(ctrl, gwin, winners, v, "MR"), out,
             f"  {v:15s} BNN CVaR - MR, no terminals, each at its winner")
        line(at_winner(ctrl, gwin, winners, v, "BNN_CVAR"), at_winner(ctrl, gwin, winners, v, "BNN_MEAN"), out,
             f"  {v:15s} BNN CVaR - BNN mean, no terminals, each at its winner")


def ties_section(ctrl, gwin, winners, out):
    """G (§14.30): the ties arm against the tie-free arm, both WITHOUT terminals at the
    family's frozen no-terminals index, paired by seed."""
    out("\nG. TIES UNDER THE CANDIDATE PROTOCOL  (ties-kept - tie-free models, both without terminals, same index)")
    for v in ("medium-diverse", "large-play"):
        for m in ("MR", "PT", "NMR"):
            t = ctrl.get((v, f"tiesnt_{m}"))
            if t is None:
                out(f"  {v:15s} {m:4s} no ties sweep yet")
                continue
            base = at_winner(ctrl, gwin, winners, v, m) if m == "MR" else ctrl.get((v, f"atwin_{m}"))
            line(t, base, out, f"  {v:15s} {m:4s} ties - tie-free, no terminals, idx {t['index']}")


def line(a, b, out, tag):
    ra, rb = R.cell_ready(a), R.cell_ready(b)
    if ra or rb:
        out(f"{tag} NOT READ -- " + "; ".join(x for x in (ra, rb) if x))
    else:
        out(f"{tag} {R.fmt_cmp(R.compare(a, b))}")


def report(ctrl, anchor, notes, out=print, grids=None, winners=None):
    out("=" * 100)
    out("IQL CONTROLS (handoff §13.13-§13.14): terminals, and motion shaping.  DESCRIPTIVE.")
    out("last-10 statistic; paired by seed; exact bootstrap over seeds, paired-t beside it")
    out("=" * 100)
    if not hold.INCLUDE:
        out("  BNN/CVaR HOLD (HANDOFF §14.1): BNN CVaR, BNN mean and ensemble CVaR are not shown; "
            "section B is omitted")
    for n in notes:
        if hold.INCLUDE or not any(x in n for x in ("BNN", "ENS_CVAR", "bnn", "ensemble CVaR")):
            out(f"  note: {n}")
    A = {v: {m: anchor.get((f"antmaze-{v}-v2", m)) for m in hold.show(ANCHOR_METHODS)} for v in VARIANTS}

    out("\nA. WHAT THE TERMINALS CARRY  (anchor, with terminals - the same reward without them)")
    for v in VARIANTS:
        for m in hold.show(("task_reward", "MR", "BNN_MEAN", "BNN_CVAR")):
            line(A[v][m], ctrl.get((v, f"noterm_{m}")), out,
                 f"  {v:15s} {m:9s} with - without terminals   ")

    out("\nB. DO THE REWARDS SEPARATE WITHOUT TERMINALS?  (BNN CVaR - MR)"
        + ("" if hold.INCLUDE else "  -- omitted: covered by the hold (§14.1)"))
    for v in (VARIANTS if hold.INCLUDE else ()):
        line(ctrl.get((v, "noterm_BNN_CVAR")), ctrl.get((v, "noterm_MR")), out,
             f"  {v:15s} without terminals              ")
        line(A[v]["BNN_CVAR"], A[v]["MR"], out,
             f"  {v:15s} with terminals (the §12 cell)  ")
        line(ctrl.get((v, "noterm_BNN_CVAR")), ctrl.get((v, "noterm_BNN_MEAN")), out,
             f"  {v:15s} BNN CVaR - BNN mean, no terminals")

    out("\nB2. WITHOUT TERMINALS, DOES A LEARNED REWARD CARRY MORE GOAL INFORMATION THAN THE TASK REWARD?"
        "  (learned - task reward, both without terminals)")
    for v in VARIANTS:
        for m in hold.show(("MR", "BNN_MEAN", "BNN_CVAR")):
            line(ctrl.get((v, f"noterm_{m}")), ctrl.get((v, "noterm_task_reward")), out,
                 f"  {v:15s} {m:9s} - task reward, no terminals")

    out("\nC. LEARNED REWARDS AGAINST MOTION SHAPING  (each method - path-length heuristic, both with terminals)")
    for v in VARIANTS:
        h = ctrl.get((v, "heuristic"))
        for m in hold.show(ANCHOR_METHODS):
            line(A[v][m], h, out, f"  {v:15s} {m:11s} - heuristic          ")

    out("\nD. THE CONTROL CELLS  (last-10 mean ± sd over seeds 1-10)")
    for v in VARIANTS:
        for kind in [k for k in ("noterm_task_reward", "noterm_MR", "noterm_BNN_MEAN", "noterm_BNN_CVAR",
                                 "heuristic") if hold.shown_kind(k)]:
            c = ctrl.get((v, kind))
            r = R.cell_ready(c)
            if r:
                out(f"  {v:15s} {kind:16s} {'no sweep' if c is None else c['sweep'] + '  ' + r}")
            else:
                x = np.array([c["scores"][s] for s in SEEDS])
                out(f"  {v:15s} {kind:16s} {c['sweep']}  idx {c['index']}  "
                    f"{x.mean():.3f} ± {x.std(ddof=1):.3f}")
    found = {}
    if grids is not None:
        found = grid_section(grids, winners or {}, ctrl, out)
        rerun_section(ctrl, A, found, winners or {}, out)
        ties_section(ctrl, found, winners or {}, out)
        for v in VARIANTS:
            for k in [f"rerun_{m}" for m in hold.show(("MR", "BNN_MEAN", "BNN_CVAR", "task_reward"))] + \
                    ["atwin_PT", "atwin_NMR", "atwin_ENS_MEAN", "atwin_heuristic"]:
                c = ctrl.get((v, k))
                if c is not None:
                    r = R.cell_ready(c)
                    out(f"  {v:15s} {k:18s} {c['sweep']}  idx {c['index']}  "
                        + (r if r else "{:.3f} ± {:.3f}".format(
                            *(lambda x: (x.mean(), x.std(ddof=1)))(np.array([c["scores"][s] for s in SEEDS])))))
    return found


def selftest():
    hold.INCLUDE = True
    rng = np.random.default_rng(0)

    def cell(vals):
        return {"scores": dict(zip(SEEDS, vals)), "robust": dict(zip(SEEDS, vals)),
                "incomplete": [], "audit": [], "sweep": "s", "index": 3, "name": "n"}
    anchor, ctrl = {}, {}
    for v in VARIANTS:
        base = rng.uniform(0.3, 0.6, 10)
        for m in ANCHOR_METHODS:
            anchor[(f"antmaze-{v}-v2", m)] = cell(base + 0.05)
        ctrl[(v, "noterm_MR")] = cell(base - 0.2)
        ctrl[(v, "noterm_BNN_CVAR")] = cell(base - 0.1)
        ctrl[(v, "heuristic")] = cell(base)
        ctrl[(v, "noterm_task_reward")] = cell(base - 0.3)
    ctrl[("large-play", "heuristic")]["incomplete"] = ["seed 4 running"]
    lines = []
    report(ctrl, anchor, [], out=lines.append)
    txt = "\n".join(lines)
    assert "medium-play     MR        with - without terminals    " in txt
    assert txt.count("diff +0.250") == 4 and txt.count("diff +0.150") == 4      # A, learned
    assert txt.count("diff +0.350") == 4                                        # A, task reward
    assert "medium-play     MR        - task reward, no terminals diff" not in txt
    assert txt.count("- task reward, no terminals") == 12                      # B2: 4 variants x 3
    assert txt.count("diff +0.100") == 8                    # B without (4) + B2 MR (4)
    assert txt.count("diff +0.200") == 4                    # B2 BNN CVaR
    assert "large-play      MR          - heuristic           NOT READ" in txt
    assert txt.count("diff +0.050") == 24                                       # C, 3 variants x 8 (NMR: 14.17)
    # --- section E: winner by the stage-4 rule, ties to the lowest index ---
    def runs(scores):
        return [{"id": f"r{i}", "normalize_reward": i, "score": s, "error": None}
                for i, s in enumerate(scores)]
    grids = {("medium-play", "noterm_BNN_CVAR"): {"sweep": "g1", "anchor": "bnn_sweeps/a", "grid": list(range(8)),
                                                  "runs": runs([.1, .2, .6, .6, .0, .0, .3, .3]), "notes": []},
             ("medium-play", "noterm_MR"): {"sweep": "g2", "anchor": "mr_sweeps/a", "grid": list(range(8)),
                                            "runs": runs([.5, .5, .4, .4, .0, .0, .2, .2]), "notes": []},
             ("large-play", "noterm_MR"): {"sweep": "g3", "anchor": "mr_sweeps/b", "grid": list(range(8)),
                                           "runs": runs([.5, .5, .4]), "notes": []}}
    lines = []
    grid_section(grids, {"bnn_sweeps/a": 2, "mr_sweeps/a": 3, "mr_sweeps/b": 3}, {}, lines.append)
    txt = "\n".join(lines)
    assert "winner 2  (same as used)" in txt                     # tie 2/3 -> 2
    assert "winner 0  (DIFFERS from 3: re-run seeds 1-10 at 0)" in txt
    assert "INCOMPLETE (3/8 scored)" in txt
    lines = []
    grid_section({("medium-play", "noterm_PT"): {"sweep": "g4", "anchor": "pt_sweeps/a", "grid": list(range(8)),
                                                "runs": runs([.1, .2, .3, .7, .0, .0, .2, .2]), "notes": []}},
                 {"pt_sweeps/a": 0}, {}, lines.append)
    assert any("noterm_PT" in x and "winner 3  (anchor index 0; seeds 1-10 run at 3)" in x for x in lines), lines
    lines = []
    grid_section({("large-play", "noterm_heuristic"): {"sweep": "g5", "anchor": "ctrl_sweeps/heuristic_antmaze_large_play",
                                                      "grid": list(range(8)), "used": 3, "notes": [],
                                                      "runs": runs([.2, .1, .1, .1, .0, .0, .1, .1])}},
                 {}, {}, lines.append)
    assert any("noterm_heuristic" in x and "winner 0  (anchor index 3; seeds 1-10 run at 0)" in x for x in lines), lines
    # --- section F: the re-run where the winner moved, the original where it did not ---
    w = {"mr_sweeps/sweep_antmaze_medium_play": 0, "bnn_sweeps/sweep_antmaze_medium_play_cvar": 2}
    g = {"mr_sweeps/sweep_antmaze_medium_play": 1, "bnn_sweeps/sweep_antmaze_medium_play_cvar": 2}
    base = rng.uniform(0.3, 0.6, 10)
    ct = {("medium-play", "noterm_MR"): cell(base), ("medium-play", "rerun_MR"): cell(base + 0.3),
          ("medium-play", "noterm_BNN_CVAR"): cell(base + 0.1)}
    assert at_winner(ct, g, w, "medium-play", "MR") is ct[("medium-play", "rerun_MR")]
    assert at_winner(ct, g, w, "medium-play", "BNN_CVAR") is ct[("medium-play", "noterm_BNN_CVAR")]
    assert at_winner(ct, g, w, "large-play", "MR") is None
    lines = []
    An = {v: {m: cell(base + 0.5) for m in ANCHOR_METHODS} for v in VARIANTS}
    rerun_section(ct, An, g, w, lines.append)
    txt = "\n".join(lines)
    fl = [x for x in txt.splitlines() if "medium-play     BNN CVaR - MR, no terminals, each at its winner" in x]
    assert len(fl) == 1 and "diff -0.200" in fl[0], txt                # re-run MR vs original BNN CVaR
    ct2 = {**ct, ("medium-play", "atwin_ENS_MEAN"): cell(base + 0.2)}
    lines = []
    rerun_section(ct2, An, g, w, lines.append)
    fl = [x for x in lines if "medium-play     ENS_MEAN  with terminals - without, at its winner" in x]
    assert len(fl) == 1 and "diff +0.300" in fl[0], lines             # anchor (base+0.5) - atwin (base+0.2)
    # --- the hold (§14.1): nothing covered is printed by default ---
    hold.INCLUDE = False
    lines = []
    report(ctrl, anchor, ["sweep x: BNN_CVAR cell", "sweep y: MR cell"], out=lines.append, grids=grids,
           winners={"bnn_sweeps/a": 2, "mr_sweeps/a": 3, "mr_sweeps/b": 3})
    rerun_section(ct, An, g, w, lines.append)
    txt = "\n".join(lines)
    body = [x for x in txt.splitlines() if "HOLD" not in x and "omitted" not in x]
    assert not any(k in x for x in body for k in ("BNN", "ENS_CVAR", "bnn_sweeps")), txt
    assert "sweep y: MR cell" in txt and "sweep x" not in txt
    assert txt.count("diff +0.250") == 4 and txt.count("diff +0.350") == 4    # A: MR and task reward stay
    assert "medium-play     MR        with terminals - without, at its winner" in txt
    assert txt.count("- heuristic") == 20                                     # C: 4 variants x 5 methods (NMR: 14.17)
    # --- section G: ties arm against the tie-free arm at the same index ---
    ctg = {("large-play", "tiesnt_MR"): {**cell(base + 0.25), "index": 6},
           ("large-play", "rerun_MR"): cell(base + 0.05)}
    lines = []
    ties_section(ctg, {"mr_sweeps/sweep_antmaze_large_play": 6}, {"mr_sweeps/sweep_antmaze_large_play": 3},
                 lines.append)
    fl = [x for x in lines if "large-play      MR   ties - tie-free, no terminals, idx 6" in x]
    assert len(fl) == 1 and "diff +0.200" in fl[0], lines
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
    winners = json.load(open(os.path.join(ROOT, "phase2_winners.json")))
    api = wandb.Api(timeout=120)
    anchor, anotes = R.collect(api, P, winners)
    gpath = os.path.join(ROOT, P.GRID_WINNERS)
    gwin = json.load(open(gpath)) if os.path.exists(gpath) else {}
    ctrl, notes = collect_controls(api, P, winners, gwin)
    grids = collect_grids(api, P)
    found = report(ctrl, anchor, notes + anotes, grids=grids, winners=winners)
    if "--write-grid-winners" in sys.argv:
        merged = dict(gwin)
        for k, w in found.items():
            if k in merged and merged[k] != w:
                raise SystemExit(f"!! grid winner for {k} changed {merged[k]} -> {w}: not written")
            merged[k] = w
        with open(gpath, "w") as f:
            json.dump(dict(sorted(merged.items())), f, indent=1)
        print(f"\n  {len(merged)} grid winner(s) in {P.GRID_WINNERS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
