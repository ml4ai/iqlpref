#!/usr/bin/env python
"""exp_readout.py -- the REGISTERED reading of Experiments 1 and 2 (H1).

Implements gp_reward-priors/HANDOFF_HP_SELECTION.md §13.9 (sparsity axis) and
§13.11 (label-noise axis), both fixed by the user on 2026-10-14 before any
result of either experiment existed, and Chapter 3's H1 and Statistical and
Reporting Protocol.  Nothing here is a free choice.  One verdict per axis; no
verdict pools the two (§13.11).

THE NOISE AXIS (§13.11) is the sparsity rule below with three substitutions:
  levels     flip fraction p in {0.1, 0.2, 0.3, 0.4, 0.45}, PRIMARY arm (training
             and test labels flipped, validation clean); the anchor is p = 0
  H1a        u = mean of d over the five flip fractions (anchor excluded)
  H1b        u = OLS slope of d on p ITSELF (linear) over all six points; H1
             predicts a POSITIVE slope (the gap grows as p rises)
  The sensitivity arm is descriptive only and is not read here.

Run with --axis sparsity (default) or --axis noise.

THE H3 GATE BESIDE EVERY VERDICT (§13.10, §13.12)
  Each (variant, level) unit's calibration gate is read from
  gp_reward-priors/calibration_results/measurements by calibration_gate.py.
  * A unit that is not PASS is CONDITIONAL.  Under §13.12 it is FLAGGED, NOT
    DROPPED: its cells enter the pooled tests unchanged, and the list of
    conditional cells is printed beside every verdict.
  * The gate is read FIRST (Chapter 3; §13.10: "before its BNN IQL results are
    read").  So while any unit of the axis has no gate result, no H1 verdict is
    printed, and the per-level curve shows no number for that unit.

THE RULE (§13.9)
  statistic    last-10 mean, results/iql_score.py, all 200 points (as §12)
  gap          d = BNN CVaR - comparator, per (variant, seed, level), PAIRED
  comparators  MR best-model, PT, BNN posterior mean -- Chapter 3's H1 names all
               three ("beats both point-estimate reward models and the BNN
               posterior mean").  Each is tested separately.
  levels       16, 32, 64, 128 pairs, and the full train-pool (the anchor)
  unit         one (variant, seed).  Seeds are the independent replicates; the
               levels within a seed share nested data and are NOT independent.
               The four variants' seed s are separate data draws, so the pooled
               test has 40 independent units.

  H1a, the gap     u = mean of d over the four SPARSE levels (16-128).  The
                   anchor is not in it: it was read by §12, as non-inferiority.
  H1b, widening    u = OLS slope of d on log2(pairs) over all five levels.  H1
                   says the gap grows as pairs fall, i.e. the slope is NEGATIVE.

  test         Wilcoxon signed-rank over the 40 units, ONE-SIDED in H1's
               direction (H1a: u > 0; H1b: u < 0), at alpha = 0.025.  Exact:
               zero differences dropped (Wilcoxon's convention), ties given
               midranks, and the p-value is the exact sign-flip distribution of
               the statistic given those ranks.  No normal approximation.
  effect size  the pooled mean of u, with a 95% bootstrap interval that
               resamples seeds WITHIN each variant, computed exactly: each
               variant's 92,378 count vectors, the four distributions convolved
               on an integer lattice.  Beside it, as a check, nothing else: the
               per-variant intervals are below and are descriptive.
  per variant  the same unit, mean and exact 10-seed interval.  DESCRIPTIVE: no
               test, no verdict.

  verdict, per comparator (Chapter 3's falsifier: "no gap, or a gap that does
  not widen")
               SUPPORTED       H1a rejects AND H1b rejects
               NOT SUPPORTED   otherwise, saying which part did not reject
  H1 as stated is the conjunction over the three comparators.  A conjunction of
  tests each at 0.025 is a valid 0.025 test of the conjunction (intersection-
  union), so no multiplicity correction is applied.  Partial support is
  reported as partial, never rounded up.

BEFORE ANY NUMBER
  The completeness gate and the config audit of §12.5, applied to every cell
  (variant x method x level), the anchor's included.  A comparator's H1 test is
  run only when ALL of its 40 cells and BNN CVaR's 40 cells are complete and
  clean.

Run from the repo root:
    python results/exp_readout.py
    python results/exp_readout.py --selftest
"""

import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import eval_readout as R                                   # noqa: E402  the §12 machinery

ALPHA = 0.025                       # one-sided, §13.9
COVERAGE = 0.95
SEEDS = R.SEEDS
PROPOSED = "BNN_CVAR"
COMPARATORS = ("MR", "PT", "BNN_MEAN")                      # Chapter 3, H1
DESCRIPTIVE = ("ENS_CVAR", "ENS_MEAN")
LEVELS = ("r16", "r32", "r64", "r128")
SPARSE_PAIRS = {"r16": 16, "r32": 32, "r64": 64, "r128": 128}
NOISE_LEVELS = ("n01", "n02", "n03", "n04", "n045")
FLIP = {"n01": 0.1, "n02": 0.2, "n03": 0.3, "n04": 0.4, "n045": 0.45}
FULL = "full"                                                # the anchor
VARIANTS = ("medium-play", "medium-diverse", "large-play", "large-diverse")
# The anchor's training set is the whole train-pool; identical for every seed of
# a variant (handoff §13.1).
TRAIN_POOL = {"medium-play": 358, "medium-diverse": 498, "large-play": 254, "large-diverse": 514}
SCORE_LATTICE = 0.001               # a last-10 mean of 100-episode success rates


def _x_sparsity(variant, level):
    return math.log2(TRAIN_POOL[variant] if level == FULL else SPARSE_PAIRS[level])


def _x_noise(variant, level):
    return 0.0 if level == FULL else FLIP[level]


# What differs between the two axes.  "widen" is H1b's one-sided direction.
AXES = {
    "sparsity": {"levels": LEVELS, "x": _x_sparsity, "widen": "less", "section": "§13.9",
                 "title": "EXPERIMENT 1 -- data sparsity",
                 "levels_text": "16, 32, 64, 128 pairs + the full train-pool",
                 "gap_text": "mean over 16-128",
                 "slope_text": "slope per doubling of pairs, < 0 widens"},
    "noise": {"levels": NOISE_LEVELS, "x": _x_noise, "widen": "greater", "section": "§13.11",
              "title": "EXPERIMENT 2 -- label noise, primary arm",
              "levels_text": "flip fraction 0.1, 0.2, 0.3, 0.4, 0.45 + the clean anchor (p = 0)",
              "gap_text": "mean over p 0.1-0.45",
              "slope_text": "slope per unit flip fraction, > 0 widens"},
}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def signed_rank_exact(u, alternative):
    """Exact one-sided Wilcoxon signed-rank test.

    Zeros dropped; ties midranked; the null distribution of W+ is the exact
    distribution over all 2^n sign assignments to the observed ranks (computed
    by dynamic programming on doubled ranks, which are integers).
    alternative "greater": H1 is u > 0, p = P(W+ >= w);  "less": p = P(W+ <= w).
    -> dict(n, n_zero, w_plus, p)
    """
    u = np.asarray(u, dtype=np.float64)
    nz = u[u != 0]
    n = nz.size
    if n == 0:
        return {"n": 0, "n_zero": int(u.size), "w_plus": 0.0, "p": 1.0}
    a = np.abs(nz)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n)
    i = 0
    while i < n:                                       # midranks over ties in |u|
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r2 = np.rint(2 * ranks).astype(np.int64)           # doubled ranks: integers
    w2 = int(r2[nz > 0].sum())
    dist = np.zeros(int(r2.sum()) + 1)
    dist[0] = 1.0
    for r in r2:                                       # each rank in or out of W+
        new = dist.copy()
        new[r:] += dist[:-r] if r else 0
        dist = new
    dist /= 2.0 ** n
    p = float(dist[w2:].sum()) if alternative == "greater" else float(dist[:w2 + 1].sum())
    return {"n": n, "n_zero": int(u.size - n), "w_plus": w2 / 2, "p": min(1.0, p)}


def group_pmf(values, delta):
    """Exact resampling distribution of the SUM of a 10-seed resample, on the
    integer lattice delta: (offset, pmf) with value = (offset + i) * delta.
    Also returns the largest rounding error, in units of delta."""
    v = np.asarray(values, dtype=np.float64)
    k = np.rint(v / delta).astype(np.int64)
    err = float(np.abs(v / delta - k).max())
    counts, probs = R.count_vectors(v.size)
    s = counts @ k
    lo = int(s.min())
    pmf = np.bincount(s - lo, weights=probs)
    return lo, pmf, err


def _convolve(a, b):
    n = a.size + b.size - 1
    m = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(a, m) * np.fft.rfft(b, m), m)[:n]
    return np.clip(out, 0.0, None)


def stratified_exact(groups, delta, coverage=COVERAGE):
    """Exact stratified bootstrap of the pooled mean of equal-size groups:
    seeds are resampled within each group, every group's 92,378 count vectors
    are enumerated, and the groups' sum distributions are convolved.
    -> dict(mean, lower, upper, prob_sum, mean_check, max_round_err)"""
    off, pmf, worst = 0, np.array([1.0]), 0.0
    n_per = None
    for g in groups:
        lo, p, err = group_pmf(g, delta)
        n_per = len(g) if n_per is None else n_per
        assert len(g) == n_per, "groups must be the same size"
        off += lo
        pmf = _convolve(pmf, p) if pmf.size > 1 or pmf[0] != 1.0 else p
        worst = max(worst, err)
    total_units = n_per * len(groups)
    vals = (off + np.arange(pmf.size)) * delta / total_units
    cum = np.cumsum(pmf)
    tol = 1e-10
    q_lo, q_hi = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    lower = float(vals[np.searchsorted(cum, q_lo - tol, side="left")])
    upper = float(vals[np.searchsorted(cum, q_hi - tol, side="left")])
    plain = float(np.mean([np.mean(g) for g in groups]))
    return {"mean": plain, "lower": lower, "upper": upper, "prob_sum": float(pmf.sum()),
            "mean_check": float((vals * pmf).sum()), "max_round_err": worst,
            "delta": delta}


def ols_slope(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def unit_delta(units, lattice):
    """The lattice the gap units sit on exactly, or a fine grid if they do not."""
    u = np.asarray(units, dtype=np.float64)
    if np.all(np.abs(u / lattice - np.rint(u / lattice)) < 1e-6):
        return lattice, True
    return 1e-6, False


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def _short(dataset):
    return dataset.replace("antmaze-", "").replace("-v2", "")


def collect_exp(api, P, winners, axis="sparsity", protocol="standard"):
    """{(variant, method, level): cell} for every experiment sweep at the
    anchor's stage-4 index.  Cells are built exactly as §12's are.

    protocol "candidate" (§14.5): the same experiment sweeps WITHOUT terminals --
    each experiment file plus drop_terminals -- and `winners` must then be the
    no-terminals grid winners (results/control_grid_winners.json), the candidate
    protocol's frozen indices.  Same reward models; only the protocol differs."""
    import iql_score
    assert protocol in ("standard", "candidate"), protocol
    sp = P.spec()
    E = P._conditions()
    by_params = {}
    for cond in AXES[axis]["levels"]:
        for name, (anchor, fixed) in P.exp_spec(sp, cond, E).items():
            if protocol == "candidate":
                fixed = dict(fixed, drop_terminals=True)
            by_params[name] = (cond, anchor, fixed)
    expsp = {n: f for n, (_, _, f) in by_params.items()}
    cells, notes = {}, []
    for sw in api.project(P.IQL_PROJECT, entity=P.ENTITY).sweeps():
        params = sw.config.get("parameters")
        hits = P.match_entry(params, expsp)
        if len(hits) != 1:
            continue
        name = hits[0]
        cond, anchor, fixed = by_params[name]
        flat = P.flat(params)
        index, win = flat.get("normalize_reward"), winners.get(anchor)
        if flat.get("seed") != P.EVAL_SEEDS or index != win:
            notes.append(f"{sw.id}: {name} seed {flat.get('seed')} index {index} "
                         f"(anchor winner {win}) -- not this experiment's lineage, ignored")
            continue
        if not R.hold.shown(P.method_of(anchor)):
            continue                                   # covered by the hold (§14.1): not fetched
        key = (_short(P.dataset_of(anchor)), P.method_of(anchor), cond)
        cell = {"sweep": sw.id, "index": index, "scores": {}, "robust": {}, "max": {},
                "incomplete": [], "audit": [], "name": name}
        if sw.config.get("program") != P.program_of(anchor):
            cell["audit"].append(f"program {sw.config.get('program')!r}")
        seen = set()
        for r in sw.runs:
            s = (r.config or {}).get("seed")
            if s in seen:
                cell["audit"].append(f"seed {s} appears twice")
            seen.add(s)
            cell["audit"] += [f"run {r.id}: {x}" for x in
                              R.audit_run(r.config or {}, name, s, fixed, index, False)]
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
        if key in cells:
            notes.append(f"{key}: two sweeps ({cells[key]['sweep']}, {sw.id}) -- cell NOT read")
            cells[key]["audit"].append(f"second sweep {sw.id} for this cell")
            continue
        cells[key] = cell
    return cells, notes


def add_anchor(cells, anchor_cells):
    """The anchor's cells as the FULL level."""
    for (dataset, method), c in anchor_cells.items():
        if method in (PROPOSED,) + COMPARATORS + DESCRIPTIVE:
            cells[(_short(dataset), method, FULL)] = c
    return cells


# --------------------------------------------------------------------------
# The reading
# --------------------------------------------------------------------------
def pairs_of(variant, level):
    return TRAIN_POOL[variant] if level == FULL else SPARSE_PAIRS[level]


def units_for(cells, comp, axis="sparsity"):
    """{variant: {"gap": [10], "slope": [10]}}, or the reason it cannot be read."""
    ax = AXES[axis]
    levels = ax["levels"]
    out = {}
    for v in VARIANTS:
        gap, slope = [], []
        for s in SEEDS:
            ds, xs = [], []
            for lv in levels + (FULL,):
                a, b = cells.get((v, PROPOSED, lv)), cells.get((v, comp, lv))
                why = R.cell_ready(a) or R.cell_ready(b)
                if why:
                    who = PROPOSED if R.cell_ready(a) else comp
                    return None, f"{v} {lv} {who}: {why}"
                ds.append(a["scores"][s] - b["scores"][s])
                xs.append(ax["x"](v, lv))
            gap.append(float(np.mean(ds[:len(levels)])))          # altered levels only
            slope.append(ols_slope(xs, ds))                        # all five levels
        out[v] = {"gap": gap, "slope": slope}
    return out, None


def h1_for(units, axis="sparsity"):
    gap = [x for v in VARIANTS for x in units[v]["gap"]]
    slope = [x for v in VARIANTS for x in units[v]["slope"]]
    t_gap = signed_rank_exact(gap, "greater")
    t_slope = signed_rank_exact(slope, AXES[axis]["widen"])
    dg, exact_g = unit_delta(gap, SCORE_LATTICE / len(AXES[axis]["levels"]))
    e_gap = stratified_exact([units[v]["gap"] for v in VARIANTS], dg)
    e_slope = stratified_exact([units[v]["slope"] for v in VARIANTS], 1e-6)
    a_ok, b_ok = t_gap["p"] < ALPHA, t_slope["p"] < ALPHA
    if a_ok and b_ok:
        v = "SUPPORTED"
    else:
        v = "NOT SUPPORTED (" + ", ".join(
            x for x, ok in (("no gap: H1a did not reject", a_ok),
                            ("no widening: H1b did not reject", b_ok)) if not ok) + ")"
    return {"t_gap": t_gap, "t_slope": t_slope, "e_gap": e_gap, "e_slope": e_slope,
            "gap_on_lattice": exact_g, "verdict": v}


GATE_DIR = os.path.join(ROOT, "gp_reward-priors", "calibration_results", "measurements")


def gate_status(axis, in_dir=GATE_DIR):
    """{(variant, level): verdict} for every unit of the axis, anchor included."""
    sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors", "scripts_bnn"))
    import calibration_gate as G
    out = {}
    for v in VARIANTS:
        vu = v.replace("-", "_")
        for lv in AXES[axis]["levels"] + (FULL,):
            cond = G.ANCHOR if lv == FULL else lv
            out[(v, lv)] = G.gate_unit(G.load_records(in_dir, cond, vu))["verdict"]
    return out


def gate_lists(gate):
    unread = [k for k, v in gate.items() if v.startswith("INCOMPLETE")]
    conditional = [k for k, v in gate.items() if v != "PASS" and k not in unread]
    return conditional, unread


def _units(keys):
    return ", ".join(f"{v} {lv}" for v, lv in keys) or "none"


def checks(e):
    bad = []
    if abs(e["prob_sum"] - 1.0) > 1e-9:
        bad.append(f"probabilities sum to {e['prob_sum']!r}")
    # the lattice mean equals the plain mean up to the rounding onto the lattice
    if abs(e["mean_check"] - e["mean"]) > e["delta"]:
        bad.append(f"weighted mean {e['mean_check']!r} != plain mean {e['mean']!r}")
    return bad


def report(cells, notes, out=print, axis="sparsity", gate=None):
    ax = AXES[axis]
    levels = ax["levels"]
    problems = 0
    gate = gate_status(axis) if gate is None else gate
    conditional, unread = gate_lists(gate)
    out("=" * 100)
    out(f"{ax['title']} READOUT ({ax['section']}): BNN CVaR against MR, PT and the BNN mean")
    out(f"levels {ax['levels_text']}; one-sided Wilcoxon at {ALPHA}, 40 (variant, seed) units; "
        f"exact stratified bootstrap.  One verdict for this axis; none pools the two axes.")
    out("=" * 100)
    for n in notes:
        out(f"  note: {n}")

    out("\nH3 GATE (§13.10), read first.  Conditional = not PASS; flagged, NOT dropped (§13.12).")
    out(f"   conditional units ({len(conditional)}): {_units(conditional)}")
    out(f"   gate not yet read ({len(unread)}): {_units(unread)}")

    out("\nA. H1, PER COMPARATOR  (SUPPORTED needs the gap AND its widening; Chapter 3's falsifier)")
    verdicts = {}
    for comp in COMPARATORS:
        if unread:
            out(f"  vs {comp:8s}  NOT READ -- the H3 gate is read first, and {len(unread)} "
                f"unit(s) have no gate result yet")
            verdicts[comp] = "NOT READ"
            continue
        units, why = units_for(cells, comp, axis)
        if units is None:
            out(f"  vs {comp:8s}  NOT READ -- {why}")
            problems += "AUDIT" in why
            verdicts[comp] = "NOT READ"
            continue
        h = h1_for(units, axis)
        bad = checks(h["e_gap"]) + checks(h["e_slope"])
        if bad:
            out(f"  vs {comp:8s}  SELF-CHECK FAILED: {bad}")
            problems += 1
            verdicts[comp] = "SELF-CHECK FAILED"
            continue
        verdicts[comp] = h["verdict"]
        tg, ts, eg, es = h["t_gap"], h["t_slope"], h["e_gap"], h["e_slope"]
        out(f"  vs {comp:8s}  {h['verdict']}")
        out(f"      CONDITIONAL cells in this test (H3 not PASS; included, §13.12): "
            f"{_units(conditional)}")
        out(f"      H1a gap ({ax['gap_text']}, BNN CVaR - {comp}):  {eg['mean']:+.4f}  "
            f"95% exact stratified [{eg['lower']:+.4f}, {eg['upper']:+.4f}]   "
            f"Wilcoxon one-sided p = {tg['p']:.4g}  (W+ {tg['w_plus']:g}, n {tg['n']}, "
            f"zeros dropped {tg['n_zero']})" + ("" if h["gap_on_lattice"] else
                                                "   [units off the score lattice: 1e-6 grid]"))
        out(f"      H1b widening ({ax['slope_text']}): {es['mean']:+.4f}  "
            f"95% exact stratified [{es['lower']:+.4f}, {es['upper']:+.4f}]   "
            f"Wilcoxon one-sided p = {ts['p']:.4g}  (W+ {ts['w_plus']:g}, n {ts['n']}, "
            f"zeros dropped {ts['n_zero']})")
        out("      per variant (DESCRIPTIVE, no test):")
        for v in VARIANTS:
            g, sl = units[v]["gap"], units[v]["slope"]
            bg, bs = R.exact_bootstrap(g), R.exact_bootstrap(sl)
            out(f"        {v:15s} gap {np.mean(g):+.4f} [{bg['lower']:+.4f}, {bg['upper']:+.4f}]  "
                f"slope {np.mean(sl):+.4f} [{bs['lower']:+.4f}, {bs['upper']:+.4f}]  "
                f"gap > 0 on {sum(x > 0 for x in g)}/10")
    full = all(v == "SUPPORTED" for v in verdicts.values())
    read = [c for c, v in verdicts.items() if v not in ("NOT READ", "SELF-CHECK FAILED")]
    out(f"\n   H1 on the {axis} axis (the conjunction over the three comparators): " +
        ("SUPPORTED" if full else
         "NOT YET READ" if len(read) < len(COMPARATORS) else
         "NOT SUPPORTED IN FULL -- " + "; ".join(f"{c}: {v}" for c, v in verdicts.items())))

    out("\nB. THE CURVES  (BNN CVaR - comparator per level; exact 10-seed interval; DESCRIPTIVE;"
        " * = H3-conditional unit)")
    for v in VARIANTS:
        for comp in COMPARATORS + DESCRIPTIVE:
            row = []
            for lv in levels + (FULL,):
                a, b = cells.get((v, PROPOSED, lv)), cells.get((v, comp, lv))
                if (v, lv) in unread:
                    row.append(f"{lv:>4s}  H3 unread ")
                    continue
                if R.cell_ready(a) or R.cell_ready(b):
                    row.append(f"{lv:>4s}   --   ")
                    continue
                c = R.compare(a, b)
                flag = "*" if (v, lv) in conditional else " "
                row.append(f"{lv:>4s}{flag}{c['mean']:+.3f} [{c['lower']:+.3f},{c['upper']:+.3f}]")
            out(f"  {v:15s} - {comp:8s} " + "  ".join(row))

    out("\nC. THE CELLS  (last-10 mean ± sd over seeds 1-10; sweep)")
    for v in VARIANTS:
        for m in (PROPOSED,) + COMPARATORS + DESCRIPTIVE:
            row = []
            for lv in levels + (FULL,):
                c = cells.get((v, m, lv))
                r = R.cell_ready(c)
                if r:
                    row.append(f"{lv:>4s} {'no sweep' if c is None else r[:22]:22s}")
                    if c is not None and c["audit"]:
                        problems += 1
                else:
                    x = np.array([c["scores"][s] for s in SEEDS])
                    row.append(f"{lv:>4s} {x.mean():.3f} ± {x.std(ddof=1):.3f} {c['sweep']:8s}")
            out(f"  {v:15s} {m:8s} " + "  ".join(row))
    out("\n" + ("" if not problems else f"!! {problems} audit or self-check failure(s)"))
    return 1 if problems else 0


# --------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)
    # --- the signed-rank test against brute force over all 2^n sign flips ---
    for n in (1, 4, 7, 9):
        for _ in range(5):
            u = np.round(rng.normal(size=n), 1)             # rounding forces ties and zeros
            res = signed_rank_exact(u, "greater")
            nz = u[u != 0]
            if nz.size == 0:
                assert res["p"] == 1.0
                continue
            from scipy.stats import rankdata
            r = rankdata(np.abs(nz))
            w = r[nz > 0].sum()
            sims = [sum(r[i] for i in range(nz.size) if sg[i]) for sg in
                    itertools.product((0, 1), repeat=nz.size)]
            brute_g = np.mean([x >= w - 1e-9 for x in sims])
            brute_l = np.mean([x <= w + 1e-9 for x in sims])
            assert abs(res["p"] - brute_g) < 1e-12, (u, res, brute_g)
            assert abs(signed_rank_exact(u, "less")["p"] - brute_l) < 1e-12
    # --- against scipy's exact test where scipy is exact (no ties, no zeros) ---
    from scipy.stats import wilcoxon
    u = rng.normal(0.02, 0.05, size=40)
    assert abs(signed_rank_exact(u, "greater")["p"] -
               wilcoxon(u, alternative="greater", method="exact").pvalue) < 1e-10
    assert abs(signed_rank_exact(u, "less")["p"] -
               wilcoxon(u, alternative="less", method="exact").pvalue) < 1e-10
    # --- the stratified bootstrap, one group, equals §12's exact bootstrap ---
    g = np.round(rng.uniform(-0.2, 0.3, size=10), 3)
    s1, b1 = stratified_exact([g], 0.001), R.exact_bootstrap(g)
    assert abs(s1["lower"] - b1["lower"]) < 1e-12 and abs(s1["upper"] - b1["upper"]) < 1e-12
    assert not checks(s1)
    # --- two small groups against brute force over every within-group resample ---
    ga, gb = np.round(rng.uniform(-0.1, 0.2, 3), 3), np.round(rng.uniform(-0.1, 0.2, 3), 3)
    means = np.sort([(ga[list(i)].mean() + gb[list(j)].mean()) / 2
                     for i in itertools.product(range(3), repeat=3)
                     for j in itertools.product(range(3), repeat=3)])
    k = means.size
    s2 = stratified_exact([ga, gb], 0.001)
    for q, nm in ((0.025, "lower"), (0.975, "upper")):
        brute = means[int(math.ceil(q * k - 1e-9)) - 1]
        assert abs(brute - s2[nm]) < 1e-9, (nm, brute, s2[nm])
    # --- four groups of ten: valid, deterministic, brackets the mean ---
    G = [np.round(rng.normal(0.03, 0.05, 10), 3) for _ in range(4)]
    s4 = stratified_exact(G, 0.001)
    assert not checks(s4) and s4["lower"] < s4["mean"] < s4["upper"]
    assert stratified_exact(G, 0.001) == s4
    sh = stratified_exact([x + 0.1 for x in G], 0.001)
    assert abs(sh["lower"] - s4["lower"] - 0.1) < 1e-9
    # --- slope ---
    assert abs(ols_slope([4, 5, 6, 7], [0.4, 0.3, 0.2, 0.1]) + 0.1) < 1e-12
    # --- end to end on synthetic cells: a widening gap is SUPPORTED, a uniform one is not ---
    def cell(vals):
        return {"scores": dict(zip(SEEDS, vals)), "robust": dict(zip(SEEDS, vals)),
                "incomplete": [], "audit": [], "sweep": "s", "index": 3, "name": "n"}

    def world(widening):
        cells = {}
        for v in VARIANTS:
            base = rng.uniform(0.3, 0.6, 10)
            for lv in LEVELS + (FULL,):
                x = math.log2(pairs_of(v, lv))
                gap = 0.25 - 0.03 * x if widening else 0.05
                noise = rng.normal(0, 0.01, 10)
                cells[(v, PROPOSED, lv)] = cell(np.round(base + gap + noise, 3))
                for m in COMPARATORS + DESCRIPTIVE:
                    cells[(v, m, lv)] = cell(np.round(base, 3))
        return cells
    def all_pass(axis):
        return {(v, lv): "PASS" for v in VARIANTS for lv in AXES[axis]["levels"] + (FULL,)}
    lines = []
    assert report(world(True), [], out=lines.append, gate=all_pass("sparsity")) == 0
    txt = "\n".join(lines)
    assert txt.count("SUPPORTED") >= 3 and "NOT SUPPORTED" not in txt, txt
    assert "H1 on the sparsity axis (the conjunction over the three comparators): SUPPORTED" in txt
    lines = []
    report(world(False), [], out=lines.append, gate=all_pass("sparsity"))
    txt = "\n".join(lines)
    assert "NOT SUPPORTED (no widening: H1b did not reject)" in txt, txt
    # --- an incomplete cell blocks the comparator's test and nothing else ---
    w = world(True)
    w[("large-play", "PT", "r64")] = dict(w[("large-play", "PT", "r64")],
                                          incomplete=["seed 3 running"])
    lines = []
    report(w, [], out=lines.append, gate=all_pass("sparsity"))
    txt = "\n".join(lines)
    assert "vs PT        NOT READ -- large-play r64 PT: INCOMPLETE" in txt, txt
    assert "vs MR        SUPPORTED" in txt and "NOT YET READ" in txt

    # --- the noise axis: the gap must GROW with p (positive slope) ---
    def noise_world(direction):
        cells = {}
        for v in VARIANTS:
            base = rng.uniform(0.3, 0.6, 10)
            for lv in NOISE_LEVELS + (FULL,):
                p = _x_noise(v, lv)
                gap = 0.02 + direction * 0.3 * p
                cells[(v, PROPOSED, lv)] = cell(np.round(base + gap + rng.normal(0, 0.01, 10), 3))
                for m in COMPARATORS + DESCRIPTIVE:
                    cells[(v, m, lv)] = cell(np.round(base, 3))
        return cells
    lines = []
    assert report(noise_world(+1), [], out=lines.append, axis="noise", gate=all_pass("noise")) == 0
    txt = "\n".join(lines)
    assert "EXPERIMENT 2" in txt and "NOT SUPPORTED" not in txt, txt
    assert "H1 on the noise axis (the conjunction over the three comparators): SUPPORTED" in txt
    lines = []
    report(noise_world(-1), [], out=lines.append, axis="noise", gate=all_pass("noise"))  # shrinks
    assert "no widening: H1b did not reject" in "\n".join(lines)
    # the same widening world read on the wrong axis has no cells: nothing is read
    lines = []
    report(noise_world(+1), [], out=lines.append, axis="sparsity", gate=all_pass("sparsity"))
    txt = "\n".join(lines)
    assert all(f"vs {c:8s}  NOT READ --" in txt for c in COMPARATORS), txt
    assert "comparators): NOT YET READ" in txt
    assert _x_noise("large-play", FULL) == 0.0 and _x_noise("large-play", "n045") == 0.45
    # --- H3 (§13.12): a failed unit is listed and the verdict is unchanged ---
    g = all_pass("sparsity")
    g[("large-play", "r16")] = "FAIL (spread interval reaches 0.5)"
    g[("medium-play", FULL)] = "UNDEFINED (no wrong or no right pair on seeds [3]) -- not passed"
    wt = world(True)
    lines = []
    report(wt, [], out=lines.append, gate=g)
    txt = "\n".join(lines)
    assert "conditional units (2): medium-play full, large-play r16" in txt, txt
    assert txt.count("CONDITIONAL cells in this test") == 3 and "vs MR        SUPPORTED" in txt
    assert "r16*" in txt                                           # flagged on the curve
    lines0 = []
    report(wt, [], out=lines0.append, gate=all_pass("sparsity"))
    pick = lambda s: [x for x in s if "H1a gap" in x or "H1b widening" in x]
    assert pick(lines) == pick(lines0)                             # flagged, NOT dropped
    # --- the gate is read first: an unread unit blocks every verdict ---
    g[("large-diverse", "r128")] = "INCOMPLETE (no measurement for seeds [1, 2])"
    lines = []
    report(wt, [], out=lines.append, gate=g)
    txt = "\n".join(lines)
    assert txt.count("NOT READ -- the H3 gate is read first") == 3 and "H3 unread" in txt
    assert abs(_x_sparsity("large-play", FULL) - math.log2(254)) < 1e-12
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winners", default=os.path.join(ROOT, "phase2_winners.json"))
    ap.add_argument("--axis", choices=tuple(AXES), default="sparsity")
    ap.add_argument("--selftest", action="store_true")
    import hold
    hold.parse()                                   # before argparse sees the flag
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    hold.refuse("exp_readout.py", "RQ2's main hypothesis: BNN CVaR against its comparators, and the H3 gate")
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    winners = json.load(open(a.winners))
    api = wandb.Api(timeout=120)
    gate = gate_status(a.axis)                     # the H3 gate first (§13.10)
    cells, notes = collect_exp(api, P, winners, a.axis)
    anchor, anotes = R.collect(api, P, winners)
    add_anchor(cells, anchor)
    return report(cells, notes + anotes, axis=a.axis, gate=gate)


if __name__ == "__main__":
    sys.exit(main())
