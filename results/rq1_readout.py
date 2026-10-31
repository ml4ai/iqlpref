#!/usr/bin/env python
"""rq1_readout.py -- the REGISTERED reading of RQ1's sensitivity test (gp_reward-priors
HANDOFF §14.5, fixed by the author 2026-10-20 before any no-terminals ladder result existed;
built 2026-10-30, §14.69, before any candidate-arm ladder score was read).

Nothing here is a free choice; every rule below is §14.5's.

WHAT IS COMPARED
  families   MR best-model, PT, NMR, ensemble MEAN (the BNN/CVaR hold, §14.1: nothing covered)
  protocols  standard (the dataset's terminals) and candidate (drop_terminals); the SAME
             reward models in both, each protocol at its own frozen normalization index
  ladders    sparsity: 16, 32, 64, 128 pairs and the full train-pool (the anchor)
             noise:    flip 0.1, 0.2, 0.3, 0.4, 0.45 (PRIMARY arm) and 0 (the anchor)
             One verdict per ladder; no verdict pools the ladders.

QUANTITIES, per (variant, seed)
  S   the last-10 statistic of the IQL run (results/iql_score.py), as everywhere
  A   held-out preference accuracy, PER PAIR, on that seed's clean, tie-free validation
      split; for the ensemble, the accuracy of its mean reward.  Measured on the box by
      results/rq1_accuracy.py (exp/rq1_accuracy.json).  For MR, PT and NMR each record's
      batch mean must equal the training run's logged eval_acc_reloaded, or A is not read.
  degradation slope   the OLS slope over ALL the ladder's points, the anchor included,
      against log2(pairs) (sparsity) or the flip fraction (noise), SIGNED so that positive
      means the quantity falls as the data degrades (sparsity: the slope; noise: minus it)

STATISTICS (as §13.9, §13.11)
  unit    one (variant, seed); the variants pooled into one test per family and ladder
  test    one-sided exact Wilcoxon signed-rank at alpha = 0.025 (zeros dropped, midranks,
          exact sign-flip distribution): exp_readout.signed_rank_exact
  effect  exact stratified seed bootstrap, 95%, pooled (exp_readout.stratified_exact) and
          per variant (eval_readout.exact_bootstrap)
  gate    every cell a test reads is complete and audit-clean before the test runs; a
          family and ladder with any cell missing is NOT READ, and nothing of its
          candidate arm is printed

READ FIRST
  1. Manipulation check, per family and ladder: the degradation slope of A > 0, on all four
     variants.  A family and ladder that fail it do not test sensitivity: FLAGGED, and
     reported beside the verdicts.
  2. Usability check, per variant: candidate protocol, full data, S_task - S_zero > 0 over
     the variant's ten seeds (the task reward at its own no-terminals index).  A variant
     that fails is excluded from every candidate-protocol test below.

THE THREE TESTS, per family and ladder, on the usable variants only
  (a) the candidate degradation slope of S > 0
  (b) per unit, candidate slope - standard slope > 0
  (c) full data, candidate: S_family - S_zero > 0 AND S_family - S_mean-constant > 0
      ((c) does not depend on the ladder; both ladders share it)
  VERDICT  SENSITIVE iff (a), (b) and (c) all reject; otherwise NOT SENSITIVE, naming the
           part that did not.  An intersection of tests each at 0.025: no correction.
  RQ1's second part holds in full on a ladder iff every family is SENSITIVE; partial
  support is reported as partial.  RQ1 is refuted on a ladder if (a) or (b) fails on the
  usable variants, or if no variant passes the usability check -- read PER FAMILY: one read
  family whose (a) or (b) does not reject refutes the ladder; a FLAGGED family does not
  (author, 2026-10-31, §14.70; decided AFTER MR's sparsity reading had been seen, and so
  disclosed).

REPORTED BESIDE, NO TEST
  - the motion reward at full data under both protocols, against each family
  - the secondary non-inferiority check on the standard protocol (the form of §12): at full
    data with terminals, is each trivial reward within 0.10 of the family?  On the exact 95%
    interval of S_trivial - S_family: NOT BEHIND iff its lower end exceeds -0.10, TOO FAR
    BEHIND iff its upper end is below -0.10, INCONCLUSIVE otherwise; per variant and pooled,
    no correction

Run from the repo root:
    python results/rq1_readout.py [--axis sparsity|noise] [--acc exp/rq1_accuracy.json]
    python results/rq1_readout.py --selftest
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import eval_readout as R                                   # noqa: E402  cells, gate, exact bootstrap
import exp_readout as X                                    # noqa: E402  signed-rank, stratified bootstrap
import hold                                                # noqa: E402  §14.1

ALPHA = 0.025
MARGIN = 0.10                       # the secondary check's margin, §14.5 (as §12)
SEEDS = R.SEEDS
VARIANTS = R.VARIANTS               # "medium-play", ...
FULL = X.FULL
FAMILIES = ("MR", "PT", "NMR", "ENS_MEAN")
ACC_FAM = {"MR": "mr", "PT": "pt", "NMR": "nmr", "ENS_MEAN": "ens_mean"}
AXES = {"sparsity": {"levels": X.LEVELS, "x": X._x_sparsity, "sign": +1.0,
                     "title": "SPARSITY (16, 32, 64, 128 pairs + the full train-pool)"},
        "noise": {"levels": X.NOISE_LEVELS, "x": X._x_noise, "sign": -1.0,
                  "title": "LABEL NOISE, primary arm (flip 0.1-0.45 + the clean anchor)"}}
ACC_PATH = os.path.join(ROOT, "exp", "rq1_accuracy.json")
SINCE = "2026-09-25T00:00:00"       # production_readout's default; every model set is newer


# --------------------------------------------------------------------------
# Units and tests (pure; self-tested)
# --------------------------------------------------------------------------
def dslope(xs, ys, sign):
    """The degradation slope: OLS slope, signed so that positive = falls as data degrades."""
    return sign * X.ols_slope(xs, ys)


def pooled_test(groups, delta=1e-6):
    """groups: {variant: [10 units]}.  One-sided (> 0) exact Wilcoxon over the pooled units,
    the exact stratified interval, and each variant's exact 10-seed interval."""
    units = [u for v in groups for u in groups[v]]
    t = X.signed_rank_exact(units, "greater")
    e = X.stratified_exact([groups[v] for v in groups], delta) if groups else None
    per = {v: R.exact_bootstrap(groups[v]) for v in groups}
    return {"t": t, "e": e, "per": per, "rejects": t["p"] < ALPHA, "n_units": len(units)}


def score_delta(groups):
    return X.unit_delta([u for v in groups for u in groups[v]], X.SCORE_LATTICE)[0]


def usability(task, zero):
    """task, zero: {variant: cell}, candidate protocol, full data.  -> ({v: result}, None) or
    (None, reasons)."""
    why = [f"{v} {who}: {R.cell_ready(c)}" for v in VARIANTS
           for who, c in (("task reward", task.get(v)), ("zero", zero.get(v))) if R.cell_ready(c)]
    if why:
        return None, why
    out = {}
    for v in VARIANTS:
        d = [task[v]["scores"][s] - zero[v]["scores"][s] for s in SEEDS]
        t = X.signed_rank_exact(d, "greater")
        out[v] = {"t": t, "e": R.exact_bootstrap(d), "usable": t["p"] < ALPHA}
    return out, None


def family_units(fam, axis, std, cand, acc):
    """{variant: {"A", "S_cand", "S_std", "diff": [10]}} or (None, reasons)."""
    ax = AXES[axis]
    lvls = ax["levels"] + (FULL,)
    why = []
    for v in VARIANTS:
        for lv in lvls:
            for arm, cells in (("standard", std), ("candidate", cand)):
                r = R.cell_ready(cells.get((v, fam, lv)))
                if r:
                    why.append(f"{arm} {v} {lv}: {r}")
            for s in SEEDS:
                if not isinstance(acc.get((fam, v, lv, s)), float):
                    why.append(f"A {v} {lv} seed {s}: {acc.get((fam, v, lv, s)) or 'not measured'}")
    if why:
        return None, why
    out = {}
    for v in VARIANTS:
        xs = [ax["x"](v, lv) for lv in lvls]
        u = {"A": [], "S_cand": [], "S_std": [], "diff": []}
        for s in SEEDS:
            a = dslope(xs, [acc[(fam, v, lv, s)] for lv in lvls], ax["sign"])
            c = dslope(xs, [cand[(v, fam, lv)]["scores"][s] for lv in lvls], ax["sign"])
            st = dslope(xs, [std[(v, fam, lv)]["scores"][s] for lv in lvls], ax["sign"])
            u["A"].append(a)
            u["S_cand"].append(c)
            u["S_std"].append(st)
            u["diff"].append(c - st)
        out[v] = u
    return out, None


def test_c(fam_full, zero, mean, usable):
    """(c): candidate, full data, family - zero and family - mean-constant, usable variants."""
    why = [f"{v} {who}: {R.cell_ready(c)}" for v in usable
           for who, c in (("family", fam_full.get(v)), ("zero", zero.get(v)), ("mean-constant", mean.get(v)))
           if R.cell_ready(c)]
    if why:
        return None, why
    res = {}
    for tag, triv in (("zero", zero), ("mean", mean)):
        g = {v: [fam_full[v]["scores"][s] - triv[v]["scores"][s] for s in SEEDS] for v in usable}
        res[tag] = pooled_test(g, score_delta(g))
    res["rejects"] = res["zero"]["rejects"] and res["mean"]["rejects"]
    return res, None


def read_family(fam, axis, std, cand, acc, use, fam_full, zero, mean):
    """The whole registered reading of one family on one ladder."""
    units, why = family_units(fam, axis, std, cand, acc)
    if why:
        return {"status": "NOT READ", "why": why}
    usable = [v for v in VARIANTS if use[v]["usable"]]
    manip = pooled_test({v: units[v]["A"] for v in VARIANTS})
    out = {"status": "READ", "manip": manip, "usable": usable, "units": units}
    if not usable:
        out["verdict"] = "NOT SENSITIVE (no variant passes the usability check: RQ1 refuted on this ladder)"
        return out
    out["a"] = pooled_test({v: units[v]["S_cand"] for v in usable})
    out["b"] = pooled_test({v: units[v]["diff"] for v in usable})
    out["c"], cwhy = test_c(fam_full, zero, mean, usable)
    if cwhy:
        return {"status": "NOT READ", "why": cwhy}
    parts = (("(a) the candidate protocol does not respond", out["a"]["rejects"]),
             ("(b) it does not respond more than the standard protocol", out["b"]["rejects"]),
             ("(c) it does not separate the family from the trivial rewards", out["c"]["rejects"]))
    v = "SENSITIVE" if all(ok for _, ok in parts) else \
        "NOT SENSITIVE (" + "; ".join(t for t, ok in parts if not ok) + ")"
    if not manip["rejects"]:
        v = f"FLAGGED: the manipulation check did not reject, so this is not a sensitivity test [{v}]"
    out["verdict"] = v
    return out


def ni_verdict(e):
    """§12's three-way form: NOT BEHIND iff the lower end > -MARGIN; TOO FAR BEHIND iff the
    upper end < -MARGIN; INCONCLUSIVE otherwise."""
    return "NOT BEHIND" if e["lower"] > -MARGIN else "TOO FAR BEHIND" if e["upper"] < -MARGIN else "INCONCLUSIVE"


def ladder_verdict(verdicts, res):
    """RQ1's second part on the ladder.  §14.5's refutation clause ("refuted on a ladder if (a)
    or (b) fails on the usable variants") is read PER FAMILY (author, 2026-10-31, §14.70): one
    read family whose (a) or (b) does not reject refutes the ladder.  A FLAGGED family (failed
    manipulation check) is not a sensitivity test and does not refute.  Holds in full only when
    every family is read and SENSITIVE."""
    tag = ", ".join(f"{f} {v.split(' (')[0].split(':')[0]}" for f, v in verdicts.items()) or "none"
    refuting = [f for f, r in res.items() if r.get("status") == "READ" and not r["verdict"].startswith("FLAGGED")
                and ("a" not in r or not r["a"]["rejects"] or not r["b"]["rejects"])]
    def fails(r):
        if "a" not in r:                                   # no usable variant
            return r["verdict"]
        return " and ".join(x for x, ok in (("(a)", r["a"]["rejects"]), ("(b)", r["b"]["rejects"])) if not ok) + \
            " did not reject"
    if refuting:
        why = "; ".join(f"{f}: {fails(res[f])}" for f in refuting)
        return (f"  REFUTED on this ladder (per-family reading, §14.70): {why}.  "
                f"{len(verdicts)} of {len(FAMILIES)} families read ({tag})")
    if len(verdicts) < len(FAMILIES):
        return f"  not yet: {len(verdicts)} of {len(FAMILIES)} families read ({tag})"
    if all(v == "SENSITIVE" for v in verdicts.values()):
        return "  HOLDS IN FULL: SENSITIVE for every family"
    return f"  PARTIAL: {tag}"


def non_inferior(triv, fam, variants=VARIANTS):
    """S_trivial - S_family, full data, with terminals, in §12's form."""
    if any(R.cell_ready(triv.get(v)) or R.cell_ready(fam.get(v)) for v in variants):
        return None
    g = {v: [triv[v]["scores"][s] - fam[v]["scores"][s] for s in SEEDS] for v in variants}
    per = {v: R.exact_bootstrap(g[v]) for v in variants}
    pooled = X.stratified_exact([g[v] for v in variants], score_delta(g))
    return {"per": per, "pooled": pooled, "per_verdict": {v: ni_verdict(per[v]) for v in variants},
            "pooled_verdict": ni_verdict(pooled)}


# --------------------------------------------------------------------------
# A, the accuracies
# --------------------------------------------------------------------------
_RUN = re.compile(r"antmaze_(medium_play|medium_diverse|large_play|large_diverse)_(mr|pt|nmr)"
                  r"(?:_(r\d+|n\d+))?_eval_(\d+)$")


def load_accuracy(path, logged=None):
    """{(FAM, variant, level, seed): float or reason}.  logged: {(fam, variant_, level, seed):
    eval_acc_reloaded} from the training runs; a best-model record whose batch mean differs
    from it by more than 1e-6 is not read (the measurement would not be the trained model's)."""
    if not os.path.exists(path):
        return {}, [f"{os.path.relpath(path, ROOT)} not found: A is not measured yet (run "
                    f"results/rq1_accuracy.py measure on the box and copy the file over)"]
    raw = json.load(open(path))
    out, notes, bad = {}, [], 0
    inv = {v: k for k, v in ACC_FAM.items()}
    for k, rec in raw.items():
        fam, lv, vu, s = k.split("|")
        key = (inv[fam], vu.replace("_", "-"), FULL if lv == "anchor" else lv, int(s))
        val = rec["correct"] / rec["n"]
        if "batch_mean" in rec and logged is not None:
            lg = logged.get((fam, vu, lv, int(s)))
            if lg is None or abs(lg - rec["batch_mean"]) > 1e-6:
                out[key] = f"batch mean {rec['batch_mean']:.6f} != logged {lg!r}"
                bad += 1
                continue
        out[key] = float(val)
    if bad:
        notes.append(f"{bad} accuracy record(s) failed the logged-accuracy cross-check")
    return out, notes


def logged_accuracies():
    """{(fam, variant_, level, seed): eval_acc_reloaded} for MR, PT and NMR from wandb."""
    sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors"))
    sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors", "scripts_bnn"))
    import production_readout as PR
    out, dup = {}, set()
    for fam in ("mr", "pt", "nmr"):
        prod, _ = PR.fetch(fam, SINCE)
        for r in prod:
            if r["state"] != "finished":
                continue
            m = _RUN.fullmatch(os.path.basename(os.path.normpath(str(r["config"].get("checkpoints_path") or ""))))
            if not m or m.group(2) != fam:
                continue
            k = (fam, m.group(1), m.group(3) or "anchor", int(m.group(4)))
            if k in out:
                dup.add(k)
            out[k] = r["summary"].get("eval_acc_reloaded")
    for k in dup:                                          # two finished runs for one model: not read
        out[k] = None
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _p(t):
    return f"p = {t['p']:.4g} (W+ {t['w_plus']:g}, n {t['n']}, {t['n_zero']} zero)"


def _iv(e):
    return f"{e['mean']:+.4f} [{e['lower']:+.4f}, {e['upper']:+.4f}]"


def report(axis, res, use, uwhy, beside, notes, out=print):
    out("=" * 100)
    out(f"RQ1 SENSITIVITY TEST (HANDOFF §14.5) -- REGISTERED.  {AXES[axis]['title']}")
    out("one-sided exact Wilcoxon at 0.025 over pooled (variant, seed) units; exact stratified 95%")
    out("=" * 100)
    if not hold.INCLUDE:
        out("  BNN/CVaR HOLD (HANDOFF §14.1): BNN CVaR, BNN mean and ensemble CVaR are not part of RQ1's test")
    for n in notes:
        out(f"  note: {n}")
    out("\nUSABILITY CHECK  (candidate protocol, full data: task reward at its own index - zero, per variant)")
    if uwhy:
        out("  NOT READ -- " + "; ".join(uwhy[:4]) + (f" (+{len(uwhy) - 4} more)" if len(uwhy) > 4 else ""))
    else:
        for v in VARIANTS:
            u = use[v]
            out(f"  {v:15s} {_iv(u['e'])}   {_p(u['t'])}   {'USABLE' if u['usable'] else 'EXCLUDED'}")
    verdicts = {}
    for fam in FAMILIES:
        r = res.get(fam)
        out(f"\n{fam}")
        if r is None or r["status"] != "READ":
            why = (r or {}).get("why", ["usability check not read"])
            out(f"  NOT READ -- {len(why)} reason(s): " + "; ".join(why[:4]) + (" ..." if len(why) > 4 else ""))
            continue
        m = r["manip"]
        out(f"  manipulation check, slope of A (all 4 variants): pooled {_iv(m['e'])}   {_p(m['t'])}   "
            f"{'REJECTS' if m['rejects'] else 'does NOT reject'}")
        for v in VARIANTS:
            out(f"      {v:15s} {_iv(m['per'][v])}")
        out(f"  usable variants: {', '.join(r['usable']) or 'none'}")
        for tag, title in (("a", "(a) candidate slope of S"), ("b", "(b) candidate - standard slope")):
            if tag not in r:
                continue
            t = r[tag]
            out(f"  {title:34s} pooled {_iv(t['e'])}   {_p(t['t'])}   {'REJECTS' if t['rejects'] else 'does NOT reject'}")
            for v in r["usable"]:
                out(f"      {v:15s} {_iv(t['per'][v])}")
        if "c" in r:
            for tag, title in (("zero", "(c) family - zero"), ("mean", "    family - mean-constant")):
                t = r["c"][tag]
                out(f"  {title:34s} pooled {_iv(t['e'])}   {_p(t['t'])}   "
                    f"{'REJECTS' if t['rejects'] else 'does NOT reject'}")
        out(f"  VERDICT: {r['verdict']}")
        verdicts[fam] = r["verdict"]
    out("\nRQ1 ON THIS LADDER")
    out(ladder_verdict(verdicts, res))
    out("\nBESIDE, NO TEST")
    for fam in FAMILIES:
        b = beside.get(fam, {})
        for tv in ("zero", "mean"):
            ni = b.get(f"ni_{tv}")
            if ni is None:
                out(f"  {fam:9s} {tv:4s} - {fam}, with terminals, full data: NOT READ")
                continue
            out(f"  {fam:9s} {tv:4s} - {fam}, with terminals, full data: pooled {_iv(ni['pooled'])} "
                f"{ni['pooled_verdict']};  " + ", ".join(f"{v} {ni['per_verdict'][v]}" for v in VARIANTS))
        for tag, title in (("motion_std", "with terminals"), ("motion_cand", "without terminals")):
            mo = b.get(tag)
            if mo:
                out(f"  {fam:9s} {fam} - motion, {title}, full data: " +
                    ", ".join(f"{v} {_iv(mo[v])}" for v in VARIANTS))
    return verdicts


# --------------------------------------------------------------------------
def collect(api, P, winners, gwin, axis):
    import control_readout as C
    std, n1 = X.collect_exp(api, P, winners, axis, "standard")
    cand, n2 = X.collect_exp(api, P, gwin, axis, "candidate")
    anchor, n3 = R.collect(api, P, winners)
    ctrl, n4 = C.collect_controls(api, P, winners, gwin)
    full = {}
    for v in VARIANTS:
        for m in FAMILIES + ("task_reward",):
            std[(v, m, FULL)] = anchor.get((f"antmaze-{v}-v2", m))
            full[(v, m)] = (C.at_winner(ctrl, gwin, winners, v, m) if m in ("task_reward", "MR")
                            else ctrl.get((v, f"atwin_{m}")))
            cand[(v, m, FULL)] = full[(v, m)]
    return std, cand, full, ctrl, n1 + n2 + n3 + n4


def run(axis, std, cand, full, ctrl, acc):
    task = {v: full[(v, "task_reward")] for v in VARIANTS}
    zero = {v: ctrl.get((v, "trivial_zero_noterm")) for v in VARIANTS}
    mean = {v: ctrl.get((v, "trivial_mean_noterm")) for v in VARIANTS}
    use, uwhy = usability(task, zero)
    res, beside = {}, {}
    for fam in FAMILIES:
        if not hold.shown(fam):
            continue
        if use is not None:
            res[fam] = read_family(fam, axis, std, cand, acc, use, {v: full[(v, fam)] for v in VARIANTS}, zero, mean)
        fam_std = {v: std.get((v, fam, FULL)) for v in VARIANTS}
        b = {"ni_zero": non_inferior({v: ctrl.get((v, "trivial_zero")) for v in VARIANTS}, fam_std),
             "ni_mean": non_inferior({v: ctrl.get((v, "trivial_mean")) for v in VARIANTS}, fam_std)}
        for tag, fam_cells, mot in (("motion_std", fam_std, {v: ctrl.get((v, "heuristic")) for v in VARIANTS}),
                                    ("motion_cand", {v: full[(v, fam)] for v in VARIANTS},
                                     {v: ctrl.get((v, "atwin_heuristic")) for v in VARIANTS})):
            if not any(R.cell_ready(fam_cells[v]) or R.cell_ready(mot[v]) for v in VARIANTS):
                b[tag] = {v: R.exact_bootstrap([fam_cells[v]["scores"][s] - mot[v]["scores"][s] for s in SEEDS])
                          for v in VARIANTS}
        beside[fam] = b
    return res, use, uwhy, beside


def selftest():
    rng = np.random.default_rng(7)

    def cell(xs):
        return {"scores": {s: float(round(x, 3)) for s, x in zip(SEEDS, xs)}, "audit": [], "incomplete": [],
                "sweep": "t"}

    def build(axis, cand_slope=0.2, std_slope=0.0, acc_slope=0.1, task_minus_zero=0.3, fam_minus_zero=0.2,
              bad_variant=None):
        ax = AXES[axis]
        lv = ax["levels"] + (FULL,)
        std, cand, full, ctrl, acc = {}, {}, {}, {}, {}
        for v in VARIANTS:
            xs = np.array([ax["x"](v, l) for l in lv])
            for i, l in enumerate(lv):
                dx = ax["sign"] * (xs[i] - xs.mean()) / (xs.max() - xs.min())      # the effect over the ladder
                cand[(v, "MR", l)] = cell(0.3 + cand_slope * dx + rng.normal(0, 0.01, 10))
                std[(v, "MR", l)] = cell(0.6 + std_slope * dx + rng.normal(0, 0.01, 10))
                for s in SEEDS:
                    acc[("MR", v, l, s)] = float(0.8 + acc_slope * dx + rng.normal(0, 0.005))
            z = rng.uniform(0, 0.01, 10)
            ctrl[(v, "trivial_zero_noterm")] = cell(z)
            ctrl[(v, "trivial_mean_noterm")] = cell(z)
            ctrl[(v, "trivial_zero")], ctrl[(v, "trivial_mean")] = cell(z), cell(z + 0.65)
            ctrl[(v, "heuristic")], ctrl[(v, "atwin_heuristic")] = cell(z + 0.6), cell(z + 0.2)
            tm = -0.1 if v == bad_variant else task_minus_zero
            full[(v, "task_reward")] = cell(z + tm + rng.normal(0, 0.01, 10))
            full[(v, "MR")] = cand[(v, "MR", FULL)] = cell(z + fam_minus_zero + rng.normal(0, 0.01, 10))
            for m in ("PT", "NMR", "ENS_MEAN"):
                full[(v, m)] = None
        return std, cand, full, ctrl, acc

    lines = []
    for axis in AXES:
        std, cand, full, ctrl, acc = build(axis)
        res, use, uwhy, beside = run(axis, std, cand, full, ctrl, acc)
        r = res["MR"]
        assert uwhy is None and all(use[v]["usable"] for v in VARIANTS)
        assert r["status"] == "READ" and r["verdict"] == "SENSITIVE", (axis, r.get("verdict"))
        assert r["manip"]["rejects"] and r["a"]["t"]["p"] < 1e-6 and r["manip"]["n_units"] == 40
        assert res["PT"]["status"] == "NOT READ"                         # PT has no cells
        # the noise sign: S falling with the flip fraction is a POSITIVE degradation slope
        assert np.mean(r["units"]["medium-play"]["S_cand"]) > 0
        verdicts = report(axis, res, use, uwhy, beside, [], out=lines.append)
        assert verdicts == {"MR": "SENSITIVE"} and any("not yet: 1 of 4" in x for x in lines)
    # (b): the standard protocol responds as much as the candidate
    std, cand, full, ctrl, acc = build("sparsity", std_slope=0.2)
    res = run("sparsity", std, cand, full, ctrl, acc)[0]
    r = res["MR"]
    assert r["verdict"].startswith("NOT SENSITIVE") and "(b)" in r["verdict"] and "(a)" not in r["verdict"]
    # the per-family reading (§14.70): one read family failing (a) or (b) refutes the ladder
    lv = ladder_verdict({"MR": r["verdict"]}, res)
    assert "REFUTED" in lv and "MR: (b) did not reject" in lv and "1 of 4" in lv, lv
    # (c): the family no better than zero
    std, cand, full, ctrl, acc = build("sparsity", fam_minus_zero=0.0)
    r = run("sparsity", std, cand, full, ctrl, acc)[0]["MR"]
    assert "(c)" in r["verdict"], r["verdict"]
    # the manipulation check fails: FLAGGED, not a sensitivity verdict
    std, cand, full, ctrl, acc = build("sparsity", acc_slope=-0.1)
    res = run("sparsity", std, cand, full, ctrl, acc)[0]
    r = res["MR"]
    assert r["verdict"].startswith("FLAGGED") and not r["manip"]["rejects"]
    assert ladder_verdict({"MR": r["verdict"]}, res).startswith("  not yet")      # FLAGGED does not refute
    # an unusable variant is left out of (a)-(c) but kept in the manipulation check
    std, cand, full, ctrl, acc = build("sparsity", bad_variant="large-diverse")
    res, use, _, _ = run("sparsity", std, cand, full, ctrl, acc)
    assert not use["large-diverse"]["usable"] and res["MR"]["usable"] == ["medium-play", "medium-diverse", "large-play"]
    assert res["MR"]["a"]["n_units"] == 30 and res["MR"]["manip"]["n_units"] == 40
    # no usable variant: refuted
    std, cand, full, ctrl, acc = build("sparsity", task_minus_zero=-0.1)
    res = run("sparsity", std, cand, full, ctrl, acc)[0]
    r = res["MR"]
    assert "no variant passes the usability check" in r["verdict"]
    assert "REFUTED" in ladder_verdict({"MR": r["verdict"]}, res)
    # the gate: one incomplete candidate cell, or one missing A, and nothing is read
    std, cand, full, ctrl, acc = build("sparsity")
    cand[("large-play", "MR", "r64")]["incomplete"] = ["seed 3 running"]
    r = run("sparsity", std, cand, full, ctrl, acc)[0]["MR"]
    assert r["status"] == "NOT READ" and "units" not in r and any("candidate large-play r64" in w for w in r["why"])
    std, cand, full, ctrl, acc = build("sparsity")
    acc[("MR", "medium-play", "r16", 4)] = "batch mean 0.5 != logged 0.6"
    r = run("sparsity", std, cand, full, ctrl, acc)[0]["MR"]
    assert r["status"] == "NOT READ" and any("A medium-play r16 seed 4" in w for w in r["why"])
    # the degradation slope's sign
    assert dslope([4, 5, 6], [0.1, 0.2, 0.3], +1) > 0 and dslope([0.0, 0.1, 0.2], [0.3, 0.2, 0.1], -1) > 0
    # the secondary check: a trivial reward far behind is not NOT BEHIND
    ni = run("sparsity", *build("sparsity"))[3]["MR"]["ni_zero"]
    assert ni["pooled_verdict"] == "TOO FAR BEHIND" and ni["pooled"]["upper"] < -0.5
    assert ni_verdict({"lower": -0.05, "upper": 0.02}) == "NOT BEHIND"
    assert ni_verdict({"lower": -0.15, "upper": -0.05}) == "INCONCLUSIVE"
    # A: the logged-accuracy cross-check
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.json")
        json.dump({"mr|anchor|medium_play|1": {"correct": 70, "n": 77, "batch_mean": 0.9},
                   "mr|r16|medium_play|2": {"correct": 60, "n": 77, "batch_mean": 0.8},
                   "ens_mean|r16|medium_play|2": {"correct": 61, "n": 77, "n_snapshots": 40}}, open(p, "w"))
        a, notes = load_accuracy(p, {("mr", "medium_play", "anchor", 1): 0.9, ("mr", "medium_play", "r16", 2): 0.7})
        assert a[("MR", "medium-play", FULL, 1)] == 70 / 77 and isinstance(a[("MR", "medium-play", "r16", 2)], str)
        assert a[("ENS_MEAN", "medium-play", "r16", 2)] == 61 / 77 and notes
        assert load_accuracy(os.path.join(d, "missing.json"))[0] == {}
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", choices=tuple(AXES), default="sparsity")
    ap.add_argument("--acc", default=ACC_PATH)
    ap.add_argument("--winners", default=os.path.join(ROOT, "phase2_winners.json"))
    ap.add_argument("--selftest", action="store_true")
    hold.parse()
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    winners = json.load(open(a.winners))
    gwin = json.load(open(os.path.join(ROOT, P.GRID_WINNERS)))
    api = wandb.Api(timeout=120)
    std, cand, full, ctrl, notes = collect(api, P, winners, gwin, a.axis)
    acc, anotes = load_accuracy(a.acc, logged_accuracies() if os.path.exists(a.acc) else None)
    res, use, uwhy, beside = run(a.axis, std, cand, full, ctrl, acc)
    keep = [n for n in notes + anotes if hold.INCLUDE or not any(x in n for x in ("BNN", "ENS_CVAR", "bnn"))]
    report(a.axis, res, use, uwhy, beside, keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
