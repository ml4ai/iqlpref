#!/usr/bin/env python
"""eval_readout.py -- the REGISTERED reading of the unaltered-data evaluation.

Implements gp_reward-priors/HANDOFF_HP_SELECTION.md §12 (fixed 2026-10-07) and
the dissertation's Statistical and Reporting Protocol.  Nothing here is a free
choice; every rule below is quoted from §12, and changing one changes a verdict.

THE RULE (§12.2)
  statistic    last-10 mean of the 100-episode evaluation means, 0-1 scale,
               from results/iql_score.py.  A run must have all 200 points.
  quantity     per-seed difference  BNN CVaR - comparator,  seeds 1-10, PAIRED
  interval     95% paired bootstrap, resampling SEEDS, computed EXACTLY by
               enumeration (§12.5): all C(19,10) = 92,378 count vectors, each
               with its multinomial probability.  lower = the smallest value
               whose cumulative probability is >= 0.025, upper = the smallest
               whose cumulative probability is >= 0.975.  No Monte Carlo.
  comparators  MR best-model and PT, each read separately.  Named in advance.
  margin       0.10, the same on every variant
  verdict      NOT BEHIND      lower end > -0.10
               TOO FAR BEHIND  upper end < -0.10
               INCONCLUSIVE    anything else
  Eight verdicts, all reported.  There is NO rule combining them.

BEFORE ANY NUMBER (§12.5)
  completeness gate   a cell is read only if all ten seeds of BOTH arms are
                      finished with all 200 evaluation points; otherwise it is
                      INCOMPLETE -- not read, and not estimated from the seeds
                      that exist.
  config audit        every run's wandb config shows the intended reward-model
                      path for its seed, the stage-4 index from
                      phase2_winners.json, gauge_reward max0, and the family's
                      own settings (BNN: bnn_alpha, bnn_n_samples -1,
                      centre_draws True).  A cell with an audit failure is not
                      read.

ALSO REPORTED, never part of a verdict (§12.2): the same paired difference and
interval for BNN CVaR - BNN mean, - ensemble CVaR, - ensemble mean, and every
method against the task reward.  Beside each interval, as a check only, the
paired-t 95% interval; if the two would give different verdicts the readout says
so, and the verdict stays the bootstrap's.

Run from the repo root:
    python results/eval_readout.py                 # the readout
    python results/eval_readout.py --selftest
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
import hold                                                # noqa: E402  §14.1

MARGIN = 0.10                       # §12.2, the same on every variant
COVERAGE = 0.95
SEEDS = list(range(1, 11))
N_COUNT_VECTORS = 92378             # C(19, 10), §12.5
COMPARATORS = ("MR", "PT")          # named in advance, §12.2
PROPOSED = "BNN_CVAR"
MECHANISM = ("BNN_MEAN", "ENS_CVAR", "ENS_MEAN")          # §12.2, not verdicts
VARIANTS = ("medium-play", "medium-diverse", "large-play", "large-diverse")
T975 = {9: 2.2621571627409915}      # t quantile for the paired-t check, n = 10

# §12.7 and §12.8: which verdict cells were NOT blind, and why.  Printed beside
# each verdict so the disclosure travels with the number.
SEEN = {
    ("medium-play", "MR"): "complete and visible when the rule was fixed (§12.7)",
    ("medium-diverse", "MR"): "read informally 2026-10-09, before this readout (§12.8)",
    ("large-play", "MR"): "read informally 2026-10-09, before this readout (§12.8)",
    ("large-diverse", "MR"): "read informally 2026-10-10, before this readout (§12.8)",
    ("medium-play", "PT"): "read informally 2026-10-11, before this readout (§12.8)",
    ("medium-diverse", "PT"): "read informally 2026-10-12, before this readout (§12.8)",
}


# --------------------------------------------------------------------------
# The exact bootstrap
# --------------------------------------------------------------------------
_ENUM = {}


def count_vectors(n):
    """(counts [K, n], probabilities [K]) for every resample of n items drawn
    with replacement, as count vectors.  K = C(2n-1, n)."""
    if n not in _ENUM:
        combos = np.array(list(itertools.combinations_with_replacement(range(n), n)),
                          dtype=np.int64)
        counts = np.zeros((combos.shape[0], n), dtype=np.int64)
        for j in range(n):
            counts[:, j] = (combos == j).sum(axis=1)
        logfact = np.array([math.lgamma(k + 1) for k in range(n + 1)])
        logp = math.lgamma(n + 1) - logfact[counts].sum(axis=1) - n * math.log(n)
        _ENUM[n] = (counts, np.exp(logp))
    return _ENUM[n]


def exact_bootstrap(d, coverage=COVERAGE):
    """Exact percentile bootstrap interval for the mean of paired differences d.

    Returns dict(lower, upper, n_vectors, prob_sum, mean_check, mean).
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    counts, probs = count_vectors(n)
    stat = counts @ d / n
    order = np.argsort(stat, kind="mergesort")
    vals, cum = stat[order], np.cumsum(probs[order])
    tol = 1e-12                                     # floating-point slack on the cumulative
    q_lo, q_hi = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    lower = float(vals[np.searchsorted(cum, q_lo - tol, side="left")])
    upper = float(vals[np.searchsorted(cum, q_hi - tol, side="left")])
    return {"lower": lower, "upper": upper, "n_vectors": int(counts.shape[0]),
            "prob_sum": float(probs.sum()), "mean_check": float((stat * probs).sum()),
            "mean": float(d.mean())}


def paired_t(d, coverage=COVERAGE):
    """(lower, upper) of the paired-t interval.  A CHECK only (§12.5)."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n - 1 in T975 and coverage == 0.95:
        tq = T975[n - 1]
    else:                                            # pragma: no cover
        from scipy import stats
        tq = float(stats.t.ppf(1 - (1 - coverage) / 2, n - 1))
    half = tq * d.std(ddof=1) / math.sqrt(n)
    return float(d.mean() - half), float(d.mean() + half)


def verdict(lower, upper, margin=MARGIN):
    if lower > -margin:
        return "NOT BEHIND"
    if upper < -margin:
        return "TOO FAR BEHIND"
    return "INCONCLUSIVE"


def self_checks(b, d):
    """§12.5's three self-checks.  -> [failures]"""
    bad = []
    if len(d) == 10 and b["n_vectors"] != N_COUNT_VECTORS:
        bad.append(f"{b['n_vectors']} count vectors, expected {N_COUNT_VECTORS}")
    if abs(b["prob_sum"] - 1.0) > 1e-9:
        bad.append(f"probabilities sum to {b['prob_sum']!r}")
    if abs(b["mean_check"] - b["mean"]) > 1e-12:
        bad.append(f"weighted mean {b['mean_check']!r} != plain mean {b['mean']!r}")
    return bad


# --------------------------------------------------------------------------
# Data: one cell = one evaluation sweep = ten runs
# --------------------------------------------------------------------------
def audit_run(cfg, name, seed, fixed, winner, oracle):
    """[problems] for one run's wandb config against the spec (§12.5)."""
    probs = []
    if cfg.get("seed") != seed:
        probs.append(f"seed {cfg.get('seed')!r}")
    if cfg.get("normalize_reward") != winner:
        probs.append(f"normalize_reward {cfg.get('normalize_reward')!r}, stage-4 winner {winner!r}")
    for k, v in fixed.items():
        if k == "reward_model_root":
            continue
        if cfg.get(k) != v:
            probs.append(f"{k}={cfg.get(k)!r}, spec {v!r}")
    if oracle:
        if cfg.get("reward_model_path") not in ("", None) or cfg.get("reward_model_root"):
            probs.append("a reward model is set on a task-reward run")
    else:
        want = f"{fixed['reward_model_root']}_{seed}"
        if cfg.get("reward_model_path") != want:
            probs.append(f"reward_model_path {cfg.get('reward_model_path')!r}, expected {want!r}")
        # The shift (gauge) is max0 for every learned reward, except where the cell's
        # spec declares otherwise: the leveling ablation's unshifted cells, "none" (§14.40).
        want_gauge = fixed.get("gauge_reward", "max0")
        if cfg.get("gauge_reward") != want_gauge:
            probs.append(f"gauge_reward {cfg.get('gauge_reward')!r}, spec {want_gauge!r}")
    return probs


def collect(api, P, winners, sp=None, method_as=None):
    """{(dataset, method): cell} for every evaluation sweep at a registered index.

    sp / method_as (handoff 14.24): read another spec under another method name,
    accepting the index the sweep ran at (method_as given).  Used for the pre-fix
    PT cells, read deliberately as the time-only reward (collect_timeonly).

    cell = dict(sweep, index, scores {seed: last-10}, robust {seed: last-20},
                incomplete [reasons], audit [problems])
    Methods are phase2_sweeps' keys, plus "task_reward_r1" for the oracle at the
    conventional index when stage 4 chose another.
    """
    import iql_score
    sp = sp or P.spec()
    cells, notes = {}, []
    for sw in api.project(P.IQL_PROJECT, entity=P.ENTITY).sweeps():
        params = sw.config.get("parameters")
        hits = P.match_entry(params, sp)
        if len(hits) != 1:
            continue
        name = hits[0]
        flat = P.flat(params)
        if P.lineage_of(flat, P.grid_of(name)) != "eval":
            continue
        index, win = flat["normalize_reward"], winners.get(name)
        oracle = P.is_oracle(name)
        method = method_as or P.method_of(name)
        if method_as:
            pass                                       # the index it ran at (pre-fix PT: its own stage 4)
        elif index != win:
            if oracle and index == P.ORACLE_CONVENTIONAL:
                method = "task_reward_r1"
            else:
                notes.append(f"{sw.id}: {name} at index {index}, not the winner {win} -- ignored")
                continue
        if not hold.shown(method):
            continue                                   # covered by the hold (§14.1): not fetched
        key = (P.dataset_of(name), method)
        cell = {"sweep": sw.id, "index": index, "scores": {}, "robust": {}, "max": {},
                "incomplete": [], "audit": [], "name": name}
        if sw.config.get("program") != P.program_of(name):
            cell["audit"].append(f"program {sw.config.get('program')!r}")
        runs = list(sw.runs)
        seen = set()
        for r in runs:
            s = (r.config or {}).get("seed")
            if s in seen:
                cell["audit"].append(f"seed {s} appears twice")
            seen.add(s)
            cell["audit"] += [f"run {r.id}: {x}" for x in
                              audit_run(r.config or {}, name, s, sp[name], index, oracle)]
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


def collect_timeonly(api, P):
    """The pre-fix PT evaluation cells (handoff §14.10), read deliberately as the
    TIME-ONLY reward, "PT_TIMEONLY": a PT model applied to windows from the
    dataset's first episode, so a reward of time since the episode start.  Matched
    on PT's spec WITHOUT seq_labeller, at the index each ran (its own pre-fix stage 4)."""
    prefix = {n: {k: v for k, v in f.items() if k != "seq_labeller"}
              for n, f in P.spec().items() if n.startswith("pt_sweeps/")}
    return collect(api, P, {}, sp=prefix, method_as="PT_TIMEONLY")


def cell_ready(cell):
    """None if the cell may be read, else the reason it may not."""
    if cell is None:
        return "no evaluation sweep"
    if cell["audit"]:
        return f"CONFIG AUDIT FAILED ({len(cell['audit'])}): {cell['audit'][0]}"
    if cell["incomplete"] or sorted(cell["scores"]) != SEEDS:
        done = len(cell["scores"])
        return f"INCOMPLETE ({done}/10 seeds scored)"
    return None


def compare(a, b, key="scores"):
    """Paired readout of cell a minus cell b.  Both must be ready."""
    d = np.array([a[key][s] - b[key][s] for s in SEEDS])
    bs = exact_bootstrap(d)
    t_lo, t_hi = paired_t(d)
    xa = np.array([a[key][s] for s in SEEDS])
    xb = np.array([b[key][s] for s in SEEDS])
    return {"d": d, "mean": float(d.mean()), "lower": bs["lower"], "upper": bs["upper"],
            "t_lower": t_lo, "t_upper": t_hi, "ahead": int((d > 0).sum()),
            "tied": int((d == 0).sum()),
            "a": (float(xa.mean()), float(xa.std(ddof=1))),
            "b": (float(xb.mean()), float(xb.std(ddof=1))),
            "checks": self_checks(bs, d), "verdict": verdict(bs["lower"], bs["upper"]),
            "t_verdict": verdict(t_lo, t_hi)}


def fmt_cmp(c):
    s = (f"{c['a'][0]:.3f} ± {c['a'][1]:.3f}  vs  {c['b'][0]:.3f} ± {c['b'][1]:.3f}   "
         f"diff {c['mean']:+.3f}   95% exact bootstrap [{c['lower']:+.3f}, {c['upper']:+.3f}]   "
         f"paired-t [{c['t_lower']:+.3f}, {c['t_upper']:+.3f}]   ahead on {c['ahead']}/10")
    return s


def report(cells, notes, out=print):
    datasets = [f"antmaze-{v}-v2" for v in VARIANTS]
    problems = 0
    out("=" * 100)
    out("§12 READOUT -- BNN CVaR against the named comparators, unaltered data, seeds 1-10")
    out(f"statistic last-10 mean; paired by seed; exact bootstrap over {N_COUNT_VECTORS} "
        f"count vectors; margin {MARGIN}")
    out("=" * 100)
    for n in notes:
        out(f"  note: {n}")

    out("\nA. THE EIGHT VERDICTS  (BNN CVaR - comparator; NOT BEHIND if the lower end > "
        f"-{MARGIN})")
    summary = []
    for d in datasets:
        for comp in COMPARATORS:
            a, b = cells.get((d, PROPOSED)), cells.get((d, comp))
            ra, rb = cell_ready(a), cell_ready(b)
            tag = f"  {d:28s} vs {comp:3s}"
            if ra or rb:
                why = "; ".join(f"{n}: {r}" for n, r in ((PROPOSED, ra), (comp, rb)) if r)
                out(f"{tag}  NOT READ -- {why}")
                problems += ("AUDIT" in why)
                summary.append((d, comp, "INCOMPLETE" if "AUDIT" not in why else "AUDIT FAILED"))
                continue
            c = compare(a, b)
            if c["checks"]:
                out(f"{tag}  SELF-CHECK FAILED: {c['checks']}")
                problems += 1
                continue
            out(f"{tag}  {c['verdict']:15s} {fmt_cmp(c)}")
            if c["t_verdict"] != c["verdict"]:
                out(f"{'':38s}!! the paired-t interval would read {c['t_verdict']}; "
                    f"the verdict is the bootstrap's, as registered")
            r = compare(a, b, key="robust")
            out(f"{'':38s}last-20 (robustness check, not a verdict): diff {r['mean']:+.3f} "
                f"[{r['lower']:+.3f}, {r['upper']:+.3f}]")
            v = d.replace("antmaze-", "").replace("-v2", "")
            if (v, comp) in SEEN:
                out(f"{'':38s}NOT A BLIND CELL: {SEEN[(v, comp)]}")
            summary.append((d, comp, c["verdict"]))

    out("\n   All eight are reported; there is no rule combining them (§12.2).")
    out("   " + " | ".join(f"{d.replace('antmaze-', '').replace('-v2', '')}/{c}: {v}"
                           for d, c, v in summary))

    out("\nB. MECHANISM COMPARISONS  (same difference and interval; NOT verdicts, §12.2)")
    for d in datasets:
        a = cells.get((d, PROPOSED))
        for other, label in (("BNN_MEAN", "BNN mean (the risk functional alone)"),
                             ("ENS_CVAR", "ensemble CVaR (how the spread was obtained)"),
                             ("ENS_MEAN", "ensemble mean")):
            b = cells.get((d, other))
            ra, rb = cell_ready(a), cell_ready(b)
            tag = f"  {d:28s} BNN CVaR - {label:44s}"
            if ra or rb:
                out(f"{tag} NOT READ -- " + "; ".join(r for r in (ra, rb) if r))
            else:
                out(f"{tag} {fmt_cmp(compare(a, b))}")

    out("\nC. EVERY METHOD AGAINST THE TASK REWARD  (not verdicts, §12.2)")
    out("   The task reward is at its stage-4 winning index; where that is not the "
        "conventional r - 1,\n   the conventional one is shown as well.")
    for d in datasets:
        for okey, olabel in (("task_reward", "task reward (winning index)"),
                             ("task_reward_r1", "task reward (conventional r - 1)")):
            o = cells.get((d, okey))
            if o is None and okey == "task_reward_r1":
                continue
            for m in (PROPOSED, "BNN_MEAN", "MR", "PT", "ENS_CVAR", "ENS_MEAN"):
                a = cells.get((d, m))
                ra, ro = cell_ready(a), cell_ready(o)
                tag = f"  {d:28s} {m:9s} - {olabel:32s}"
                if ra or ro:
                    out(f"{tag} NOT READ -- " + "; ".join(r for r in (ra, ro) if r))
                else:
                    out(f"{tag} {fmt_cmp(compare(a, o))}")

    out("\nD. THE CELLS  (last-10 mean ± sd over seeds 1-10; sweep; index)")
    for d in datasets:
        for m in ("task_reward", "task_reward_r1", "MR", "PT", "ENS_MEAN", "ENS_CVAR",
                  "BNN_MEAN", PROPOSED):
            c = cells.get((d, m))
            if c is None:
                if m != "task_reward_r1":
                    out(f"  {d:28s} {m:15s} no evaluation sweep")
                continue
            r = cell_ready(c)
            if r:
                out(f"  {d:28s} {m:15s} {c['sweep']}  idx {c['index']}  {r}")
                for x in c["audit"][:3]:
                    out(f"{'':46s}audit: {x}")
            else:
                x = np.array([c["scores"][s] for s in SEEDS])
                out(f"  {d:28s} {m:15s} {c['sweep']}  idx {c['index']}  "
                    f"{x.mean():.3f} ± {x.std(ddof=1):.3f}")
    n_read = sum(1 for _, _, v in summary if v in ("NOT BEHIND", "TOO FAR BEHIND", "INCONCLUSIVE"))
    out(f"\n{n_read} of 8 verdict cells read; {8 - n_read} not yet complete."
        + ("" if not problems else f"  !! {problems} audit or self-check failure(s)"))
    return 1 if problems else 0


# --------------------------------------------------------------------------
def selftest():
    # the enumeration itself
    counts, probs = count_vectors(10)
    assert counts.shape == (N_COUNT_VECTORS, 10) and (counts.sum(axis=1) == 10).all()
    assert abs(probs.sum() - 1.0) < 1e-12
    assert len({tuple(c) for c in counts}) == N_COUNT_VECTORS
    # probability of the resample that takes every seed once is 10!/10^10
    once = probs[(counts == 1).all(axis=1)]
    assert once.size == 1 and abs(once[0] - math.factorial(10) / 10 ** 10) < 1e-15
    # against brute force over ALL n^n ordered resamples, for a small n
    rng = np.random.default_rng(0)
    for n in (3, 4, 5):
        d = rng.normal(size=n)
        means = np.array([d[list(ix)].mean() for ix in itertools.product(range(n), repeat=n)])
        means.sort()
        k = means.size
        for q, name in ((0.025, "lower"), (0.975, "upper")):
            brute = means[int(math.ceil(q * k - 1e-9)) - 1]      # smallest v with P(X<=v) >= q
            got = exact_bootstrap(d)[name]
            assert abs(brute - got) < 1e-12, (n, name, brute, got)
    # the three registered self-checks hold on arbitrary data, and the interval
    # is deterministic and brackets the mean
    d = rng.normal(0.02, 0.05, size=10)
    b = exact_bootstrap(d)
    assert not self_checks(b, d) and b["lower"] < d.mean() < b["upper"]
    assert exact_bootstrap(d) == b
    assert self_checks(dict(b, prob_sum=0.9), d) and self_checks(dict(b, n_vectors=5), d)
    # a constant difference has a degenerate interval at that constant
    c = exact_bootstrap(np.full(10, -0.2))
    assert abs(c["lower"] + 0.2) < 1e-12 and abs(c["upper"] + 0.2) < 1e-12
    # shifting every difference shifts both ends by the same amount
    s = exact_bootstrap(d + 0.3)
    assert abs(s["lower"] - b["lower"] - 0.3) < 1e-12 and abs(s["upper"] - b["upper"] - 0.3) < 1e-12
    # verdicts: the three cases, and the boundary is strict
    assert verdict(-0.05, 0.10) == "NOT BEHIND"
    assert verdict(-0.30, -0.12) == "TOO FAR BEHIND"
    assert verdict(-0.15, -0.05) == "INCONCLUSIVE"
    assert verdict(-0.10, 0.20) == "INCONCLUSIVE"            # lower must be > -0.10
    assert verdict(-0.30, -0.10) == "INCONCLUSIVE"           # upper must be < -0.10
    # paired t against the textbook formula
    lo, hi = paired_t(d)
    half = 2.2621571627409915 * d.std(ddof=1) / math.sqrt(10)
    assert abs(lo - (d.mean() - half)) < 1e-12 and abs(hi - (d.mean() + half)) < 1e-12
    # completeness gate and audit gate
    full = {"scores": {s: 0.5 for s in SEEDS}, "robust": {s: 0.5 for s in SEEDS},
            "incomplete": [], "audit": [], "sweep": "x", "index": 3}
    assert cell_ready(full) is None and cell_ready(None) == "no evaluation sweep"
    nine = dict(full, scores={s: 0.5 for s in SEEDS[:9]}, incomplete=["seed 10 running"])
    assert cell_ready(nine).startswith("INCOMPLETE (9/10")
    assert cell_ready(dict(full, audit=["run a: bnn_alpha=0.5, spec 0.95"])).startswith("CONFIG AUDIT")
    # the audit catches each thing §12.5 names
    fx = {"reward_model_root": "~/r/antmaze_x_bnn_eval", "bnn_reward_model": True,
          "bnn_n_samples": -1, "bnn_alpha": 0.95, "centre_draws": True, "query_length": 1,
          "config_path": "c.yaml"}
    ok = dict(fx, seed=4, normalize_reward=3, gauge_reward="max0",
              reward_model_path="~/r/antmaze_x_bnn_eval_4")
    assert audit_run(ok, "n", 4, fx, 3, False) == []
    for k, v in (("normalize_reward", 2), ("bnn_alpha", 0.0), ("bnn_n_samples", 500),
                 ("centre_draws", False), ("gauge_reward", "none"),
                 ("reward_model_path", "~/r/antmaze_x_bnn_eval_5"), ("seed", 5)):
        assert audit_run(dict(ok, **{k: v}), "n", 4, fx, 3, False), k
    assert audit_run({"seed": 1, "normalize_reward": 1, "config_path": "c.yaml",
                      "reward_model_path": ""}, "n", 1, {"config_path": "c.yaml"}, 1, True) == []
    assert audit_run({"seed": 1, "normalize_reward": 1, "config_path": "c.yaml",
                      "reward_model_path": "~/x"}, "n", 1, {"config_path": "c.yaml"}, 1, True)
    # an end-to-end read on synthetic cells, including the not-read paths
    def cell(vals):
        return {"scores": dict(zip(SEEDS, vals)), "robust": dict(zip(SEEDS, vals)),
                "incomplete": [], "audit": [], "sweep": "s", "index": 3, "name": "n"}
    base = rng.uniform(0.4, 0.6, size=10)
    D = "antmaze-medium-play-v2"
    cells = {(D, PROPOSED): cell(base + 0.05), (D, "MR"): cell(base),
             (D, "PT"): cell(base + 0.30),
             ("antmaze-large-play-v2", PROPOSED): cell(base),
             ("antmaze-large-play-v2", "MR"): nine}
    lines = []
    assert report(cells, [], out=lines.append) == 0
    txt = "\n".join(lines)
    assert "vs MR   NOT BEHIND" in txt and "vs PT   TOO FAR BEHIND" in txt
    assert "NOT READ -- MR: INCOMPLETE (9/10" in txt
    assert "NOT A BLIND CELL" in txt and "2 of 8 verdict cells read" in txt
    c = compare(cells[(D, PROPOSED)], cells[(D, "MR")])
    assert abs(c["mean"] - 0.05) < 1e-12 and c["ahead"] == 10
    # audit_run: the shift is max0 unless the cell's spec declares another (§14.40)
    base = {"config_path": "c", "reward_model_root": "~/r", "query_length": 1}
    cfg = {"seed": 3, "normalize_reward": 2, "config_path": "c", "query_length": 1,
           "reward_model_path": "~/r_3", "gauge_reward": "max0"}
    assert audit_run(cfg, "n", 3, base, 2, False) == []
    assert audit_run(dict(cfg, gauge_reward="none"), "n", 3, base, 2, False)          # unshifted, not declared
    assert audit_run(dict(cfg, gauge_reward="none"), "n", 3, dict(base, gauge_reward="none"), 2, False) == []
    assert audit_run(cfg, "n", 3, dict(base, gauge_reward="none"), 2, False)           # declared none, ran max0
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winners", default=os.path.join(ROOT, "phase2_winners.json"))
    ap.add_argument("--selftest", action="store_true")
    import hold
    hold.parse()                                   # before argparse sees the flag
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    hold.refuse("eval_readout.py", "the §12 verdicts of BNN CVaR, and the BNN and ensemble cells")
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    winners = json.load(open(a.winners))
    cells, notes = collect(wandb.Api(timeout=120), P, winners)
    return report(cells, notes)


if __name__ == "__main__":
    sys.exit(main())
