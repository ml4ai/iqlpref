#!/usr/bin/env python
"""iql_score.py — the canonical IQL score: mean of the last 10 evaluation points.

One definition, used for BOTH stage-4 normalization selection and reported
results (gp_reward-priors/HANDOFF_HP_SELECTION.md §4.3.107, §5, §7.1).  Selection
and reporting must use the identical statistic; import this module rather than
re-deriving the number.

Definition
----------
An IQL run trains 1,000,000 steps and evaluates every 5,000 steps: 200 evaluation
points, each `mean_score` = the success rate over 100 episodes.

    score(run) = mean of `mean_score` over the LAST 10 evaluation points
                 (the last 50,000 training steps; 1,000 episodes)

Robustness check: the same with the last 20 points.  Why the last 10 (decided
2026-09-15): an end-of-training statistic is what an offline method can deliver
without online checkpoint selection.  n = 10 was fixed from the learning curves'
own shape and noise, never from method comparisons: the end-of-training window is
flat, and a longer window cuts between-run variance by only ~5% while adding
drift.  The mean, not the median, because it is the success rate pooled over
1,000 episodes.  The old statistic, the max over all 200 points, is spike-driven
and resolved half as many method comparisons.  It is kept here only as
`statistic="max"`, to reproduce and disclose earlier numbers.

Validity rules
--------------
* History is sorted by step, so "last" means last in training, whatever order
  wandb returns it in.
* A run must have exactly EVAL_POINTS evaluation points.  Crashed, still-running
  or truncated runs are NOT scored on a partial curve: they are reported and
  excluded.  `require_complete=False` disables this, for inspection only.
* A NaN score raises: a missing evaluation must be investigated, not averaged
  around.

Stage 4
-------
`select_normalization(runs)` groups a seed-0 grid by `normalize_reward` and picks
the index with the highest score.  With one run per index (§5: stage 4 exists to
avoid a bad normalization, not to find the best one), a duplicate index or a
missing one is an error.  Exact ties go to the LOWEST index.  Scores move in steps
of 0.0001, so ties are rare but possible; lowest-index is a fixed, disclosed rule,
not a judgment.

Usage
-----
    python results/iql_score.py champlin-university-of-arizona/IQL-pref/<sweep_id> [...]
    python results/iql_score.py --stage4 champlin-university-of-arizona/IQL-pref/<sweep_id>
    python results/iql_score.py --selftest
"""

import argparse
import math
import sys

import numpy as np

METRIC = "mean_score"
EVAL_POINTS = 200
LAST_N = 10
ROBUST_N = 20
STATISTICS = ("last_n_mean", "max")


class IncompleteRun(ValueError):
    """Raised when a run does not have the full EVAL_POINTS evaluation curve."""


def score_curve(scores, n=LAST_N, statistic="last_n_mean",
                expected_points=EVAL_POINTS, require_complete=True):
    """Score one evaluation curve, given in TRAINING ORDER."""
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise IncompleteRun("empty evaluation curve")
    if np.isnan(arr).any():
        raise ValueError(f"{int(np.isnan(arr).sum())} NaN evaluation point(s)")
    if require_complete and arr.size != expected_points:
        raise IncompleteRun(f"{arr.size} evaluation points, expected {expected_points}")
    if statistic == "last_n_mean":
        if arr.size < n:
            raise IncompleteRun(f"{arr.size} evaluation points, fewer than n = {n}")
        return float(arr[-n:].mean())
    if statistic == "max":
        return float(arr.max())
    raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}")


def run_curve(run, metric=METRIC, samples=100000):
    """(steps, scores) for one wandb run, sorted by step.

    Uses the sampled-history endpoint with a sample budget far above the number of
    evaluation rows, so every evaluation point is returned.
    """
    rows = run.history(keys=[metric], samples=samples, pandas=False)
    rows = sorted((r for r in rows if r.get(metric) is not None), key=lambda r: r["_step"])
    return [r["_step"] for r in rows], [r[metric] for r in rows]


def score_run(run, n=LAST_N, statistic="last_n_mean", metric=METRIC, require_complete=True):
    """dict(id, name, state, n_points, score, error) for one wandb run."""
    out = {"id": run.id, "name": run.name, "state": run.state,
           "normalize_reward": (run.config or {}).get("normalize_reward"),
           "seed": (run.config or {}).get("seed"), "n_points": 0, "score": None, "error": None}
    try:
        steps, scores = run_curve(run, metric)
        out["n_points"] = len(scores)
        if require_complete and run.state != "finished":
            raise IncompleteRun(f"state {run.state!r}")
        out["score"] = score_curve(scores, n=n, statistic=statistic,
                                   require_complete=require_complete)
    except (IncompleteRun, ValueError) as e:
        out["error"] = str(e)
    return out


def sweep_runs(api, sweep_path):
    return list(api.sweep(sweep_path).runs)


def select_normalization(scored, expected_indices=range(8)):
    """Stage 4: pick the normalize_reward index with the highest score.

    scored: score_run() dicts from ONE seed-0 grid.  Returns (winner_index, table),
    where table is sorted best-first.  Raises if an index is duplicated, missing,
    or unscored, since each must be decided on exactly one complete run (§5).
    """
    by_idx = {}
    for s in scored:
        i = s["normalize_reward"]
        if i in by_idx:
            raise ValueError(f"normalize_reward {i} appears more than once ({by_idx[i]['id']}, {s['id']})")
        by_idx[i] = s
    missing = sorted(set(expected_indices) - set(by_idx))
    if missing:
        raise ValueError(f"normalize_reward indices missing: {missing}")
    bad = {i: s["error"] for i, s in by_idx.items() if s["score"] is None}
    if bad:
        raise ValueError(f"unscored indices (fix or rerun before selecting): {bad}")
    table = sorted(by_idx.values(), key=lambda s: (-s["score"], s["normalize_reward"]))
    return table[0]["normalize_reward"], table


def _selftest():
    rng = np.random.default_rng(0)
    curve = list(rng.uniform(0, 1, EVAL_POINTS))
    assert math.isclose(score_curve(curve), float(np.mean(curve[-10:])))
    assert math.isclose(score_curve(curve, n=ROBUST_N), float(np.mean(curve[-20:])))
    assert math.isclose(score_curve(curve, statistic="max"), max(curve))
    for bad, exc in ((curve[:150], IncompleteRun), ([], IncompleteRun),
                     (curve[:-1] + [float("nan")], ValueError)):
        try:
            score_curve(bad)
            raise AssertionError("should have raised")
        except exc:
            pass
    assert math.isclose(score_curve(curve[:150], require_complete=False), float(np.mean(curve[140:150])))
    runs = [{"id": f"r{i}", "normalize_reward": i, "score": s, "error": None}
            for i, s in enumerate([0.40, 0.55, 0.55, 0.10, 0.0, 0.0, 0.30, 0.31])]
    win, table = select_normalization(runs)
    assert win == 1 and [t["normalize_reward"] for t in table[:2]] == [1, 2]   # tie -> lowest index
    for broken in (runs[:-1], runs + [dict(runs[0])],
                   [dict(r, score=None, error="x") if r["normalize_reward"] == 3 else r for r in runs]):
        try:
            select_normalization(broken)
            raise AssertionError("should have raised")
        except ValueError:
            pass
    print("iql_score self-test: PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Validity rules")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="*", metavar="entity/project/sweep_id")
    ap.add_argument("--n", type=int, default=LAST_N)
    ap.add_argument("--statistic", choices=STATISTICS, default="last_n_mean")
    ap.add_argument("--stage4", action="store_true", help="select normalize_reward on each sweep")
    ap.add_argument("--allow-incomplete", action="store_true", help="inspection only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return 0
    if not a.sweeps:
        ap.error("give at least one sweep, or --selftest")
    import wandb
    api = wandb.Api(timeout=300)
    label = f"last-{a.n} mean" if a.statistic == "last_n_mean" else "max over evals (LEGACY)"
    for path in a.sweeps:
        scored = [score_run(r, n=a.n, statistic=a.statistic, require_complete=not a.allow_incomplete)
                  for r in sweep_runs(api, path)]
        print(f"\n=== {path}  [{label}] ===")
        for s in sorted(scored, key=lambda s: (s["normalize_reward"] is None, s["normalize_reward"] or 0, s["seed"] or 0)):
            val = f"{s['score']:.4f}" if s["score"] is not None else f"EXCLUDED ({s['error']})"
            print(f"  {s['id']:<10} seed {str(s['seed']):>4}  idx {str(s['normalize_reward']):>4}  "
                  f"points {s['n_points']:>3}  {val}")
        ok = [s["score"] for s in scored if s["score"] is not None]
        if ok:
            print(f"  n = {len(ok)} scored ({len(scored) - len(ok)} excluded): mean {np.mean(ok):.4f}"
                  + (f", sd {np.std(ok, ddof=1):.4f}" if len(ok) > 1 else ""))
        if a.stage4:
            try:
                win, table = select_normalization(scored)
                gap = table[0]["score"] - table[1]["score"]
                print(f"  STAGE 4 WINNER: normalize_reward = {win}  ({table[0]['score']:.4f}; "
                      f"runner-up idx {table[1]['normalize_reward']} at {table[1]['score']:.4f}, gap {gap:.4f})")
            except ValueError as e:
                print(f"  STAGE 4: cannot select -- {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
