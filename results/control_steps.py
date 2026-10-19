#!/usr/bin/env python
"""control_steps.py -- steps-to-goal with and without terminals (handoff §13.14.10).

DESCRIPTIVE, no verdict.  The question: when a policy trained WITHOUT terminals
reaches the goal, does it head there (about as many steps as with terminals), or
wander into it (many more)?  The steps statistic is results_table.ipynb's Table 2,
reproduced exactly: per run, the success-weighted mean of avg_steps_to_goal over
the last 10 evaluation points (episodes that fail have no steps; a run with no
success in the window has no value).  Paired by seed, within a reward family.

    python results/control_steps.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_readout as R                                   # noqa: E402
from iql_score import EVAL_POINTS, LAST_N                  # noqa: E402

ENTITY, PROJECT = "champlin-university-of-arizona", "IQL-pref"
METRIC, STEPS = "mean_score", "avg_steps_to_goal"

# (variant, reward): (with terminals = the anchor evaluation sweep, without terminals)
PAIRS = {
    ("medium-play", "MR"): ("7lzll81d", "jk2f7imj"),
    ("medium-play", "BNN CVaR"): ("tuw2u1wl", "nzytopm9"),
    ("medium-diverse", "MR"): ("nr1ceqlk", "znpybqlg"),
    ("medium-diverse", "BNN CVaR"): ("qs64gidp", "k1is90ma"),
}
REFERENCE = {("medium-play", "task reward, stage-4 index 3"): "79ke5lc3",
             ("medium-play", "task reward, conventional r-1"): "nv89vprk",
             ("medium-diverse", "task reward"): "9pbsrxod"}


def run_steps(run, n=LAST_N):
    """(success rate, success-weighted steps) over the last n points; steps None if
    no episode succeeded.  None, None for an incomplete run."""
    if run.state != "finished":
        return None, None
    rows = run.history(keys=[METRIC, STEPS], samples=100000, pandas=False)
    rows = sorted((r for r in rows if r.get(METRIC) is not None), key=lambda r: r["_step"])
    if len(rows) != EVAL_POINTS:
        return None, None
    w = rows[-n:]
    rate = np.array([r[METRIC] for r in w], dtype=float)
    steps = np.array([r.get(STEPS, np.nan) for r in w], dtype=float)
    ok = (rate > 0) & np.isfinite(steps) & (steps >= 0)
    return float(rate.mean()), (float((steps[ok] * rate[ok]).sum() / rate[ok].sum()) if ok.any() else None)


def by_seed(api, sweep_id):
    out = {}
    for r in api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}").runs:
        out[r.config.get("seed")] = run_steps(r)
    return out


def main():
    import wandb
    api = wandb.Api(timeout=120)
    print("STEPS-TO-GOAL, with vs without terminals (success-weighted, last 10 evaluations; DESCRIPTIVE)")
    print("steps are counted over SUCCESSFUL episodes only; 'succ' = last-10 success rate\n")
    for (v, rew), (w_id, wo_id) in PAIRS.items():
        a, b = by_seed(api, w_id), by_seed(api, wo_id)
        both = [s for s in R.SEEDS if a.get(s, (None, None))[1] is not None
                and b.get(s, (None, None))[1] is not None]
        sa = [a[s][1] for s in R.SEEDS if a.get(s, (None, None))[1] is not None]
        sb = [b[s][1] for s in R.SEEDS if b.get(s, (None, None))[1] is not None]
        print(f"  {v:15s} {rew:9s} with terminals    steps {np.mean(sa):6.1f} ± {np.std(sa, ddof=1):5.1f} "
              f"({len(sa)} seeds with a success)   succ {np.mean([a[s][0] for s in R.SEEDS]):.3f}")
        print(f"  {'':15s} {'':9s} without terminals steps {np.mean(sb):6.1f} ± {np.std(sb, ddof=1):5.1f} "
              f"({len(sb)} seeds with a success)   succ {np.mean([b[s][0] for s in R.SEEDS]):.3f}")
        if len(both) >= 3:
            d = np.array([b[s][1] - a[s][1] for s in both])
            line = (f"without - with, paired over {len(both)} seeds: {d.mean():+.1f} steps "
                    f"(seeds: {', '.join(f'{x:+.0f}' for x in d)})")
            if len(both) == 10:
                bs = R.exact_bootstrap(d)
                line += f"   95% exact [{bs['lower']:+.1f}, {bs['upper']:+.1f}]"
            print(f"  {'':15s} {'':9s} {line}")
        print()
    print("REFERENCE: the task reward, with terminals")
    for (v, lab), sid in REFERENCE.items():
        a = by_seed(api, sid)
        sa = [a[s][1] for s in R.SEEDS if a.get(s, (None, None))[1] is not None]
        print(f"  {v:15s} {lab:32s} steps {np.mean(sa):6.1f} ± {np.std(sa, ddof=1):5.1f}   "
              f"succ {np.mean([a[s][0] for s in R.SEEDS]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
