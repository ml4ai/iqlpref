#!/usr/bin/env python
"""offset_measure.py -- the leveling ablation's offset measurement (handoff §14.40). No IQL runs.

For every reward model of the leveling ablation (medium_diverse; MR, PT, NMR, ensemble
mean; ties kept and removed; seeds 1-10), the offline dataset is labelled EXACTLY as
iql_eval.train() labels it, and measured WITHOUT leveling:
    m, s, min, max     the raw per-transition reward
    m / s              the offset relative to the spread
    R_max - R_min      the trajectory-return range the range normalizations divide by
                       (iql_eval.return_reward_range), and the mean trajectory length H
and, for the family's two fixed indices (its stage-4 winner, used with terminals; its
no-terminals winner, used without), the reward's mean and sd AFTER modify_reward,
unleveled and leveled (gauge max0): the offset each index passes on to IQL.

    measure (on the box; needs d4rl and the models; resumable):
        cd ~/iqlpref && python results/offset_measure.py measure
        -> exp/offset_measurements.json   (exp/ is gitignored; copy it over by hand)
    report (anywhere): python results/offset_measure.py report
    python results/offset_measure.py --selftest

Ties-kept against ties-removed, paired by seed, exact 95% intervals (P4 of §14.40).
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "exp", "offset_measurements.json")
ENV = "antmaze-medium-diverse-v2"
FAMS = ("mr", "pt", "nmr", "ens_mean")
ARMS = ("tiefree", "ties")
SEEDS = tuple(range(1, 11))


def stats(dataset, idx_with, idx_without, gauge, modify, rrange):
    """The measurement for one labelled dataset (a dict as qlearning_dataset_* returns)."""
    r = np.asarray(dataset["rewards"], dtype=np.float64)
    min_ret, max_ret, trj = rrange(dataset, 1000)
    out = {"m": float(r.mean()), "s": float(r.std()), "min": float(r.min()), "max": float(r.max()),
           "m_over_s": float(r.mean() / r.std()) if r.std() > 0 else float("nan"),
           "ret_range": float(max_ret - min_ret), "H": float(np.mean(trj))}
    for tag, idx in (("with", idx_with), ("without", idx_without)):
        if idx is None:
            continue
        for lev in (False, True):
            d = copy.deepcopy({k: dataset[k] for k in ("rewards", "terminals")})
            d["rewards"] = np.asarray(d["rewards"], dtype=np.float32).copy()
            if lev:
                gauge(d, "max0")
            if idx:
                modify(d, ENV, idx)
            x = np.asarray(d["rewards"], dtype=np.float64)
            key = f"idx_{tag}_{'lev' if lev else 'nolev'}"
            out[key] = {"index": int(idx), "mean": float(x.mean()), "sd": float(x.std()),
                        "mean_over_sd": float(x.mean() / x.std()) if x.std() > 0 else float("nan")}
    return out


def model_dir(models_root, fam, seed, arm):
    base = {"mr": "mr", "pt": "pt", "nmr": "nmr", "ens_mean": "mr"}[fam]
    return os.path.join(models_root, f"antmaze_medium_diverse_{base}{'_ties' if arm == 'ties' else ''}_eval_{seed}")


def cmd_measure(a):
    import d4rl  # noqa: F401
    import gym
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    sys.argv = _argv
    winners = json.load(open(os.path.join(ROOT, "phase2_winners.json")))
    gwin = json.load(open(os.path.join(ROOT, P.GRID_WINNERS)))
    anchors = {f: P.LEVEL_FAMILIES[f].format(v="medium_diverse") for f in FAMS}
    env = gym.make(ENV)
    sdim, adim = env.observation_space.shape[0], env.action_space.shape[0]
    raw = env.get_dataset()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for fam in FAMS:
        iw, inw = winners.get(anchors[fam]), gwin.get(anchors[fam])
        for arm in ARMS:
            for s in SEEDS:
                key = f"{fam}|{arm}|{s}"
                if key in res and not a.force:
                    continue
                d = model_dir(a.models_root, fam, s, arm)
                if not os.path.isdir(d):
                    print(f"  {key}: no models at {d}")
                    continue
                try:
                    if fam == "mr":
                        ds = Q.qlearning_dataset_mr(env, Q.load_mlp_reward_model(d, device=a.device), dataset=raw)
                    elif fam == "pt":
                        ds = Q.qlearning_dataset_pt(env, Q.load_pt_reward_model(d, device=a.device), 100, dataset=raw)
                    elif fam == "nmr":
                        ds = Q.qlearning_dataset_pt(env, Q.load_nmr_reward_model(d, sdim, adim, device=a.device), 100,
                                                    dataset=raw)
                    else:
                        ds = Q.qlearning_dataset_mr_ensemble(env, d, alpha=0.0, burn_in=100, device=a.device,
                                                             centre_draws=True, dataset=raw)
                except Exception as e:                       # noqa: BLE001  one model must not stop the run
                    print(f"  {key}: FAILED ({type(e).__name__}: {str(e)[:120]}) -- not recorded")
                    continue
                res[key] = stats(ds, iw, inw, Q.gauge_reward, Q.modify_reward, Q.return_reward_range)
                res[key]["dir"] = d
                print(f"  {key}: m {res[key]['m']:.4g}  s {res[key]['s']:.4g}  m/s {res[key]['m_over_s']:+.3f}  "
                      f"range {res[key]['ret_range']:.4g}")
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"\n  {len(res)} measurement(s) in {a.out}")
    return 0


def cmd_report(a, out=print):
    import eval_readout as R
    res = json.load(open(a.out))
    out("=" * 100)
    out("OFFSET MEASUREMENT, medium_diverse (handoff §14.40, P4).  DESCRIPTIVE.  ties - tie-free, paired by seed.")
    out("=" * 100)
    keys = [("m", "raw mean m"), ("s", "raw sd s"), ("m_over_s", "m / s"), ("ret_range", "return range"),
            ("idx_with_nolev.mean", "mean after the with-terminals index, unleveled"),
            ("idx_without_nolev.mean", "mean after the no-terminals index, unleveled"),
            ("idx_without_nolev.mean_over_sd", "mean/sd after the no-terminals index, unleveled")]
    for fam in FAMS:
        out(f"\n{fam.upper()}")
        for k, lab in keys:
            def get(arm, s):
                v = res.get(f"{fam}|{arm}|{s}")
                for part in k.split("."):
                    v = v.get(part) if isinstance(v, dict) else None
                return v
            got = {arm: [get(arm, s) for s in SEEDS] for arm in ARMS}
            have = {arm: [x for x in got[arm] if x is not None] for arm in ARMS}
            cells = "   ".join(f"{arm} {np.mean(have[arm]):+.4g}" if have[arm] else f"{arm} --" for arm in ARMS)
            line = f"  {lab:50s} {cells}"
            if all(x is not None for arm in ARMS for x in got[arm]):
                d = np.array(got["ties"]) - np.array(got["tiefree"])
                bs = R.exact_bootstrap(d)
                line += f"   ties - tie-free {d.mean():+.4g} [{bs['lower']:+.4g}, {bs['upper']:+.4g}]"
            out(line)
    return 0


def selftest():
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import ast
    src = open(os.path.join(ROOT, "algorithms", "offline", "iql_eval.py")).read()
    fns = {n.name: ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}
    g = {"np": np}
    exec("\n\n".join(fns[k] for k in ("return_reward_range", "gauge_reward", "modify_reward")), g)
    rng = np.random.default_rng(0)
    N = 4000
    term = np.zeros(N, bool)
    term[999::1000] = True
    r = (rng.normal(size=N) - 3.0).astype(np.float32)                      # a learned reward with offset -3
    st = stats({"rewards": r, "terminals": term}, 3, 7, g["gauge_reward"], g["modify_reward"],
               g["return_reward_range"])
    assert abs(st["m"] - r.mean()) < 1e-5 and abs(st["m_over_s"] - r.mean() / r.std()) < 1e-5
    assert st["H"] == 1000.0 and st["ret_range"] > 0          # each transition carries its trajectory's length
    # index 7 subtracts min_ret/len per step: with equal trajectory lengths the constant offset cancels
    # up to the trajectory with the smallest return, so its unleveled mean is much nearer -1 than index 3's
    assert abs(st["idx_without_nolev"]["mean"] + 1) < abs(st["idx_with_nolev"]["mean"] + 1)
    # leveling (max0) then index 3 differs from unleveled by exactly the scaled max
    assert st["idx_with_lev"]["index"] == 3 and st["idx_without_lev"]["index"] == 7
    assert model_dir("/m", "ens_mean", 4, "ties") == "/m/antmaze_medium_diverse_mr_ties_eval_4"
    assert model_dir("/m", "pt", 10, "tiefree") == "/m/antmaze_medium_diverse_pt_eval_10"
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure")
    m.add_argument("--models-root", default=os.path.join(ROOT, "exp", "reward_learning"))
    m.add_argument("--out", default=OUT)
    m.add_argument("--device", default="cuda")
    m.add_argument("--force", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "measure":
        return cmd_measure(a)
    if a.cmd == "report":
        return cmd_report(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
