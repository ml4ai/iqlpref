#!/usr/bin/env python
"""calibration_clite.py -- C-lite: what makes the BNN's logits about 2x too small?  (gp_reward-priors
HANDOFF §14.99, §14.102, §14.103, §14.164; built §14.177)

Stages A-B3 and the round-5 selection read (§14.93-§14.103) leave two layers of under-scaling:
a common factor of about 2 on the whole reward, from the FIXED design every trial shares (the
sampler's burn-in and schedule, or the functional prior as a whole), and an extra factor on the
within-cell (motion) part, attributed to the prior's design-fixed nugget.  C-lite separates them
causally: ONE variant and seed, 8 chains, the production config with ONE change per arm.

    base      the production config unchanged (8 chains: chains 0-7 of the production run,
              which are deterministic in (seed, chain index) -- checked against stage B)
    burnin4   num_burn_in_steps x 4                       tests the sampler (candidate 2)
    nmeas0    n_meas 0: likelihood-only SGHMC             tests the functional prior as a whole
    nugget10  map_sig_n2 x 10                             tests the nugget (the within-cell layer)

Reading guide (fixed before any C-lite run, §14.177): compare each arm's logit scale with base's,
on the same 107 validation pairs, as the ratio of median |d| (and the fitted temperature):
    burn-in x 4 lifts BOTH parts        -> the sampler does not reach the posterior's scale
    n_meas 0 lifts BOTH parts           -> the functional prior as a whole holds the logits down
    only nugget x 10 lifts the WITHIN-CELL part -> the nugget causes layer 2 (§14.102)
    none lifts                          -> neither; the remaining candidate is the likelihood
                                           weighting (§14.99 B2 found it standard)
One variant and one seed: a mechanism probe, not an estimate.  DESCRIPTIVE, POST HOC.

SCOPE: §14.92's scoped exception to the BNN/CVaR hold -- the BNN reward models' calibration
diagnostics only.  No IQL result is read, and C-lite trains no model any IQL run uses: its runs
log to the W&B project BNN-clite and write under exp/calibration_clite/ (box-generated,
gitignored).  The analysis text belongs in gp_reward-priors/calibration_results/held_bnn_cvar/.

    run (ON THE BOX, from ~/iqlpref; trains the arms not yet done, one GPU each, at once; then
         measures them; resumable):
        python results/calibration_clite.py run --gpus 0 1 2 3
    measure (box; also the last step of run):
        python results/calibration_clite.py measure
    analyse (here, after copying exp/calibration_clite/meas/ over):
        /opt/anaconda3/envs/irl/bin/python results/calibration_clite.py analyse \\
            > gp_reward-priors/calibration_results/held_bnn_cvar/mechanism_clite.txt
    /opt/anaconda3/envs/irl/bin/python results/calibration_clite.py --selftest
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPR = os.path.join(ROOT, "gp_reward-priors")
for p in (HERE, ROOT, GPR, os.path.join(GPR, "scripts_bnn")):
    sys.path.insert(0, p)

VARIANT, SEED, CHAINS = "medium_diverse", 1, 8      # the lowest BNN temperature of the four (§14.98)
OUT = os.path.join(ROOT, "exp", "calibration_clite")
MEAS_DIR = os.path.join(OUT, "meas")
MODELS = os.path.join(ROOT, "exp", "reward_learning")             # MR's anchor model (B3's reference)
STAGE_B = os.path.join(ROOT, "exp", "calibration_b")              # the production run, measured in stage B
TRAINER = os.path.join("scripts_bnn", "run_bnn_training_antmaze_eval.py")
CONFIG = os.path.join(GPR, "scripts_bnn", f"antmaze_{VARIANT}_bnn_antmaze_eval.yaml")
PROJECT, GROUP = "BNN-clite", "C-lite"
ARMS = ("base", "burnin4", "nmeas0", "nugget10")


def production():
    import yaml
    return yaml.safe_load(open(CONFIG))


def arm_overrides(arm, prod):
    """The ONE change each arm makes to the production config (pyrallis flags)."""
    return {"base": {}, "burnin4": {"num_burn_in_steps": 4 * int(prod["num_burn_in_steps"])},
            "nmeas0": {"n_meas": 0}, "nugget10": {"map_sig_n2": 10 * float(prod["map_sig_n2"])}}[arm]


def arm_root(arm):
    return os.path.join(OUT, arm)


def train_cmd(arm, prod, py=sys.executable):
    """The trainer's command line, as train_rewards.sh builds it for one BNN job (absolute paths;
    run from the submodule root), plus 8 chains on one GPU, C-lite's own output and W&B project,
    and the arm's one change."""
    meas = prod["measurement_dataset"]
    meas = meas if os.path.isabs(meas) else os.path.join(GPR, meas)
    cmd = [py, TRAINER, "--config_path", CONFIG, "--data_root", os.path.join(GPR, "data", "antmaze"),
           "--measurement_dataset", meas, "--seed", str(SEED),
           "--num_chains", str(CHAINS), "--chains_per_gpu", str(CHAINS),
           "--OUT_DIR", os.path.join(arm_root(arm), f"antmaze_{VARIANT}_bnn_eval"),
           "--project", PROJECT, "--group", GROUP, "--name", f"clite_{arm}"]
    for k, x in arm_overrides(arm, prod).items():
        cmd += [f"--{k}", str(x)]
    return cmd


# --------------------------------------------------------------------------
# run / measure (box)
# --------------------------------------------------------------------------
def cmd_run(a):
    prod = production()
    todo = [arm for arm in ARMS if not os.path.exists(os.path.join(arm_root(arm), "DONE"))]
    if len(a.gpus) < len(todo):
        sys.exit(f"!! {len(todo)} arm(s) to train need {len(todo)} GPUs; got {a.gpus}")
    procs = {}
    for arm, g in zip(todo, a.gpus):
        os.makedirs(arm_root(arm), exist_ok=True)
        log = open(os.path.join(arm_root(arm), "train.log"), "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = train_cmd(arm, prod, a.python)
        print(f"[{time.strftime('%FT%T')}] {arm} on GPU {g}: {' '.join(cmd[2:])}", flush=True)
        procs[arm] = subprocess.Popen(cmd, cwd=GPR, env=env, stdout=log, stderr=subprocess.STDOUT)
    for arm, p in procs.items():
        rc = p.wait()
        mark = "DONE" if rc == 0 else "FAILED"
        open(os.path.join(arm_root(arm), mark), "w").write(f"exit {rc} {time.strftime('%FT%T')}\n")
        print(f"[{time.strftime('%FT%T')}] {arm}: {mark} (exit {rc})", flush=True)
    return cmd_measure(a)


def cmd_measure(a):
    import torch
    import calibration_stage_b as B
    import calibration_stage_b3 as B3
    torch.set_num_threads(a.threads)
    os.makedirs(MEAS_DIR, exist_ok=True)
    jobs = [("mr", lambda: B.measure_base("mr", VARIANT, SEED, a.device, MODELS), "json")]
    for arm in ARMS:
        if os.path.exists(os.path.join(arm_root(arm), "DONE")):
            jobs.append((f"{arm}_b", lambda arm=arm: B.measure_bnn(VARIANT, SEED, a.device, MODELS, arm_root(arm)), "json"))
            jobs.append((f"{arm}_b3", lambda arm=arm: B3.measure_one(VARIANT, SEED, a.device, MODELS, arm_root(arm)), "npz"))
        else:
            print(f"  {arm}: not trained (no DONE marker) -- not measured")
    for name, fn, kind in jobs:
        path = os.path.join(MEAS_DIR, f"{name}.{kind}")
        if os.path.exists(path) and not a.force:
            continue
        try:
            rec = fn()
        except Exception as e:                               # noqa: BLE001  one arm must not stop the rest
            print(f"  {name}: FAILED ({type(e).__name__}: {str(e)[:160]})")
            continue
        if kind == "json":
            tmp = path + ".tmp"
            json.dump(rec, open(tmp, "w"))
        else:
            tmp = path + ".tmp.npz"
            np.savez_compressed(tmp, **{k: np.asarray(x) for k, x in rec.items()})
        os.replace(tmp, path)
        print(f"  {name}: measured -> {path}")
    return 0


# --------------------------------------------------------------------------
# analyse (here)
# --------------------------------------------------------------------------
def scale_ratio(d, ref):
    """median |d| / |ref| over pairs: one model's logit scale against another's, pair by pair."""
    d, ref = np.abs(np.asarray(d, float)), np.abs(np.asarray(ref, float))
    return float(np.median(d / np.maximum(ref, 1e-9)))


def arm_stats(b, b3, mr):
    import calibration_mechanism as CM
    import calibration_stage_b as B
    import calibration_stage_b3 as B3
    dec = B.chain_decomposition(b["chain_mean"], b["chain_var"], b["chain_n"])
    d, y = dec["dbar"], np.asarray(b["y1"]) > 0.5
    st = B3.seed_stats(b3)
    return {"d": d, "acc": float(((d > 0) == y).mean()), "T": CM.fit_temperature(d, y),
            "ce": CM.ce(CM.sig(d), y), "vs_mr": scale_ratio(d, mr["d"]),
            "parts_vs_mr": {k: st["ratio"][k] for k in ("d_pos", "d_within", "sd_pos", "sd_within")},
            "parts": {k: st["bnn"][k] for k in ("d_pos", "d_within")},
            "b3_gap": float(np.abs(st["bnn"]["d"] - d).max()),
            "between_share": dec["between_share"], "chains": len(b["chain_n"])}


def verdict(rel):
    """The reading guide applied to the arms' scale relative to base: rel[arm] = (whole, pos, within)."""
    lift = lambda x: x >= 1.25                               # a lift of 25% or more counts as a lift
    out = []
    if "burnin4" in rel and lift(rel["burnin4"][1]) and lift(rel["burnin4"][2]):
        out.append("burn-in x 4 lifts both parts: the SAMPLER does not reach the posterior's scale")
    if "nmeas0" in rel and lift(rel["nmeas0"][1]) and lift(rel["nmeas0"][2]):
        out.append("n_meas 0 lifts both parts: the FUNCTIONAL PRIOR as a whole holds the logits down")
    if "nugget10" in rel and lift(rel["nugget10"][2]) and not lift(rel["nugget10"][1]):
        out.append("only nugget x 10 lifts the within-cell part: the NUGGET causes layer 2")
    return out or ["no arm lifts the logits by 25% or more in the pattern of a candidate"]


def cmd_analyse(a, out=print):
    def load(name, kind):
        p = os.path.join(a.inp, f"{name}.{kind}")
        if not os.path.exists(p):
            return None
        return json.load(open(p)) if kind == "json" else dict(np.load(p, allow_pickle=True))

    out("BNN H3 CALIBRATION, C-LITE (handoff §14.177; scoped hold exception; post hoc, descriptive)")
    out(f"{VARIANT}, seed {SEED}, {CHAINS} chains per arm; the same validation pairs in every arm; MR = the anchor MR model")
    mr = load("mr", "json")
    if mr is None:
        out("  MR's measurement missing -- nothing read")
        return 1
    S = {}
    for arm in ARMS:
        b, b3 = load(f"{arm}_b", "json"), load(f"{arm}_b3", "npz")
        if b is None or b3 is None:
            out(f"  {arm:9s} not measured -- not read")
            continue
        S[arm] = arm_stats(b, b3, mr)
        if S[arm]["b3_gap"] >= 1e-3:
            out(f"  {arm:9s} !! B3's pooled mean differs from stage B's by {S[arm]['b3_gap']:.2e} -- not read")
            del S[arm]
    if "base" not in S:
        out("  base not read -- nothing is compared")
        return 1
    prod = os.path.join(a.stage_b, f"bnn_{VARIANT}_{SEED}.json")
    if os.path.exists(prod):
        pc = np.asarray(json.load(open(prod))["chain_mean"], float)[:CHAINS]
        bc = np.asarray(load("base_b", "json")["chain_mean"], float)
        gap = float(np.abs(pc - bc).max()) if pc.shape == bc.shape else float("nan")
        out(f"  check: base's {CHAINS} chains vs the production run's chains 0-{CHAINS - 1} (stage B), max |difference| "
            f"{gap:.2e}" + ("" if gap < 1e-3 else "  (not bit-identical: base is still the comparison arm; noted)"))
    else:
        out(f"  check: stage B's production measurement not found ({prod}); base not compared with it")
    out(f"\n  {'arm':9s} {'change':28s} {'acc':>6s} {'T':>6s} {'CE':>6s}  |d|/|d_MR|: whole  positional  within-cell")
    pr = production()
    for arm, s in S.items():
        ch = ", ".join(f"{k} {v}" for k, v in arm_overrides(arm, pr).items()) or "none (production)"
        pv = s["parts_vs_mr"]
        out(f"  {arm:9s} {ch:28s} {s['acc']:6.3f} {s['T']:6.3f} {s['ce']:6.3f}  {s['vs_mr']:17.2f}  "
            f"{pv['d_pos']:10.2f}  {pv['d_within']:10.2f}")
    b0 = S["base"]
    rel = {}
    out("\n  each arm against base (median |d_arm| / |d_base| over the same pairs):  whole  positional  within-cell")
    for arm, s in S.items():
        if arm == "base":
            continue
        rel[arm] = (scale_ratio(s["d"], b0["d"]), scale_ratio(s["parts"]["d_pos"], b0["parts"]["d_pos"]),
                    scale_ratio(s["parts"]["d_within"], b0["parts"]["d_within"]))
        out(f"  {arm:9s} {rel[arm][0]:56.2f}  {rel[arm][1]:10.2f}  {rel[arm][2]:10.2f}")
    out("\n  reading guide (§14.177, fixed before the runs; a lift = a ratio of 1.25 or more):")
    for line in verdict(rel):
        out(f"    {line}")
    return 0


def selftest():
    prod = {"num_burn_in_steps": 20000, "map_sig_n2": 0.001, "measurement_dataset": "data/x.hdf5"}
    assert arm_overrides("base", prod) == {}
    assert arm_overrides("burnin4", prod) == {"num_burn_in_steps": 80000}
    assert arm_overrides("nmeas0", prod) == {"n_meas": 0}
    assert abs(arm_overrides("nugget10", prod)["map_sig_n2"] - 0.01) < 1e-15
    # every arm's command differs from base's in its one change only, and never writes a production model
    base = train_cmd("base", prod, "PY")
    for arm in ARMS:
        c = train_cmd(arm, prod, "PY")
        extra = c[len(base):]
        assert c[:len(base)] == [x.replace("clite_base", f"clite_{arm}").replace(f"{os.sep}base{os.sep}", f"{os.sep}{arm}{os.sep}")
                                 for x in base], arm
        assert len(extra) == (0 if arm == "base" else 2), (arm, extra)
        od = c[c.index("--OUT_DIR") + 1]
        assert od.startswith(OUT) and "reward_learning" not in od, od
        assert c[c.index("--project") + 1] == PROJECT and c[c.index("--num_chains") + 1] == str(CHAINS)
        assert c[c.index("--seed") + 1] == str(SEED)
    assert train_cmd("base", prod, "PY")[train_cmd("base", prod, "PY").index("--measurement_dataset") + 1] == os.path.join(GPR, "data/x.hdf5")
    # the real production config: the values the arms multiply
    pr = production()
    assert int(pr["num_burn_in_steps"]) == 20000 and float(pr["map_sig_n2"]) == 0.001, pr
    assert pr["n_meas"] == 256 and pr["gp_prior_type"] == "map_informed"
    # scale ratios and the reading guide
    rng = np.random.default_rng(0)
    d = rng.normal(0, 1, 200)
    assert abs(scale_ratio(2 * d, d) - 2) < 1e-12 and abs(scale_ratio(d, d) - 1) < 1e-12
    v = verdict({"burnin4": (2, 2, 2), "nmeas0": (1, 1, 1), "nugget10": (1.1, 1.0, 1.1)})
    assert len(v) == 1 and "SAMPLER" in v[0], v
    v = verdict({"burnin4": (1, 1, 1), "nmeas0": (1, 1, 1), "nugget10": (1.4, 1.0, 1.8)})
    assert len(v) == 1 and "NUGGET" in v[0], v
    v = verdict({"burnin4": (1, 1, 1), "nmeas0": (1.1, 1.3, 1.1), "nugget10": (1.4, 1.3, 1.8)})
    assert v[0].startswith("no arm"), v
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("run", "measure"):
        m = sub.add_parser(name)
        m.add_argument("--device", default="cpu")
        m.add_argument("--threads", type=int, default=4)
        m.add_argument("--force", action="store_true", help="re-measure arms already measured")
        if name == "run":
            m.add_argument("--gpus", nargs="+", required=True, help="one GPU per arm still to train")
            m.add_argument("--python", default=sys.executable, help="the interpreter for the trainer")
    n = sub.add_parser("analyse")
    n.add_argument("--inp", default=MEAS_DIR)
    n.add_argument("--stage-b", default=STAGE_B)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "measure":
        return cmd_measure(a)
    if a.cmd == "analyse":
        return cmd_analyse(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
