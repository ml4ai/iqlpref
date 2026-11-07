#!/usr/bin/env python
"""normtest_readout.py -- the normalization test's REGISTERED decision (gp_reward-priors HANDOFF
§14.104, as amended by §14.108).  Which rule the replication (both arms) and Studies 2-3 use.

Fixed before any run.  Arms: S (Study 1's selection, re-run leak-free: its grid winner on reward
model 0 / IQL seed 50), N (index 0), F (index 2), O3 (one index per protocol x dataset, the best
mean over the four families on S's grids; run only if N and F both fail).  Units: 40 (dataset,
evaluation seed) per family, paired across arms -- reward model s with IQL seed 50 + s, s = 1-10,
shown as units 1-10.  Families MR, PT, NMR, ENS_MEAN (no BNN, §14.1).  Last-10 scores.

PER PROTOCOL, for each alternative A (exact stratified 95% intervals of §14.5, pooled over the
datasets; per-dataset intervals beside):
  (i)   no family disadvantaged: for EVERY family, the lower end of S_A - S_S is above -0.10
  (ii)  every family disadvantaged alike: for EVERY pair (f, g), the per-unit contrast
        (S_A,f - S_A,g) - (S_S,f - S_S,g) lies wholly inside (-0.10, +0.10)
  FLOOR (§14.108, on (ii)): every family clears the zero reward under A -- per family, S_A,f -
        S_zero > 0 by the one-sided exact Wilcoxon at 0.025 over the 40 units (test (c)'s form)
  A PASSES the protocol if (i) holds, or failing that (ii) and the floor hold.
DECISION: one rule for both protocols (§14.108): the first of N, F, O3 that passes under BOTH;
O3 is run only if N and F both fail; if every alternative fails, S.

GATE: nothing of an arm's evaluation cells is printed until every cell the decision reads is
complete (§14.104's do-not): states only until then.

    python results/normtest_readout.py
    python results/normtest_readout.py --selftest
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import eval_readout as R                                   # noqa: E402
import exp_readout as X                                    # noqa: E402

MARGIN = 0.10
ALPHA = 0.025
UNITS = list(range(1, 11))
ARMS = ("N", "F", "O3")


def delta(groups):
    return X.unit_delta([u for g in groups for u in g], X.SCORE_LATTICE)[0]


def interval(groups):
    """Exact stratified 95% interval over the datasets' unit lists."""
    gl = [list(g) for g in groups.values()]
    return X.stratified_exact(gl, delta(gl))


def units(cell):
    return [cell["scores"][u] for u in UNITS]


def judge(arm_cells, s_cells, zero, fams, variants):
    """One protocol, one alternative.  arm_cells / s_cells: {(fam, v): cell}; zero: {v: cell}.
    -> dict(passes, i, ii, floor, detail)."""
    out = {"i": {}, "ii": {}, "floor": {}}
    for f in fams:
        g = {v: np.subtract(units(arm_cells[(f, v)]), units(s_cells[(f, v)])) for v in variants}
        e = interval(g)
        out["i"][f] = {"e": e, "per": {v: R.exact_bootstrap(list(g[v])) for v in variants},
                       "ok": e["lower"] > -MARGIN}
        z = [u for v in variants for u in np.subtract(units(arm_cells[(f, v)]), units(zero[v]))]
        t = X.signed_rank_exact(z, "greater")
        out["floor"][f] = {"t": t, "ok": t["p"] < ALPHA}
    for f, g_ in itertools.combinations(fams, 2):
        c = {v: (np.subtract(units(arm_cells[(f, v)]), units(arm_cells[(g_, v)]))
                 - np.subtract(units(s_cells[(f, v)]), units(s_cells[(g_, v)]))) for v in variants}
        e = interval(c)
        out["ii"][(f, g_)] = {"e": e, "ok": e["lower"] > -MARGIN and e["upper"] < MARGIN}
    out["i_ok"] = all(x["ok"] for x in out["i"].values())
    out["ii_ok"] = all(x["ok"] for x in out["ii"].values())
    out["floor_ok"] = all(x["ok"] for x in out["floor"].values())
    out["passes"] = out["i_ok"] or (out["ii_ok"] and out["floor_ok"])
    out["how"] = "(i)" if out["i_ok"] else "(ii) + floor" if out["passes"] else "fails"
    return out


def decide(results, o3_run):
    """results: {arm: {protocol: judge() or None}} -> (rule, reason)."""
    for arm in ARMS:
        if arm == "O3" and not o3_run:
            if all(not all(results[a][p]["passes"] for p in results[a]) for a in ("N", "F")):
                return None, "N and F both fail: run O3 (normtest.py write --o3 after `winners`), then re-read"
            continue
        r = results.get(arm)
        if r and all(x["passes"] for x in r.values()):
            how = ", ".join(f"{p}: {x['how']}" for p, x in r.items())
            return arm, f"{arm} passes under both protocols ({how})"
    return "S", "no alternative passes under both protocols: S (leak-free selection at seed 100, §14.83 decision 4)"


# --------------------------------------------------------------------------
def collect(api, P, NT):
    """({(fam, v, prot, idx): cell}, {(v, prot): zero cell}, notes)."""
    import iql_score
    sp = P.spec()
    fixed_of, kind = {}, {}
    for v in P.VARIANTS:
        for prot in NT.PROTOCOLS:
            for fam in NT.ANCHORS:
                n = f"{fam}|{v}|{prot}"
                fixed_of[n], kind[n] = NT.family_fixed(sp, fam, v, prot), ("learned", fam, v, prot)
            n = f"zero|{v}|{prot}"
            fixed_of[n], kind[n] = NT.zero_fixed(sp, v, prot), ("zero", None, v, prot)
    cells, zeros, notes = {}, {}, []
    for sw in P.iql_sweeps(api):
        params = sw.config.get("parameters")
        hits = P.match_entry(params, fixed_of)
        if len(hits) != 1:
            continue
        flat = P.flat(params)
        if flat.get("seed") != NT.EVAL_SEEDS:
            continue                                  # S's grids (seed 50), or the replication's zero cells
        k, fam, v, prot = kind[hits[0]]
        idx = flat.get("normalize_reward")
        if k == "zero" and idx != 0:
            notes.append(f"{sw.id}: zero cell at index {idx} -- ignored")
            continue
        fixed = fixed_of[hits[0]]
        cell = {"sweep": sw.id, "index": idx, "scores": {}, "robust": {}, "max": {}, "incomplete": [], "audit": []}
        seen = set()
        for r in sw.runs:
            cfg = r.config or {}
            s = cfg.get("seed")
            if s not in NT.EVAL_SEEDS or s in seen:
                cell["audit"].append(f"seed {s} {'repeated' if s in seen else 'outside 51-60'}")
                continue
            seen.add(s)
            for key, want in fixed.items():
                if key != "reward_model_root" and cfg.get(key) != want:
                    cell["audit"].append(f"seed {s}: {key}={cfg.get(key)!r}, spec {want!r}")
            if cfg.get("normalize_reward") != idx:
                cell["audit"].append(f"seed {s}: normalize_reward {cfg.get('normalize_reward')}")
            if k == "learned":
                want = f"{fixed['reward_model_root']}_{s - NT.OFFSET}"
                if cfg.get("reward_model_path") != want:
                    cell["audit"].append(f"seed {s}: reward_model_path {cfg.get('reward_model_path')!r}, expected {want!r}")
            elif cfg.get("reward_model_path") not in ("", None):
                cell["audit"].append(f"seed {s}: a reward model on the zero reward")
            if r.state != "finished":
                cell["incomplete"].append(f"seed {s} {r.state}")
                continue
            try:
                _, cv = iql_score.run_curve(r)
                cell["scores"][s - NT.OFFSET] = iql_score.score_curve(cv, n=iql_score.LAST_N)
            except (iql_score.IncompleteRun, ValueError) as e:
                cell["incomplete"].append(f"seed {s}: {e}")
        missing = [s for s in NT.EVAL_SEEDS if s - NT.OFFSET not in cell["scores"] and
                   not any(x.startswith(f"seed {s}") for x in cell["incomplete"])]
        if missing:
            cell["incomplete"].append(f"seeds not started: {missing}")
        key = (v, prot) if k == "zero" else (fam, v, prot, idx)
        target = zeros if k == "zero" else cells
        if key in target:
            notes.append(f"{key}: two sweeps ({target[key]['sweep']}, {sw.id}) -- NOT read")
            target[key]["audit"].append(f"second sweep {sw.id}")
            continue
        target[key] = cell
    return cells, zeros, notes


def run(cells, zeros, gwin, NT, variants, out=print):
    fams = [NT.METHOD[f] for f in NT.ANCHORS]
    o3_run = any(NT.o3_key(v, p) in gwin for v in variants for p in NT.PROTOCOLS) and all(
        (f, v, p, gwin.get(NT.o3_key(v, p))) in cells for f in NT.ANCHORS for v in variants for p in NT.PROTOCOLS)
    need = []
    for p in NT.PROTOCOLS:
        for v in variants:
            need.append(("zero", v, p, zeros.get((v, p))))
            for f in NT.ANCHORS:
                s = gwin.get(NT.grid_name(v, f, p))
                need.append((f"{f} S (idx {s})", v, p, cells.get((f, v, p, s)) if s is not None else None))
                for arm, k in NT.ARM_INDEX.items():
                    need.append((f"{f} {arm}", v, p, cells.get((f, v, p, k))))
    why = [f"{w} {v} {p}: {R.cell_ready(c) or ''}" for w, v, p, c in need if R.cell_ready(c)]
    out("=" * 100)
    out("THE NORMALIZATION TEST (HANDOFF §14.104, §14.108) -- REGISTERED.  One rule for both protocols.")
    out("=" * 100)
    if why:
        out(f"  NOT READ -- {len(why)} of {len(need)} required cells not ready; nothing of the arms is shown (§14.104):")
        for w in why[:12]:
            out(f"    {w}")
        if len(why) > 12:
            out(f"    ... {len(why) - 12} more")
        return None
    results = {}
    for arm in ARMS:
        if arm == "O3" and not o3_run:
            continue
        results[arm] = {}
        for p in NT.PROTOCOLS:
            A = {(NT.METHOD[f], v): cells[(f, v, p, NT.ARM_INDEX[arm] if arm in NT.ARM_INDEX else gwin[NT.o3_key(v, p)])]
                 for f in NT.ANCHORS for v in variants}
            S = {(NT.METHOD[f], v): cells[(f, v, p, gwin[NT.grid_name(v, f, p)])] for f in NT.ANCHORS for v in variants}
            Z = {v: zeros[(v, p)] for v in variants}
            results[arm][p] = judge(A, S, Z, fams, variants)
            j = results[arm][p]
            out(f"\n{arm} ({'index ' + str(NT.ARM_INDEX[arm]) if arm in NT.ARM_INDEX else 'O3 indices'}), "
                f"{'WITH' if p == 'std' else 'WITHOUT'} terminals: {j['how'].upper() if j['passes'] else 'FAILS'}")
            for f in fams:
                e, fl = j["i"][f]["e"], j["floor"][f]["t"]
                out(f"  (i)    {f:9s} S_A - S_S {e['mean']:+.3f} [{e['lower']:+.3f}, {e['upper']:+.3f}]  "
                    f"{'ok' if j['i'][f]['ok'] else 'NO'}   | floor S_A - S_zero p = {fl['p']:.3g} {'ok' if j['floor'][f]['ok'] else 'NO'}")
                out("         per dataset " + ", ".join(f"{v} {x['mean']:+.3f} [{x['lower']:+.3f}, {x['upper']:+.3f}]"
                                                    for v, x in j["i"][f]["per"].items()))
            for (f, g_), x in j["ii"].items():
                out(f"  (ii)   {f} - {g_}: {x['e']['mean']:+.3f} [{x['e']['lower']:+.3f}, {x['e']['upper']:+.3f}]  "
                    f"{'ok' if x['ok'] else 'NO'}")
    rule, reason = decide(results, o3_run)
    out(f"\nDECISION: {rule or 'not yet'} -- {reason}")
    out("\nBESIDE: each arm's mean ± sd (last-10, units 1-10) and S's selected indices")
    for p in NT.PROTOCOLS:
        for v in variants:
            for f in NT.ANCHORS:
                s = gwin[NT.grid_name(v, f, p)]
                row = [f"S(idx {s}) {np.mean(units(cells[(f, v, p, s)])):.3f} ± {np.std(units(cells[(f, v, p, s)]), ddof=1):.3f}"]
                for arm, k in NT.ARM_INDEX.items():
                    c = cells[(f, v, p, k)]
                    row.append(f"{arm} {np.mean(units(c)):.3f} ± {np.std(units(c), ddof=1):.3f}")
                out(f"  {p:4s} {v:15s} {NT.METHOD[f]:9s} " + "   ".join(row))
    return rule


def selftest():
    import types
    rng = np.random.default_rng(0)
    NT = types.SimpleNamespace(PROTOCOLS=("std", "cand"), ANCHORS={"mr": 0, "pt": 0, "nmr": 0, "ens_mean": 0},
                               METHOD={"mr": "MR", "pt": "PT", "nmr": "NMR", "ens_mean": "ENS_MEAN"},
                               ARM_INDEX={"N": 0, "F": 2}, grid_name=lambda v, f, p: f"g|{v}|{f}|{p}",
                               o3_key=lambda v, p: f"o3|{v}|{p}")
    V = ("medium-play", "large-play")

    def cell(m):
        return {"scores": {u: float(round(x, 3)) for u, x in zip(UNITS, np.clip(m + rng.normal(0, 0.01, 10), 0, 1))},
                "incomplete": [], "audit": [], "sweep": "s", "index": 0}
    base = {"mr": 0.6, "pt": 0.5, "nmr": 0.55, "ens_mean": 0.62}

    def make(n_shift, f_level):
        cells, zeros, gwin = {}, {}, {}
        for p in NT.PROTOCOLS:
            for v in V:
                zeros[(v, p)] = cell(0.0)
                for f, b in base.items():
                    gwin[NT.grid_name(v, f, p)] = 5
                    cells[(f, v, p, 5)] = cell(b)                          # S
                    cells[(f, v, p, 0)] = cell(b + n_shift.get(f, 0.0))    # N
                    cells[(f, v, p, 2)] = cell(f_level if f_level is not None else b)   # F
        return cells, zeros, gwin
    # N equal to S: passes by (i), adopted
    lines = []
    cells, zeros, gwin = make({}, None)
    assert run(cells, zeros, gwin, NT, V, out=lines.append) == "N", lines[-12:]
    # N hurts one family by 0.3: fails (i); its contrasts break (ii); F equal to S: F adopted
    cells, zeros, gwin = make({"pt": -0.3}, None)
    assert run(cells, zeros, gwin, NT, V, out=lines.append) == "F"
    # N hurts PT; F collapses every family to ~0 (alike, but not above the zero reward): the floor
    # stops (ii) from passing a degenerate rule.  S scores well (so (i) fails) with small family gaps
    # (so every contrast -(S_f - S_g) lies inside +-0.10 and (ii) alone would pass).
    base.update({"mr": 0.42, "pt": 0.40, "nmr": 0.41, "ens_mean": 0.43})
    cells, zeros, gwin = make({"pt": -0.3}, 0.0)
    j = judge({(NT.METHOD[f], v): cells[(f, v, "cand", 2)] for f in NT.ANCHORS for v in V},
              {(NT.METHOD[f], v): cells[(f, v, "cand", 5)] for f in NT.ANCHORS for v in V},
              {v: zeros[(v, "cand")] for v in V}, list(NT.METHOD.values()), V)
    assert not j["i_ok"] and j["ii_ok"] and not j["floor_ok"] and not j["passes"], (j["i_ok"], j["ii_ok"], j["floor_ok"])
    lines = []
    r = run(cells, zeros, gwin, NT, V, out=lines.append)
    assert r is None and any("DECISION: not yet" in x and "run O3" in x for x in lines), lines[-20:]
    # the gate: one incomplete cell and nothing is read
    cells, zeros, gwin = make({}, None)
    cells[("pt", "large-play", "std", 0)]["incomplete"] = ["seed 3 running"]
    lines = []
    assert run(cells, zeros, gwin, NT, V, out=lines.append) is None and any("NOT READ" in x for x in lines)
    assert not any("S_A - S_S" in x for x in lines)
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    import normtest as NT
    sys.argv = _argv
    path = os.path.join(ROOT, NT.WINNERS)
    gwin = json.load(open(path)) if os.path.exists(path) else {}
    cells, zeros, notes = collect(wandb.Api(timeout=120), P, NT)
    for n in notes:
        print(f"  note: {n}")
    run(cells, zeros, gwin, NT, P.VARIANTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
