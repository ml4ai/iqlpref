#!/usr/bin/env python
"""ladder_status.py -- the sweep check's ladder section (handoff §14.80).  DESCRIPTIVE.

Three reports, with the reading rules built in:

  DAMAGE      every IQL sweep (superseded ones are skipped by iql_sweeps, §14.73) created since
              --since that has a FAILED run.  A grid sweep never retries a failed seed, so each
              one listed is a cell to supersede and re-run whole (§14.73).  States only.
  STANDARD    each standard-protocol (with terminals) ladder cell of the axis: its own mean ± sd
              when complete, else its completeness (§13.9: nothing comparative).
  CANDIDATE   each candidate-protocol (without terminals) ladder cell and each rescaling
              companion (§14.78): run STATES ONLY -- finished / total, sweep state.  No score
              is fetched: §14.5 reads a family's candidate arm only in its registered reading
              (results/rq1_readout.py); rescale_readout.py shows companions after it.

    python results/ladder_status.py [--axis sparsity|noise] [--since 2026-10-30]
"""

import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", choices=("sparsity", "noise"), default="sparsity")
    ap.add_argument("--since", default="2026-10-30")
    import hold
    hold.parse()
    a = ap.parse_args()
    import wandb
    _argv, sys.argv = sys.argv, [sys.argv[0]]
    import phase2_sweeps as P
    import exp_readout as X
    sys.argv = _argv
    E = P._conditions()
    sp = P.spec()
    winners = json.load(open(os.path.join(ROOT, "phase2_winners.json")))
    gwin = json.load(open(os.path.join(ROOT, P.GRID_WINNERS)))
    conds = E.SPARSITY if a.axis == "sparsity" else E.NOISE_PRIMARY + E.NOISE_ORACLE
    names = {}                                          # every ladder / companion file of every protocol
    for c in E.SPARSITY + E.NOISE_PRIMARY + E.NOISE_ORACLE:
        for n, (an, f) in P.exp_spec(sp, c, E).items():
            if not P._held(an):
                names[n] = (f, winners.get(an), "standard", c)
        for n, (an, f, i) in P.exp_noterm_spec(sp, c, gwin, E).items():
            names[n] = (f, i, "candidate", c)
    for n, (an, f, k, c) in P.rescale_arm_spec(sp, gwin, E).items():
        names[n] = (f, k, "companion", c or "full")
    fixed = {n: v[0] for n, v in names.items()}
    api = wandb.Api(timeout=120)
    found, damage = collections.defaultdict(list), []
    for sw in P.iql_sweeps(api):
        params = sw.config.get("parameters")
        if params is None:
            continue
        flat = P.flat(params)
        hits = [h for h in P.match_entry(params, fixed) if flat.get("normalize_reward") == names[h][1]]
        runs = list(sw.runs)
        st = collections.Counter(r.state for r in runs)
        if st.get("failed") and runs and min(r.created_at for r in runs) >= a.since:
            damage.append((hits[0] if len(hits) == 1 else f"(unmatched) {hits}", sw.id, sw.state, dict(st)))
        if len(hits) == 1:
            found[hits[0]].append((sw.id, sw.state, st.get("finished", 0), len(runs)))

    print("=" * 100)
    print(f"LADDER STATUS (HANDOFF §14.80), {a.axis}.  Candidate cells and companions: run states only (§14.5).")
    print("=" * 100)
    print(f"\nDAMAGE: sweeps with a failed run, created since {a.since} (superseded ones excluded)")
    for d in sorted(damage) or [("none",)]:
        print("  ", *d)

    print("\nSTANDARD (with terminals): each cell's own mean ± sd when complete (§13.9)")
    cells, _ = X.collect_exp(api, P, winners, a.axis, "standard")
    import eval_readout as R
    for key in sorted(cells):
        c = cells[key]
        why = R.cell_ready(c)
        if why:
            print(f"  {key}  {c['sweep']}  {why}")
        else:
            x = [c["scores"][s] for s in R.SEEDS]
            print(f"  {key}  {c['sweep']}  idx {c['index']}  {statistics.mean(x):.3f} ± {statistics.stdev(x):.3f}")

    for prot in ("candidate", "companion"):
        print(f"\n{prot.upper()}: states only (finished / runs), no scores")
        for n in sorted(names):
            f, k, p, c = names[n]
            if p != prot or (c not in conds and not (prot == "companion" and c == "full")):
                continue
            hits = found.get(n)
            if hits:
                print(f"  {n:70s} idx {k}  " + ", ".join(f"{s} {state} {fin}/{tot}" for s, state, fin, tot in hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
