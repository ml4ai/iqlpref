#!/usr/bin/env python
"""restore_test.py -- can the registered readings be recomputed from the W&B backup alone?
(handoff §14.141)

Runs each readout below UNCHANGED, in its own process, with wandb.Api replaced by the backup
(results/wandb_offline.py) and every network connection refused, then compares its output
with the same readout's output from W&B:
    rq1_sparsity    rq1_readout.py                  RQ1's registered reading, sparsity ladder
    rq1_noise       rq1_readout.py --axis noise     RQ1's registered reading, noise ladder
    control         control_readout.py              the IQL controls (A-I)
    leveling        leveling_readout.py             the shift ablation
    rescale         rescale_readout.py              the rescaling companions, sparsity
    rescale_noise   rescale_readout.py --axis noise the rescaling companions, noise
rq1_readout reads A from the backup's copy of exp/rq1_accuracy.json (--acc), not the
checkout's, so nothing but the backup and the repository's tracked files is used.

The W&B side is, by default, the committed results/*_latest.txt files (the outputs of the
last sweep check); with --live the readout is run against W&B now instead.  Either way runs
that finished between the backup and the comparison change some lines (cells completing,
"ignored" notes for running sweeps), so the report separates:
    IDENTICAL   every line equal once the "note:" lines (which list sweeps still running) and
                wandb's log lines are set aside
    DIFFERS     the differing lines are printed; each must be explained by runs that changed
                after the backup (rerun the backup, then this test, to confirm)

Run ON THE BOX, from ~/iqlpref:
    python results/restore_test.py [--backup DIR] [--live] [--only rq1_sparsity control ...]
"""

import argparse
import difflib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BACKUP = "/media/mule/scratch/loren/wandb_backup"
READOUTS = {                                     # name: (script, args, committed W&B output)
    "rq1_sparsity": ("rq1_readout.py", [], "rq1_readout_latest.txt"),
    "rq1_noise": ("rq1_readout.py", ["--axis", "noise"], "rq1_readout_noise_latest.txt"),
    "control": ("control_readout.py", [], "control_readout_latest.txt"),
    "leveling": ("leveling_readout.py", [], "leveling_readout_latest.txt"),
    "rescale": ("rescale_readout.py", [], None),
    "rescale_noise": ("rescale_readout.py", ["--axis", "noise"], None),
}
# Runs the readout in a child process with W&B replaced by the backup.
OFFLINE = ("import runpy, sys; sys.path.insert(0, {here!r}); import wandb_offline; "
           "wandb_offline.install({backup!r}); sys.argv = {argv!r}; "
           "runpy.run_path({script!r}, run_name='__main__')")


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + (("\n[stderr]\n" + p.stderr) if p.returncode else "")


def offline_output(name, backup):
    script, args, _ = READOUTS[name]
    path = os.path.join(HERE, script)
    if script == "rq1_readout.py":
        acc = os.path.join(backup, "local", "exp", "rq1_accuracy.json")
        if not os.path.exists(acc):
            return 1, f"{acc} not in the backup: rerun results/wandb_backup.py on the box"
        args = args + ["--acc", acc]
    code = OFFLINE.format(here=HERE, backup=backup, argv=[path] + args, script=path)
    return run([sys.executable, "-c", code])


def live_output(name, live):
    script, args, latest = READOUTS[name]
    if live or latest is None:
        return run([sys.executable, os.path.join(HERE, script)] + args)
    p = os.path.join(HERE, latest)
    return (0, open(p).read()) if os.path.exists(p) else (1, f"{latest} not found")


def comparable(text):
    """The lines that do not depend on when the readout ran."""
    return [ln.rstrip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith(("note:", "wandb:", "[stderr]"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backup", default=BACKUP)
    ap.add_argument("--live", action="store_true", help="compare with the readouts run against W&B now")
    ap.add_argument("--only", nargs="*", choices=tuple(READOUTS), help="these readouts only")
    ap.add_argument("--out", help="directory for each readout's offline and W&B outputs (default: none kept)")
    a = ap.parse_args()
    m = json.load(open(os.path.join(a.backup, "MANIFEST.json")))
    print(f"backup {a.backup}: finished {m.get('finished_utc')}, complete {m.get('complete')}, "
          f"{m.get('index', {}).get('runs')} runs; W&B side: {'live now' if a.live else 'committed *_latest.txt'}")
    verdicts = {}
    for name in a.only or READOUTS:
        oc, off = offline_output(name, a.backup)
        lc, liv = live_output(name, a.live)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            open(os.path.join(a.out, f"{name}.offline.txt"), "w").write(off)
            open(os.path.join(a.out, f"{name}.wandb.txt"), "w").write(liv)
        if oc:
            verdicts[name] = "FAILED offline"
            print(f"\n{name}: FAILED reading the backup\n" + off[-3000:])
            continue
        if lc:
            verdicts[name] = "no W&B output to compare"
            print(f"\n{name}: offline OK; no W&B output to compare ({liv.strip()[:200]})")
            continue
        x, y = comparable(liv), comparable(off)
        if x == y:
            verdicts[name] = "IDENTICAL"
            print(f"\n{name}: IDENTICAL ({len(y)} lines)")
        else:
            d = [ln for ln in difflib.unified_diff(x, y, "W&B", "backup", n=0, lineterm="") if not ln.startswith("@@")]
            verdicts[name] = f"DIFFERS ({sum(ln[:1] in '+-' for ln in d[2:])} lines)"
            print(f"\n{name}: DIFFERS -- check each line against runs that changed after the backup")
            print("\n".join(d[:80]) + ("\n  ..." if len(d) > 80 else ""))
    print("\nSUMMARY  " + "  ".join(f"{k}: {v}" for k, v in verdicts.items()))
    return 0 if all(v == "IDENTICAL" for v in verdicts.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
