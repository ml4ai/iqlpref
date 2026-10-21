#!/usr/bin/env python
"""hold.py -- the BNN/CVaR hold (gp_reward-priors HANDOFF §14.1; author, 2026-10-20).

The benchmarking chapter shows no result from a method it has not introduced, so
until the author lifts the hold no readout prints a COVERED result:

    covered      BNN CVaR, BNN mean, ensemble CVaR   (every cell, grid, control,
                 diagnostic and readout)
    not covered  MR, PT, ensemble mean, task reward, motion heuristic, trivial
                 rewards, NMR

Every readout imports this module.  Readouts that mix families FILTER the covered
ones out by default; readouts whose every result is covered REFUSE to run.  Either
way, `--include-held` restores the old behaviour, and is for use only once the
author has lifted the hold.

    python results/hold.py --selftest
"""

import os
import sys

COVERED = frozenset({"BNN_CVAR", "BNN_MEAN", "ENS_CVAR"})
FLAG = "--include-held"
INCLUDE = False
BANNER = ("!! " + FLAG + ": printing results covered by the BNN/CVaR hold (HANDOFF §14.1). "
          "Only the author lifts the hold.")


def parse(argv=None):
    """Consume --include-held from argv (sys.argv by default); return INCLUDE."""
    global INCLUDE
    argv = sys.argv if argv is None else argv
    if FLAG in argv:
        while FLAG in argv:
            argv.remove(FLAG)
        INCLUDE = True
        print(BANNER)
    return INCLUDE


def shown(method):
    """True if a result of this method (e.g. "MR", "BNN_CVAR") may be printed."""
    return INCLUDE or method not in COVERED


def shown_kind(kind):
    """Control-cell kinds: noterm_<METHOD>, rerun_<METHOD>, heuristic, ..."""
    for p in ("noterm_", "rerun_"):
        if kind.startswith(p):
            return shown(kind[len(p):])
    return True


def show(methods):
    return tuple(m for m in methods if shown(m))


def held_config(cfg):
    """A run's (or a sweep's flattened) config: is it a covered reward?
    BNN rewards live under *_bnn_eval roots (mean and CVaR alike); the ensemble's
    CVaR is an MR ensemble with mr_alpha > 0."""
    root = str(cfg.get("reward_model_root") or "").lower()
    if "_bnn_" in root or root.endswith("_bnn") or "bnn_eval" in root:
        return True
    if cfg.get("mr_ensemble") and float(cfg.get("mr_alpha") or 0.0) > 0.0:
        return True
    return False


def held_path(path):
    """A sweep or model path: bnn_sweeps/..., ctrl_sweeps/*bnn*, ensemble *cvar*."""
    p = path.replace("\\", "/").lower()
    base = os.path.basename(p)
    if "bnn" in p:
        return True
    return ("ensemble_sweeps/" in p or base.startswith("ensemble")) and "cvar" in base


def refuse(tool, what):
    """For a tool whose every result is covered: exit unless --include-held."""
    if parse():
        return
    sys.exit(f"{tool}: not run. Every result it prints ({what}) is covered by the "
             f"BNN/CVaR hold (gp_reward-priors HANDOFF §14.1). {FLAG} overrides, "
             f"only once the author has lifted the hold.")


def selftest():
    global INCLUDE
    INCLUDE = False
    assert show(("task_reward", "MR", "PT", "ENS_MEAN", "ENS_CVAR", "BNN_MEAN", "BNN_CVAR")) == \
        ("task_reward", "MR", "PT", "ENS_MEAN")
    assert not shown_kind("noterm_BNN_CVAR") and not shown_kind("rerun_BNN_MEAN")
    assert shown_kind("noterm_MR") and shown_kind("heuristic") and shown_kind("noterm_task_reward")
    assert held_config({"reward_model_root": "~/iqlpref/exp/reward_learning/antmaze_large_play_bnn_eval"})
    assert held_config({"reward_model_root": "~/x/antmaze_large_play_bnn_eval/r16", "mr_alpha": 0.0})
    assert held_config({"reward_model_root": "~/x/antmaze_large_play_mr_eval", "mr_ensemble": True,
                        "mr_alpha": 0.95})
    assert not held_config({"reward_model_root": "~/x/antmaze_large_play_mr_eval", "mr_ensemble": True,
                            "mr_alpha": 0.0})
    assert not held_config({"reward_model_root": "~/x/antmaze_large_play_mr_eval"})
    assert not held_config({"reward_model_root": None})
    assert held_path("bnn_sweeps/sweep_antmaze_medium_play_cvar") and held_path(
        "ctrl_sweeps/noterm_antmaze_medium_play_bnn_mean.yaml")
    assert held_path("ensemble_sweeps/sweep_antmaze_large_play_cvar.yaml")
    assert not held_path("ensemble_sweeps/sweep_antmaze_large_play_mean.yaml")
    assert not held_path("ctrl_sweeps/rerun_noterm_antmaze_medium_play_mr.yaml")
    argv = ["x", FLAG, "--other"]
    assert parse(argv) and argv == ["x", "--other"] and shown("BNN_CVAR")
    INCLUDE = False
    print("hold selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
