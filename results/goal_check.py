#!/usr/bin/env python
"""goal_check.py -- which goal do the D4RL AntMaze datasets' rewards and terminals mark?

Two hypotheses, tested directly on the raw datasets (env.get_dataset(), no qlearning_dataset
filtering):
  EVAL       reward = 1 (and terminal) exactly when the ant is within 0.5 of the environment's
             fixed TARGET goal -- the evaluation goal -- whatever the data-collecting ant was
             commanded to reach.
  COMMANDED  reward = 1 when the ant is within 0.5 of ITS OWN commanded goal (the dataset's
             `infos/goal`, one per segment of the collection run).

For each dataset this reports:
  - the dataset's keys, sizes, and whether rewards and terminals coincide;
  - the environment's target goal (as gym.make sets it, with its small jitter), and the
    centroid and spread of every rewarded position -- if the rewards mark one place, the
    rewarded positions sit in one small disc;
  - for each hypothesis, a confusion table over every transition: rewarded or not, against
    within 0.5 of that hypothesis's goal or not (both the position before the step and after
    it, since the reward may be computed on either);
  - the commanded goals: how many distinct ones, how many transitions come within 0.5 of
    their OWN commanded goal, and how many of those are rewarded.  Under COMMANDED nearly all
    are; under EVAL only those whose commanded goal happens to sit at the target are.

Run ON THE BOX (needs d4rl), from ~/iqlpref:
    python results/goal_check.py > exp/goal_check.txt 2>&1
    python results/goal_check.py --selftest
"""

import argparse
import sys

import numpy as np

RADIUS = 0.5
VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")


def confusion(rewarded, near):
    """{'rew&near', 'rew&far', 'norew&near', 'norew&far'} counts."""
    r, n = np.asarray(rewarded, bool), np.asarray(near, bool)
    return {"rew&near": int((r & n).sum()), "rew&far": int((r & ~n).sum()),
            "norew&near": int((~r & n).sum()), "norew&far": int((~r & ~n).sum())}


def agreement(c):
    """Fraction of transitions where 'rewarded' equals 'near the goal'."""
    tot = sum(c.values())
    return (c["rew&near"] + c["norew&far"]) / tot if tot else float("nan")


def analyse(obs, rewards, terminals, timeouts, target, goals=None):
    """Pure: arrays from the raw dataset -> dict of findings."""
    xy = np.asarray(obs)[:, :2]
    rew = np.asarray(rewards) > 0
    term = np.asarray(terminals).astype(bool)
    # the position after each step: the next row, except across an episode boundary (timeout
    # or the last row), where the next row belongs to another episode -- those rows are dropped
    nxt = np.vstack([xy[1:], xy[-1:]])
    end = np.asarray(timeouts).astype(bool) | (np.arange(len(xy)) == len(xy) - 1)
    keep = ~end
    out = {"n": int(len(xy)), "rewarded": int(rew.sum()), "terminals": int(term.sum()),
           "reward_eq_terminal": bool(np.array_equal(rew, term)), "target": np.asarray(target, float)}
    if rew.any():
        rp = xy[rew]
        c = rp.mean(0)
        out["rew_centroid"] = c
        out["rew_spread_max"] = float(np.linalg.norm(rp - c, axis=1).max())
        out["rew_dist_to_target_max"] = float(np.linalg.norm(rp - out["target"], axis=1).max())
    d_eval_pre = np.linalg.norm(xy - out["target"], axis=1)
    d_eval_post = np.linalg.norm(nxt - out["target"], axis=1)
    out["eval_pre"] = confusion(rew[keep], (d_eval_pre <= RADIUS)[keep])
    out["eval_post"] = confusion(rew[keep], (d_eval_post <= RADIUS)[keep])
    if goals is not None:
        g = np.asarray(goals)[:, :2]
        out["n_goals"] = int(len(np.unique(np.round(g, 1), axis=0)))
        d_cmd_pre = np.linalg.norm(xy - g, axis=1)
        d_cmd_post = np.linalg.norm(nxt - g, axis=1)
        out["cmd_pre"] = confusion(rew[keep], (d_cmd_pre <= RADIUS)[keep])
        out["cmd_post"] = confusion(rew[keep], (d_cmd_post <= RADIUS)[keep])
        at_target = np.linalg.norm(g - out["target"], axis=1) <= 2.0          # commanded goal ~ the target
        near_own = (d_cmd_post <= RADIUS) & keep
        out["near_own_goal"] = int(near_own.sum())
        out["near_own_goal_rewarded"] = int((near_own & rew).sum())
        out["near_own_goal_elsewhere"] = int((near_own & ~at_target).sum())
        out["near_own_goal_elsewhere_rewarded"] = int((near_own & ~at_target & rew).sum())
        out["cmd_goal_at_target_share"] = float(at_target.mean())
    return out


def report(name, o, out=print):
    out(f"\n=== {name}: {o['n']} transitions, {o['rewarded']} rewarded, {o['terminals']} terminals; "
        f"rewards == terminals: {o['reward_eq_terminal']}")
    t = o["target"]
    out(f"  env target goal (this gym.make, jittered): ({t[0]:.3f}, {t[1]:.3f})")
    if "rew_centroid" in o:
        c = o["rew_centroid"]
        out(f"  rewarded positions: centroid ({c[0]:.3f}, {c[1]:.3f}); farthest from the centroid "
            f"{o['rew_spread_max']:.3f}; farthest from the env target {o['rew_dist_to_target_max']:.3f}")
    for key, lab in (("eval_pre", "EVAL, position before the step"), ("eval_post", "EVAL, position after the step"),
                     ("cmd_pre", "COMMANDED, before"), ("cmd_post", "COMMANDED, after")):
        if key in o:
            c = o[key]
            out(f"  {lab:34s} agreement {agreement(c):.5f}   rewarded&near {c['rew&near']}, rewarded&far "
                f"{c['rew&far']}, unrewarded&near {c['norew&near']}")
    if "n_goals" in o:
        out(f"  commanded goals (infos/goal): about {o['n_goals']} distinct; {100 * o['cmd_goal_at_target_share']:.1f}% "
            f"of transitions have a commanded goal within 2.0 of the target")
        out(f"  transitions within 0.5 of their OWN commanded goal: {o['near_own_goal']}, of which rewarded "
            f"{o['near_own_goal_rewarded']}")
        out(f"    ... whose commanded goal is AWAY from the target: {o['near_own_goal_elsewhere']}, of which rewarded "
            f"{o['near_own_goal_elsewhere_rewarded']}   (COMMANDED predicts all; EVAL predicts none)")


def run():
    import d4rl  # noqa: F401
    import gym
    print("Which goal do the AntMaze datasets' rewards mark?  (results/goal_check.py; raw env.get_dataset())")
    for v in VARIANTS:
        name = f"antmaze-{v.replace('_', '-')}-v2"
        env = gym.make(name)
        ds = env.get_dataset()
        print(f"\n--- {name}: dataset keys {sorted(ds.keys())}")
        target = np.asarray(env.unwrapped.target_goal, float).reshape(-1)[:2]
        goals = ds.get("infos/goal")
        o = analyse(ds["observations"], ds["rewards"], ds["terminals"], ds["timeouts"], target, goals)
        report(name, o)
    return 0


def selftest():
    rng = np.random.default_rng(0)
    n = 5000
    xy = rng.uniform(0, 20, size=(n, 2))
    target = np.array([18.0, 18.0])
    xy[100:140] = target + rng.uniform(-0.3, 0.3, size=(40, 2))          # 40 steps at the target
    goals = np.repeat(rng.uniform(0, 20, size=(50, 2)), n // 50, axis=0)   # commanded goals, 100 rows each
    xy[2000:2030] = goals[2000] + rng.uniform(-0.2, 0.2, size=(30, 2))    # 30 steps at an own goal elsewhere
    obs = np.concatenate([xy, np.zeros((n, 27))], 1)
    timeouts = np.zeros(n, bool)
    timeouts[999::1000] = True
    # EVAL world: reward on the pre-step position near the target
    rew = (np.linalg.norm(xy - target, axis=1) <= RADIUS).astype(float)
    o = analyse(obs, rew, rew > 0, timeouts, target, goals)
    assert o["reward_eq_terminal"] and agreement(o["eval_pre"]) == 1.0, o["eval_pre"]
    assert o["near_own_goal_elsewhere"] > 0 and o["near_own_goal_elsewhere_rewarded"] == 0, o
    assert agreement(o["cmd_post"]) < agreement(o["eval_pre"])
    # COMMANDED world: reward near each transition's own goal
    rew2 = (np.linalg.norm(xy - goals, axis=1) <= RADIUS).astype(float)
    o2 = analyse(obs, rew2, rew2 > 0, timeouts, target, goals)
    assert agreement(o2["cmd_pre"]) == 1.0 and o2["near_own_goal_elsewhere_rewarded"] > 0, o2
    assert o2["rew_spread_max"] > o["rew_spread_max"]                      # rewards scattered, not one disc
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    return selftest() if a.selftest else run()


if __name__ == "__main__":
    sys.exit(main())
