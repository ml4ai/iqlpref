#!/usr/bin/env python
"""leak_measure.py -- do normalization indices 6 and 7 put the goal back into the reward?
(handoff §14.82; DESCRIPTIVE; no IQL runs.)  Run ON THE BOX from ~/iqlpref (needs D4RL and
the reward models).

iql_eval builds a reward, shifts it (gauge max0), normalizes it (modify_reward) and only then
drops the terminals.  At indices 6 and 7 modify_reward subtracts min_ret / trj_len from every
transition, where trj_len is the length of the transition's "trajectory" as
return_reward_range cuts the data: at every TERMINAL, i.e. every goal arrival (§13.13.1), or
every 1000 steps.  Then it divides by (max_ret - min_ret) and multiplies by 1000.  So the
index-7 reward is

    r7 = 1000 * r / R  +  1000 * (-min_ret) / (trj_len * R)  -  1,    R = max_ret - min_ret

the reward model's own part plus a BONUS that grows as the segment shortens.  Consecutive
goal steps are all terminals, so each is a length-1 segment and gets the largest bonus.

For each variant and reward (the motion reward; MR's anchor models, seeds 1-3), exactly as
iql_eval labels and shifts it, this reports:
  - the bonus's mean and share of total bonus mass on TERMINAL (goal) transitions vs the rest;
  - the fraction of transitions in length-1 segments, and their share of terminals;
  - the spread (sd) of the reward's own part against the bonus's;
  - the bonus's cell map against BFS distance to the evaluation goal (Spearman rho; negative
    = rises toward the goal), and the goal cell's rank;
  - the same bonus computed with segments cut every 1000 steps only (no terminals): what a
    leak-free index 7 would add.

    python results/leak_measure.py > exp/leak_measure.txt 2>&1
    python results/leak_measure.py --selftest
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "gp_reward-priors"))

H = 1000                                                # modify_reward's max_episode_steps


def bonus_terms(rewards, terminals, H=H, use_terminals=True):
    """(own part, bonus, trj_len, min_ret, max_ret) of the index-7 reward (before its - 1),
    mirroring iql_eval.return_reward_range / modify_reward index 7.  With use_terminals=False
    the segments are cut every H steps only: the leak-free variant."""
    r = np.asarray(rewards, dtype=np.float64)
    d = np.asarray(terminals).astype(bool) if use_terminals else np.zeros(len(r), dtype=bool)
    returns, trj = [], np.zeros(len(r))
    ep_ret, ep_len, i = 0.0, 0, 0
    for j in range(len(r)):
        ep_ret += r[j]
        ep_len += 1
        trj[i:j + 1] = ep_len
        if d[j] or ep_len == H:
            i = j + 1
            returns.append(ep_ret)
            ep_ret, ep_len = 0.0, 0
    lo, hi = min(returns), max(returns)
    R = hi - lo
    return H * r / R, H * (-lo) / (trj * R), trj, lo, hi


def summarize(own, bonus, trj, term):
    term = np.asarray(term).astype(bool)
    one = trj == 1
    return {"bonus_term_mean": float(bonus[term].mean()) if term.any() else float("nan"),
            "bonus_rest_mean": float(bonus[~term].mean()),
            "bonus_mass_on_terminals": float(bonus[term].sum() / bonus.sum()),
            "share_terminals": float(term.mean()),
            "share_len1": float(one.mean()),
            "len1_that_are_terminals": float(term[one].mean()) if one.any() else float("nan"),
            "own_sd": float(own.std()), "bonus_sd": float(bonus.std())}


def run(a):
    import d4rl  # noqa: F401
    import gym
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    from optbnn.gp.maze_layouts import extract_maze_from_env
    from goal_structure import bfs_distance, cell_means, cells_of, goal_rho
    for v in a.variants.split():
        name = f"antmaze-{v.replace('_', '-')}-v2"
        env = gym.make(name)
        free, scaling, offset = extract_maze_from_env(name)
        u = env.unwrapped
        goal_xy = np.asarray(getattr(u, "target_goal"), dtype=float).reshape(1, -1)[:, :2]
        gr, gc = cells_of(goal_xy, scaling, offset)
        dist = bfs_distance(free, (int(gr[0]), int(gc[0])))
        print(f"\n=== {name}: goal cell ({gr[0]}, {gc[0]})")
        raw = env.get_dataset()
        rewards = {"motion": None}
        for s in [int(x) for x in a.seeds.split()]:
            rewards[f"MR seed {s}"] = s
        for lab, s in rewards.items():
            if s is None:
                ds = d4rl.qlearning_dataset(env)
                ds["rewards"] = Q.heuristic_rewards(ds, "path_length")
            else:
                m = Q.load_mlp_reward_model(os.path.join("exp", "reward_learning", f"antmaze_{v}_mr_eval_{s}"),
                                            device=a.device)
                ds = Q.qlearning_dataset_mr(env, m, dataset=raw)
            Q.gauge_reward(ds, "max0")
            term = np.asarray(ds["terminals"]).astype(bool)
            # the decomposition must reproduce modify_reward exactly
            chk = {k: np.array(ds[k], copy=True) for k in ("rewards", "terminals")}
            Q.modify_reward(chk, name, 7)
            own, bonus, trj, lo, hi = bonus_terms(ds["rewards"], term)
            err = float(np.abs(own + bonus - 1.0 - chk["rewards"]).max())
            st = summarize(own, bonus, trj, term)
            rr, cc = cells_of(np.asarray(ds["observations"])[:, :2], scaling, offset)
            rho_b, rank_b, k = goal_rho(cell_means(bonus, rr, cc), dist)
            rho_o, rank_o, _ = goal_rho(cell_means(own, rr, cc), dist)
            own_f, bonus_f, _, _, _ = bonus_terms(ds["rewards"], term, use_terminals=False)
            print(f"  {lab:12s} min_ret {lo:.4g} max_ret {hi:.4g}  (index-7 decomposition max error {err:.2g})")
            print(f"    bonus: on terminals mean {st['bonus_term_mean']:.4g}, elsewhere mean {st['bonus_rest_mean']:.4g};"
                  f" {100 * st['bonus_mass_on_terminals']:.1f}% of bonus mass on the {100 * st['share_terminals']:.2f}%"
                  f" of transitions that are terminals")
            print(f"    length-1 segments: {100 * st['share_len1']:.2f}% of transitions, "
                  f"{100 * st['len1_that_are_terminals']:.1f}% of them terminals")
            print(f"    sd: reward's own part {st['own_sd']:.4g}, bonus {st['bonus_sd']:.4g}")
            print(f"    cell maps vs BFS distance: bonus rho {rho_b:+.3f} (goal cell rank {rank_b} of {k}); "
                  f"own part rho {rho_o:+.3f} (goal cell rank {rank_o} of {k})")
            print(f"    leak-free (cut every {H} steps only): bonus sd {bonus_f.std():.4g}, "
                  f"on terminals mean {bonus_f[term].mean():.4g}, elsewhere {bonus_f[~term].mean():.4g}")
    return 0


def selftest():
    # 20 steps: a 10-step stretch, then 3 consecutive terminals, then 7 more; H = 1000
    r = -np.ones(20) * 0.1
    term = np.zeros(20, dtype=bool)
    term[10:13] = True
    own, bonus, trj, lo, hi = bonus_terms(r, term)
    # segments: [0..10] (11 steps, ends at the first terminal), [11], [12], [13..19] (7)
    assert list(trj[:11]) == [11] * 11 and trj[11] == 1 and trj[12] == 1 and list(trj[13:]) == [7] * 7
    assert np.isclose(lo, -1.1) and np.isclose(hi, -0.1)
    assert bonus[11] > 10 * bonus[0] and np.argmax(bonus) in (11, 12)          # length-1 goal steps dominate
    st = summarize(own, bonus, trj, term)
    assert st["bonus_mass_on_terminals"] > 0.5 and st["len1_that_are_terminals"] == 1.0
    # the decomposition reproduces iql_eval's index 7 exactly (the same arithmetic, re-done here)
    r7 = (r - lo / trj) / (hi - lo) * H - 1.0
    assert np.allclose(own + bonus - 1.0, r7)
    # leak-free: no terminal cuts -> equal fixed-length segments -> a constant bonus
    r2 = -0.1 * np.arange(1, 21) / 20                   # two 10-step segments with different returns
    _, bf, tf, _, _ = bonus_terms(r2, term, H=10, use_terminals=False)
    assert np.isfinite(bf).all() and np.allclose(bf, bf[0]) and (tf == 10).all()
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="medium_play medium_diverse large_play large_diverse")
    ap.add_argument("--seeds", default="1 2 3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    return selftest() if a.selftest else run(a)


if __name__ == "__main__":
    sys.exit(main())
