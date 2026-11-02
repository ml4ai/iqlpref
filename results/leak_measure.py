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

THE LEAK AUDIT (--audit; handoff §14.83, registered for the corrected candidate protocol)
  Criterion: the normalized reward must NOT DEPEND ON THE TERMINALS.  For every variant and
  every index 0-7, modify_reward run on the dataset's real terminals and on all-zero
  terminals must agree EXACTLY (max |difference| = 0), under the protocol's split.  The
  goal can reach a transition through the normalization only via the terminals, and the
  check is made on rewards that are not the learned ones -- the task reward, the motion
  reward and three Gaussian draws, each shifted as iql_eval shifts a learned reward -- so
  passing it holds for any reward model.  Index 0 is iql_eval's "no modify_reward".
  Reported beside: the residual sd of the best global affine fit of the normalized reward
  on its input (relative to the normalized reward's sd), which is 0 when the index is one
  global map.  Both splits are printed; only --split's verdict is the audit's.

    python results/leak_measure.py --audit [--split length] > exp/leak_audit.txt 2>&1
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


INDICES = range(8)


def normalized(Q, rewards, terminals, idx, split):
    """The reward iql_eval trains on at this index (before drop_terminals); index 0 is
    iql_eval's `if config.normalize_reward:` skip."""
    ds = {"rewards": np.array(rewards, dtype=np.float32, copy=True), "terminals": np.asarray(terminals)}
    if idx:
        Q.modify_reward(ds, "antmaze-audit", idx, split=split)
    return ds["rewards"]


def audit_rewards(Q, rewards, terminals, split):
    """{index: (max |real terminals - zero terminals|, relative affine residual sd)}."""
    zeros = np.zeros_like(np.asarray(terminals))
    x = np.asarray(rewards, dtype=np.float64)
    out = {}
    for i in INDICES:
        y = normalized(Q, rewards, terminals, i, split).astype(np.float64)
        d = float(np.abs(y - normalized(Q, rewards, zeros, i, split)).max())
        A = np.stack([x, np.ones_like(x)], axis=1)
        res = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        out[i] = (d, float(res.std() / y.std()) if y.std() > 0 else 0.0)
    return out


def audit(a):
    import d4rl  # noqa: F401
    import gym
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    rng = np.random.default_rng(0)
    verdict = True
    for v in a.variants.split():
        name = f"antmaze-{v.replace('_', '-')}-v2"
        ds = d4rl.qlearning_dataset(gym.make(name))
        term = np.asarray(ds["terminals"]).astype(bool)
        rewards = {"task": np.asarray(ds["rewards"], dtype=np.float32)}
        mot = {"rewards": Q.heuristic_rewards(ds, "path_length")}
        Q.gauge_reward(mot, "max0")
        rewards["motion"] = mot["rewards"]
        for k in range(3):
            g = {"rewards": rng.standard_normal(len(term)).astype(np.float32)}
            Q.gauge_reward(g, "max0")
            rewards[f"gauss {k}"] = g["rewards"]
        print(f"\n=== {name}: {len(term)} transitions, {term.sum()} terminals")
        for split in ("terminals", "length"):
            print(f"  split {split!r}" + ("  <- the audit" if split == a.split else ""))
            for lab, r in rewards.items():
                res = audit_rewards(Q, r, term, split)
                ok = all(d == 0.0 for d, _ in res.values())
                if split == a.split:
                    verdict &= ok
                print(f"    {lab:8s} {'PASS' if ok else 'FAIL'}  " +
                      "  ".join(f"{i}:{d:.2g}/{f:.1g}" for i, (d, f) in res.items()))
    print(f"\nLEAK AUDIT (split {a.split!r}): {'PASS' if verdict else 'FAIL'} "
          "(per index: max |real - zero terminals| / relative affine residual sd)")
    return 0 if verdict else 1


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
    # the audit, on iql_eval's own modify_reward (d4rl / gym stubbed if absent): 3,000 steps,
    # clustered terminals.  "length" passes at every index; "terminals" fails at 2-7.
    import types
    for m in ("d4rl", "gym"):
        try:
            __import__(m)
        except ImportError:
            sys.modules[m] = types.ModuleType(m)
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    rng = np.random.default_rng(1)
    r = (rng.standard_normal(3000) - 3.0).astype(np.float32)
    term = np.zeros(3000, dtype=bool)
    term[[400, 401, 402, 1700, 1701, 2550]] = True
    res = audit_rewards(Q, r, term, "length")
    assert all(d == 0.0 for d, _ in res.values()), res
    assert all(f < 1e-3 for _, f in res.values()), res                 # one global map (float32: 4/5 ~1.5e-4)
    res = audit_rewards(Q, r, term, "terminals")
    assert res[0][0] == 0.0 and res[1][0] == 0.0, res                    # 0, 1 never read terminals
    assert all(res[i][0] > 0 for i in range(2, 8)), res                  # 2-7 do under "terminals"
    assert res[6][1] > 1e-3 and res[7][1] > 1e-3, res                    # and 6, 7 are not one map
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="medium_play medium_diverse large_play large_diverse")
    ap.add_argument("--seeds", default="1 2 3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--audit", action="store_true", help="the registered leak audit (§14.83)")
    ap.add_argument("--split", choices=("terminals", "length"), default="length",
                    help="--audit: the split whose verdict counts")
    a = ap.parse_args()
    return selftest() if a.selftest else audit(a) if a.audit else run(a)


if __name__ == "__main__":
    sys.exit(main())
