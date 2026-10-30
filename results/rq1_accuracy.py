#!/usr/bin/env python
"""rq1_accuracy.py -- quantity A of RQ1's manipulation check (handoff §14.5, §14.69).  No IQL runs.

A is the held-out preference accuracy, PER PAIR, on each seed's clean, tie-free
validation split.  It is measured for every reward model RQ1's ladders use:
  mr        MR best model                (best_model.pt)
  pt        PT                           (best_model.pt)
  nmr       NMR                          (best_model.pt)
  ens_mean  the MR snapshot ensemble's mean reward: the MR run's checkpoints from
            epoch 100 on (the ensemble sweeps' mr_burn_in), averaged
It covers the anchor and every sparsity and primary-noise level, the four variants, and
seeds 1-10.  The validation split never changes with the level (§13.6, §11.3).

HOW
- **Best models.** Each family is scored with its own trainer's `evaluation`, as in
  training: MRTrainer for MR, PTTrainer for PT and for NMR (NMR trains with PTTrainer).
  The models are rebuilt with iql_eval's loaders. All pairs go in ONE batch, so the
  trainer's eval_acc is the per-pair accuracy. The batch mean over unshuffled batches of
  64 is recorded beside it: that is exactly what training logged as `eval_acc_reloaded`,
  and rq1_readout checks the two against each other.
- **The ensemble mean.** Each snapshot's mean-pooled segment logit (bt_pool, as
  MRTrainer pools) is averaged over snapshots. Pooling is linear, so this is the pooled
  logit of the mean reward. The per-draw centring the IQL labelling applies
  (centre_draws) adds the same constant to both segments of a pair, so it cancels.
- Ties of the two logits go to the first segment, as torch.argmax does in training.

    measure (on the box; needs d4rl and the models; resumable):
        cd ~/iqlpref && python results/rq1_accuracy.py measure
        -> exp/rq1_accuracy.json   (exp/ is gitignored; copy it over by hand)
    python results/rq1_accuracy.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPR = os.path.join(ROOT, "gp_reward-priors")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, GPR)
sys.path.insert(0, os.path.join(GPR, "scripts_bnn"))

OUT = os.path.join(ROOT, "exp", "rq1_accuracy.json")
FAMS = ("mr", "pt", "nmr", "ens_mean")
LEVELS = ("anchor", "r16", "r32", "r64", "r128", "n01", "n02", "n03", "n04", "n045")
VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
SEEDS = tuple(range(1, 11))
BURN_IN = 100                       # ensemble_sweeps/*_mean.yaml mr_burn_in
BATCH = 64                          # the trainers' validation loader (batch_size 64, shuffle=False)


def key(fam, level, variant, seed):
    return f"{fam}|{level}|{variant}|{seed}"


def model_dir(models_root, fam, level, variant, seed):
    import experiment_conditions as E
    base = E.model_base(variant, "mr" if fam == "ens_mean" else fam, None if level == "anchor" else level)
    return os.path.join(models_root, f"{base}_{seed}")


def val_file(variant, seed):
    import experiment_conditions as E
    env = E.env_of(variant)
    return os.path.join(GPR, "data", "antmaze", env, "eval", f"seed_{seed}", f"{env}_pref_val_{seed}.hdf5")


def batches(ds, size):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=size, shuffle=False)


def trainer_scores(trainer, ds):
    """(per-pair correct count, batch mean over batches of 64) with the trainer's own evaluation."""
    (one,) = list(batches(ds, len(ds)))
    acc = trainer.evaluation(one)["eval_acc"]
    correct = int(round(acc * len(ds)))
    assert abs(correct / len(ds) - acc) < 1e-6, (acc, len(ds))
    bm = float(np.mean([trainer.evaluation(b)["eval_acc"] for b in batches(ds, BATCH)]))
    return correct, bm


def pooled_logits_mlp(net, batch, bt_pool):
    """(B, 2) mean-pooled segment logits of an MLP reward, exactly as MRTrainer.evaluation forms them."""
    import torch
    from optbnn.utils.util import bt_pool_logit
    states, actions, _, mask, states_2, actions_2, _, mask_2, _ = batch
    B, T, _ = states.size()
    with torch.no_grad():
        x = torch.cat([torch.cat([states, actions], -1).reshape(B * T, -1),
                       torch.cat([states_2, actions_2], -1).reshape(B * T, -1)], 0)
        pred = net(x)
        p1 = pred[: B * T].reshape(B, T) * mask
        p2 = pred[B * T:].reshape(B, T) * mask_2
        return torch.stack([bt_pool_logit(p1, mask, bt_pool), bt_pool_logit(p2, mask_2, bt_pool)], 1).numpy()


def correct_from_logits(fx, labels):
    return int((np.argmax(fx, axis=1) == np.argmax(np.asarray(labels), axis=1)).sum())


def cmd_measure(a):
    import d4rl  # noqa: F401
    import gym
    import torch
    import yaml
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    from optbnn.training.training import MRTrainer, PTTrainer
    from optbnn.utils import util
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    levels = a.levels.split() if a.levels else LEVELS
    data = {}
    for v in VARIANTS:
        for s in SEEDS:
            data[(v, s)] = util.Pref_H5Dataset(val_file(v, s), -1)
    sdim, adim = data[(VARIANTS[0], 1)].states.shape[2], data[(VARIANTS[0], 1)].actions.shape[2]
    for fam in FAMS:
        for lv in levels:
            for v in VARIANTS:
                for s in SEEDS:
                    k = key(fam, lv, v, s)
                    if k in res and not a.force:
                        continue
                    d = model_dir(a.models_root, fam, lv, v, s)
                    if not os.path.isdir(d):
                        print(f"  {k}: no models at {d}")
                        continue
                    ds = data[(v, s)]
                    bt_pool = (yaml.safe_load(open(os.path.join(d, "config.yaml"))) or {}).get("bt_pool", "mean")
                    try:
                        if fam == "mr":
                            net = Q.load_mlp_reward_model(d, device="cpu")
                            c, bm = trainer_scores(MRTrainer(net, None, 1, "cpu", bt_pool), ds)
                            (one,) = list(batches(ds, len(ds)))           # the ensemble path's scoring, checked
                            assert correct_from_logits(pooled_logits_mlp(net, one, bt_pool), ds.labels) == c, k
                            rec = {"correct": c, "batch_mean": bm}
                        elif fam in ("pt", "nmr"):
                            net = (Q.load_pt_reward_model(d, device="cpu") if fam == "pt"
                                   else Q.load_nmr_reward_model(d, sdim, adim, device="cpu"))
                            c, bm = trainer_scores(PTTrainer(net, None, "cpu", bt_pool), ds)
                            rec = {"correct": c, "batch_mean": bm}
                        else:
                            (one,) = list(batches(ds, len(ds)))
                            acts = Q._mr_activations(d)
                            fx, n_snap, net = 0.0, 0, None
                            for p in Q._discover_mr_snapshots(d, BURN_IN):
                                state = Q._strip_compile_prefix(torch.load(p, map_location="cpu",
                                                                           weights_only=False)["net"])
                                if net is None:
                                    net = Q._build_mlp_reward_model(state, acts, "cpu")
                                net.load_state_dict(state)
                                net.eval()
                                fx = fx + pooled_logits_mlp(net, one, bt_pool)
                                n_snap += 1
                            rec = {"correct": correct_from_logits(fx / n_snap, ds.labels), "n_snapshots": n_snap}
                    except Exception as e:                       # noqa: BLE001  one model must not stop the run
                        print(f"  {k}: FAILED ({type(e).__name__}: {str(e)[:120]}) -- not recorded")
                        continue
                    rec.update(n=len(ds), dir=d)
                    res[k] = rec
                    print(f"  {k}: {rec['correct']}/{rec['n']}" +
                          (f"  batch mean {rec['batch_mean']:.6f}" if "batch_mean" in rec else
                           f"  ({rec['n_snapshots']} snapshots)"))
                    os.makedirs(os.path.dirname(a.out), exist_ok=True)
                    json.dump(res, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"\n  {len(res)} measurement(s) in {a.out}")
    return 0


def selftest():
    import torch
    from optbnn.bnn.nets.mlp import MLP
    from optbnn.training.training import MRTrainer
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    B, T, sd, ad = 77, 6, 3, 2

    class DS(torch.utils.data.Dataset):
        def __init__(self):
            f = lambda *s: torch.tensor(rng.normal(size=s), dtype=torch.float32)
            self.states, self.actions = f(B, T, sd), f(B, T, ad)
            self.states_2, self.actions_2 = f(B, T, sd), f(B, T, ad)
            self.attn_mask = torch.tensor(rng.random((B, T)) > 0.2, dtype=torch.float32)
            self.attn_mask_2 = torch.tensor(rng.random((B, T)) > 0.2, dtype=torch.float32)
            self.attn_mask[:, 0] = self.attn_mask_2[:, 0] = 1.0
            first = rng.random(B) > 0.5
            self.labels = torch.tensor(np.stack([first, ~first], 1), dtype=torch.float32)
            self.ts = torch.zeros(B, T, dtype=torch.int64)

        def __len__(self):
            return B

        def __getitem__(self, i):
            return (self.states[i], self.actions[i], self.ts[i], self.attn_mask[i], self.states_2[i],
                    self.actions_2[i], self.ts[i], self.attn_mask_2[i], self.labels[i])

    ds = DS()
    nets = [MLP(sd + ad, 1, [8, 8], "relu").eval() for _ in range(3)]
    (one,) = list(batches(ds, len(ds)))
    # the trainer's one-batch accuracy is per pair, and equals the MLP pooling path
    for net in nets:
        c, bm = trainer_scores(MRTrainer(net, None, 1, "cpu", "mean"), ds)
        assert c == correct_from_logits(pooled_logits_mlp(net, one, "mean"), ds.labels)
        # the batch mean is the unweighted mean over [0:64] and [64:77]
        fx = pooled_logits_mlp(net, one, "mean")
        ok = np.argmax(fx, 1) == np.argmax(ds.labels.numpy(), 1)
        assert abs(bm - (ok[:64].mean() + ok[64:].mean()) / 2) < 1e-6
    # the ensemble mean's pooled logit is the mean of the members' pooled logits
    fxs = [pooled_logits_mlp(n, one, "mean") for n in nets]

    class Mean(torch.nn.Module):
        def forward(self, x):
            return torch.stack([n(x) for n in nets]).mean(0)
    assert np.allclose(np.mean(fxs, 0), pooled_logits_mlp(Mean(), one, "mean"), atol=1e-5)
    # a per-snapshot constant (centring) does not change the decision under mean pooling
    shifted = [f + c for f, c in zip(fxs, (3.0, -7.0, 0.5))]
    assert correct_from_logits(np.mean(shifted, 0), ds.labels) == correct_from_logits(np.mean(fxs, 0), ds.labels)
    # ties go to the first segment, as torch.argmax
    assert correct_from_logits(np.array([[1.0, 1.0]]), np.array([[1.0, 0.0]])) == 1
    assert key("ens_mean", "r16", "large_play", 3) == "ens_mean|r16|large_play|3"
    assert model_dir("/m", "ens_mean", "r16", "large_play", 3) == "/m/antmaze_large_play_mr_r16_eval_3"
    assert model_dir("/m", "nmr", "anchor", "medium_play", 10) == "/m/antmaze_medium_play_nmr_eval_10"
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure")
    m.add_argument("--out", default=OUT)
    m.add_argument("--models-root", default=os.path.join(ROOT, "exp", "reward_learning"))
    m.add_argument("--levels", default=None, help='e.g. "anchor r16 r32 r64 r128" (default: all)')
    m.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "measure":
        return cmd_measure(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
