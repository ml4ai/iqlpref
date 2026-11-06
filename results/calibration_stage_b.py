#!/usr/bin/env python
"""calibration_stage_b.py -- why the BNN's H3 calibration gate fails, STAGE B (gp_reward-priors
HANDOFF §14.96).  Stages A-A3 (§14.93-§14.95) found: the BNN's mean logits are ~2x too small
(one global temperature fixes it); its posterior spread is mostly scale-proportional and
otherwise blind to novelty, in (x, y) and in its own input space.  Candidate mechanism: the
map-informed prior is a function of maze POSITION only.  Stage B separates the mechanisms.

SCOPE: the author's scoped exception to the BNN/CVaR hold (§14.92): the BNN reward models'
calibration diagnostics only -- no IQL result.  Outputs: per-model JSON in exp/calibration_b/
(gitignored; box-generated); the analysis text belongs in
gp_reward-priors/calibration_results/held_bnn_cvar/.  POST HOC, DESCRIPTIVE.

  (i)   TEMPERATURE, OTHER FAMILIES.  MR, PT and NMR anchor models (uncovered), seeds 1-10, on
        the SAME validation splits: each pair's mean-pooled logit difference, from each
        trainer's own pooling (checked: its accuracy equals the trainer's eval_acc).  Is the
        2x under-scaling the BNN's alone?
  (ii)  CHAINS.  Every BNN draw's logit difference, kept per chain: each chain's mean and
        variance per pair.  Between-chain vs within-chain variance; how much of the
        between-chain part is one SCALE factor per chain; is each chain sharper than the
        mixture (one temperature per chain index)?
  (iii) THE PRIOR, EXACTLY.  The map prior k = amp2 (sig_c2 + sig_g2 Kgeo[c(s), c(s')] +
        sig_n2 1[i=j]) gives the mean-pooled logit difference of a pair the variance
            amp2 [ sig_g2 (w1 - w2)' Kgeo (w1 - w2) + sig_n2 (1/n1 + 1/n2) ]
        (the constant cancels; w = each segment's cell-occupancy fractions, n its valid
        steps).  Built with the LIVE layout (diagnose_sampling_tail._live_prior), as the run
        built it.  Does the prior's spread track novelty?  Does the posterior's follow it?
  (iv)  POSITIONAL FAMILIARITY.  Per segment, the occupancy-weighted count of that seed's
        training steps in its cells; a pair's is the smaller of its two.

    measure (ON THE BOX; needs d4rl, the BNN chains and the models; resumable):
        cd ~/iqlpref && python results/calibration_stage_b.py measure [--device cuda]
        -> exp/calibration_b/{bnn,mr,pt,nmr}_<variant>_<seed>.json   (copy the folder over)
    analyse (here):
        /opt/anaconda3/envs/irl/bin/python results/calibration_stage_b.py analyse \\
            > gp_reward-priors/calibration_results/held_bnn_cvar/mechanism_stage_b.txt
    /opt/anaconda3/envs/irl/bin/python results/calibration_stage_b.py --selftest
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPR = os.path.join(ROOT, "gp_reward-priors")
for p in (HERE, ROOT, GPR, os.path.join(GPR, "scripts_bnn")):
    sys.path.insert(0, p)

OUT = os.path.join(ROOT, "exp", "calibration_b")
VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
SEEDS = tuple(range(1, 11))
BASE_FAMS = ("mr", "pt", "nmr")


# --------------------------------------------------------------------------
# Pure pieces (self-tested)
# --------------------------------------------------------------------------
def occupancy(cells, mask, n_cells):
    """[B, T] cell indices and valid-step mask -> ([B, n_cells] fractions, [B] valid counts)."""
    B = cells.shape[0]
    w = np.zeros((B, n_cells))
    n = mask.sum(1).astype(float)
    for b in range(B):
        c = cells[b][mask[b]]
        if len(c):
            w[b] = np.bincount(c, minlength=n_cells) / len(c)
    return w, n


def prior_logit_sd(w1, n1, w2, n2, Kgeo, amp2, sig_g2, sig_n2):
    dw = w1 - w2
    var = amp2 * (sig_g2 * np.einsum("bi,ij,bj->b", dw, Kgeo, dw) + sig_n2 * (1.0 / n1 + 1.0 / n2))
    return np.sqrt(np.maximum(var, 0.0))


def chain_decomposition(cm, cv, cn):
    """cm, cv [C, B] per-chain mean and variance of the logit difference; cn [C] draws.
    -> dict: pooled mean, total variance, between share, scale share, chain sharpness."""
    cm, cv, cn = np.asarray(cm, float), np.asarray(cv, float), np.asarray(cn, float)
    wt = cn / cn.sum()
    dbar = (wt[:, None] * cm).sum(0)
    within = (wt[:, None] * cv).sum(0)
    between = (wt[:, None] * (cm - dbar) ** 2).sum(0)
    total = within + between
    a = (cm @ dbar) / max(float(dbar @ dbar), 1e-12)               # one scale factor per chain
    resid = cm - a[:, None] * dbar
    dev = cm - dbar
    scale_share = 1.0 - float((wt * (resid ** 2).sum(1)).sum() / max(float((wt * (dev ** 2).sum(1)).sum()), 1e-12))
    sharp = float(np.median(np.abs(cm).mean(0) / np.maximum(np.abs(dbar), 1e-12)))
    return {"dbar": dbar, "total": total, "between_share": float(between.mean() / max(total.mean(), 1e-12)),
            "scale_share": scale_share, "chain_scale_sd": float(np.std(a)), "sharpness": sharp}


# --------------------------------------------------------------------------
# measure (box)
# --------------------------------------------------------------------------
def pt_logits(net, batch, bt_pool):
    """(B, 2) segment logits exactly as PTTrainer.evaluation forms them (PT and NMR)."""
    import torch
    from optbnn.utils.util import bt_pool_logit
    states, actions, timesteps, am, states_2, actions_2, timesteps_2, am2, _ = batch
    B, T, _ = states.size()
    with torch.no_grad():
        out, _ = net(torch.cat([states, states_2], 0), torch.cat([actions, actions_2], 0),
                     torch.cat([timesteps, timesteps_2], 0).to(torch.int), torch.cat([am, am2], 0))
        pred = out["weighted_sum"]
        f1 = bt_pool_logit(pred[:B].reshape(B, T) * am, am, bt_pool)
        f2 = bt_pool_logit(pred[B:].reshape(B, T) * am2, am2, bt_pool)
    return torch.stack([f1.reshape(-1), f2.reshape(-1)], 1).cpu().numpy()


def measure_base(fam, v, s, device, models_root):
    import torch
    import yaml
    sys.path.insert(0, os.path.join(ROOT, "algorithms", "offline"))
    import iql_eval as Q
    import rq1_accuracy as RA
    from optbnn.training.training import MRTrainer, PTTrainer
    from optbnn.utils import util
    d = RA.model_dir(models_root, fam, "anchor", v, s)
    ds = util.Pref_H5Dataset(RA.val_file(v, s), -1)
    (one,) = list(RA.batches(ds, len(ds)))
    bt_pool = (yaml.safe_load(open(os.path.join(d, "config.yaml"))) or {}).get("bt_pool", "mean")
    one = [x.to(device) for x in one]
    if fam == "mr":
        net = Q.load_mlp_reward_model(d, device=device)
        acc = MRTrainer(net, None, 1, device, bt_pool).evaluation(one)["eval_acc"]   # before .cpu() moves it
        fx = RA.pooled_logits_mlp(net.cpu(), [x.cpu() for x in one], bt_pool)
    else:
        sdim, adim = ds.states.shape[2], ds.actions.shape[2]
        net = (Q.load_pt_reward_model(d, device=device) if fam == "pt"
               else Q.load_nmr_reward_model(d, sdim, adim, device=device))
        fx = pt_logits(net, one, bt_pool)
        acc = PTTrainer(net, None, device, bt_pool).evaluation(one)["eval_acc"]
    y1 = np.asarray(ds.labels)[:, 0]
    mine = float(((fx[:, 0] > fx[:, 1]) == (y1 > 0.5)).mean())
    assert abs(mine - acc) < 1e-6, (fam, v, s, mine, acc)      # the trainer's own pooling, reproduced
    return {"y1": y1.tolist(), "d": (fx[:, 0] - fx[:, 1]).tolist(), "acc": acc, "dir": d, "device": device}


def measure_bnn(v, s, device, models_root):
    import torch
    import diagnose_sampling_tail as D
    from optbnn.bnn.nets.mlp import MLP
    from optbnn.utils import util
    from optbnn.utils.util import bt_pool_logit_np
    run_dir = os.path.join(models_root, f"antmaze_{v}_bnn_eval_{s}")
    cfg = D._load_run_config(run_dir)
    (val, _), (train, _) = D._resolve_dataset(cfg, "val"), D._resolve_dataset(cfg, "train")
    assert "/noise/" not in val and "/reduction/" not in val, val
    X, y = util.load_pref_data(val, training_ratio=1.0)
    B, _, T, dd = X.shape
    obs = dd - 1
    am1, am2 = X[:, 0, :, obs] > 0, X[:, 1, :, obs] > 0
    m1f, m2f = am1.astype(np.float64), am2.astype(np.float64)        # as calibration_gate pools
    pool = cfg.get("bt_pool", "mean")
    x1 = torch.from_numpy(X[:, 0, :, :obs].reshape(-1, obs).astype(np.float32)).to(device)
    x2 = torch.from_numpy(X[:, 1, :, :obs].reshape(-1, obs).astype(np.float32)).to(device)
    net = MLP(input_dim=obs, output_dim=1, hidden_dims=[cfg["width"]] * cfg["depth"], activation_fn="relu").to(device)
    net.eval()
    cm, cv, cn = [], [], []
    for cid in D._resolve_chain_ids(run_dir, None, None, cfg["num_chains"]):
        ds_ = []
        for w in D._load_chain_weights(run_dir, cid, device):
            with torch.no_grad():
                for p, a in zip(net.parameters(), w):
                    p.copy_(torch.from_numpy(a).to(device))
                o1 = net(x1).cpu().numpy().reshape(B, T).astype(np.float64)
                o2 = net(x2).cpu().numpy().reshape(B, T).astype(np.float64)
            ds_.append(bt_pool_logit_np(o1 * m1f, m1f, pool) - bt_pool_logit_np(o2 * m2f, m2f, pool))
        ds_ = np.stack(ds_)
        cm.append(ds_.mean(0).tolist())
        cv.append(ds_.var(0).tolist())
        cn.append(int(ds_.shape[0]))
    prior, _, live = D._live_prior(cfg, quiet=True)
    if not live:
        raise RuntimeError(f"{run_dir}: the live layout did not load -- the prior basis would not be the run's")
    K = prior.Kgeo.cpu().numpy()
    nc = K.shape[0]
    c1 = prior.cell_of(X[:, 0, :, :obs].reshape(-1, obs)).reshape(B, T)
    c2 = prior.cell_of(X[:, 1, :, :obs].reshape(-1, obs)).reshape(B, T)
    w1, n1 = occupancy(c1, am1, nc)
    w2, n2 = occupancy(c2, am2, nc)
    psd = prior_logit_sd(w1, n1, w2, n2, K, prior.amp2, prior.sig_g2, prior.sig_n2)
    Xt, _ = util.load_pref_data(train, training_ratio=1.0)
    tm = np.concatenate([Xt[:, 0, :, obs], Xt[:, 1, :, obs]]) > 0
    tc = np.concatenate([prior.cell_of(Xt[:, k, :, :obs].reshape(-1, obs)).reshape(-1, T) for k in (0, 1)])
    counts = np.bincount(tc[tm], minlength=nc).astype(float)
    fam1, fam2 = w1 @ counts, w2 @ counts
    return {"y1": np.asarray(y)[:, 0].tolist(), "chain_mean": cm, "chain_var": cv, "chain_n": cn,
            "prior_sd": psd.tolist(), "pos_fam": np.minimum(fam1, fam2).tolist(), "run_dir": run_dir,
            "map": {k: float(getattr(prior, k)) for k in ("amp2", "sig_c2", "sig_g2", "sig_n2", "eta")},
            "n_cells": int(nc)}


def cmd_measure(a):
    import torch
    torch.set_num_threads(a.threads)          # small forwards; leave the cores to the IQL agents
    os.makedirs(a.out, exist_ok=True)
    for v in VARIANTS:
        for s in SEEDS:
            for fam in ("bnn",) + BASE_FAMS:
                path = os.path.join(a.out, f"{fam}_{v}_{s}.json")
                if os.path.exists(path) and not a.force:
                    continue
                try:
                    rec = measure_bnn(v, s, a.device, a.models_root) if fam == "bnn" else \
                        measure_base(fam, v, s, a.device, a.models_root)
                except Exception as e:                       # noqa: BLE001  one model must not stop the run
                    print(f"  {fam} {v} {s}: FAILED ({type(e).__name__}: {str(e)[:160]})")
                    continue
                tmp = path + ".tmp"
                json.dump(rec, open(tmp, "w"))
                os.replace(tmp, path)
                print(f"  {fam} {v} {s}: {len(rec['y1'])} pairs" +
                      (f", {len(rec['chain_n'])} chains" if fam == "bnn" else f", acc {rec['acc']:.3f}"))
    return 0


# --------------------------------------------------------------------------
# analyse (here)
# --------------------------------------------------------------------------
def cmd_analyse(a, out=print):
    import calibration_gate as G
    import calibration_mechanism as CM
    from scipy.stats import spearmanr

    def load(fam, v):
        recs = {}
        for s in SEEDS:
            p = os.path.join(a.inp, f"{fam}_{v}_{s}.json")
            if os.path.exists(p):
                recs[s] = json.load(open(p))
        return recs

    out("BNN H3 CALIBRATION, MECHANISM STAGE B (handoff §14.96; scoped hold exception; post hoc, descriptive)")
    for v in VARIANTS:
        out(f"\n=== {v}")
        # (i) one temperature per family, leave-one-seed-out, on the plug-in logits
        bnn = load("bnn", v)
        for fam in ("bnn",) + BASE_FAMS:
            recs = bnn if fam == "bnn" else load(fam, v)
            if len(recs) != len(SEEDS):
                out(f"  (i)  {fam:4s} {len(recs)} of {len(SEEDS)} models -- not read")
                continue
            rr = {}
            for s, r in recs.items():
                d = (chain_decomposition(r["chain_mean"], r["chain_var"], r["chain_n"])["dbar"] if fam == "bnn"
                     else np.asarray(r["d"], float))
                rr[s] = {"d_mean": d, "y1": r["y1"], "p_pred": CM.sig(d)}
            t, cb, ca, eb, ea = CM.loso_temperature(rr, "plug")
            acc = np.mean(np.concatenate([(np.asarray(x["d_mean"]) > 0) == (np.asarray(x["y1"]) > 0.5) for x in rr.values()]))
            out(f"  (i)  {fam:4s} accuracy {acc:.3f}  T {t:.3f}  held-out ECE {eb:.3f} -> {ea:.3f}  CE {cb:.3f} -> {ca:.3f}")
        if len(bnn) != len(SEEDS):
            continue
        # (ii) chains; (iii) the prior; (iv) positional familiarity -- per seed, then averaged
        rows, tch, gap = [], {}, 0.0
        for s, r in bnn.items():
            dec = chain_decomposition(r["chain_mean"], r["chain_var"], r["chain_n"])
            g = json.load(open(os.path.join(CM.MEAS, f"antmaze_{v}_bnn_eval_{s}.json")))
            gap = max(gap, float(np.abs(dec["dbar"] - np.asarray(g["d_mean"])).max()))   # same model, same pairs
            sp = np.sqrt(dec["total"])
            d = dec["dbar"]
            y = np.asarray(r["y1"]) > 0.5
            wrong = (d > 0) != y
            _, b, _, _ = CM.spread_split(d, sp)
            excess = sp ** 2 - b * d ** 2
            psd, fam_ = np.asarray(r["prior_sd"]), np.asarray(r["pos_fam"])
            nov = CM.pair_novelty_input(v, s)
            ok = 0 < wrong.sum() < wrong.size
            rows.append([dec["between_share"], dec["scale_share"], dec["sharpness"],
                         spearmanr(psd, sp)[0], float(np.median(sp / np.maximum(psd, 1e-12))),
                         spearmanr(psd, nov["z_body"])[0], spearmanr(psd, nov["raw"])[0],
                         G.auroc(psd, wrong) if ok else np.nan,
                         G.auroc(-fam_, wrong) if ok else np.nan, spearmanr(-fam_, sp)[0], spearmanr(-fam_, excess)[0]])
            for c, m in enumerate(r["chain_mean"]):                # one temperature per chain index
                tch.setdefault(c, {})[s] = {"d_mean": np.asarray(m), "y1": r["y1"], "p_pred": CM.sig(np.asarray(m))}
        R = np.nanmean(np.asarray(rows, float), 0)
        out(f"  check: the chains' pooled mean logit vs the gate's measurement, max |difference| {gap:.2e}"
            + ("" if gap < 1e-3 else "  !! NOT THE SAME MODELS OR PAIRS -- nothing below is read"))
        if gap >= 1e-3:
            continue
        tc = [CM.loso_temperature(tch[c], "plug")[0] for c in sorted(tch)[: a.chains] if len(tch[c]) == len(SEEDS)]
        out(f"  (ii) variance between chains {R[0]:.2f} of the total; of the between-chain part, one scale factor "
            f"per chain explains {R[1]:.2f}; median mean_c|d_c| / |d_bar| {R[2]:.2f}; per-chain T (first "
            f"{len(tc)} chain indices) median {np.median(tc):.3f} [{np.min(tc):.3f}, {np.max(tc):.3f}]")
        out(f"  (iii) prior sd vs posterior sd: Spearman {R[3]:+.3f}, median ratio posterior/prior {R[4]:.3f}; "
            f"prior sd vs novelty: z_body {R[5]:+.3f}, raw {R[6]:+.3f}; error-AUROC of the prior sd {R[7]:.3f}")
        out(f"  (iv) positional novelty (fewer training steps in the pair's cells): error-AUROC {R[8]:.3f}; "
            f"Spearman with the posterior sd {R[9]:+.3f}, with the excess spread {R[10]:+.3f}")
    return 0


def selftest():
    import torch
    sys.path.insert(0, GPR)
    from optbnn.gp.maze_layouts import get_antmaze_layout
    from optbnn.gp.models.map_informed_prior import MapInformedGPPrior
    # (iii) the exact prior variance of a mean-pooled logit difference, against Monte Carlo
    fm, sc, off = get_antmaze_layout("medium")
    pr = MapInformedGPPrior(fm, sc, off, eta=1.0, sig_c2=2.0, sig_g2=1.5, sig_n2=0.3, amp2=4.0)
    rng = np.random.default_rng(0)
    B, T = 6, 7
    xy = rng.uniform(0, 20, size=(2, B, T, 2))
    mask = rng.random((2, B, T)) < 0.8
    mask[:, :, 0] = True
    cells = [pr.cell_of(xy[k].reshape(-1, 2)).reshape(B, T) for k in (0, 1)]
    w1, n1 = occupancy(cells[0], mask[0], pr.n_cells)
    w2, n2 = occupancy(cells[1], mask[1], pr.n_cells)
    sd = prior_logit_sd(w1, n1, w2, n2, pr.Kgeo.numpy(), pr.amp2, pr.sig_g2, pr.sig_n2)
    idx = np.concatenate([cells[0].ravel(), cells[1].ravel()])
    K = pr._gram_from_idx(idx).numpy()
    L = np.linalg.cholesky(K + 1e-10 * np.eye(len(idx)))
    f = (L @ rng.standard_normal((len(idx), 40000))).T.reshape(40000, 2, B, T)
    d = (f[:, 0] * mask[0]).sum(-1) / n1 - (f[:, 1] * mask[1]).sum(-1) / n2
    assert np.allclose(d.std(0), sd, rtol=0.03), (d.std(0), sd)
    # (ii) chains that differ only by a scale factor: all between-chain variance is scale
    base = rng.normal(0, 2, 50)
    cm = np.stack([a * base for a in (0.5, 1.0, 1.5, 2.0)])
    dec = chain_decomposition(cm, np.full_like(cm, 0.01), [60] * 4)
    assert dec["scale_share"] > 0.999 and dec["between_share"] > 0.9 and abs(dec["sharpness"] - 1.0) < 1e-9
    # chains that disagree in sign: the mixture is softer than every chain
    cm = np.stack([base, -0.2 * base, base, base])
    dec = chain_decomposition(cm, np.full_like(cm, 0.01), [60] * 4)
    assert abs(dec["sharpness"] - 0.8 / 0.7) < 1e-9, dec["sharpness"]     # mean |d_c| 0.8|b|, |d_bar| 0.7|b|
    # occupancy: fractions over valid steps only
    w, n = occupancy(np.array([[0, 0, 1, 2]]), np.array([[True, True, True, False]]), 3)
    assert np.allclose(w, [[2 / 3, 1 / 3, 0]]) and n[0] == 3
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    m = sub.add_parser("measure")
    m.add_argument("--out", default=OUT)
    m.add_argument("--models-root", default=os.path.join(ROOT, "exp", "reward_learning"))
    m.add_argument("--device", default="cpu")
    m.add_argument("--threads", type=int, default=4, help="torch CPU threads (default 4)")
    m.add_argument("--force", action="store_true")
    n = sub.add_parser("analyse")
    n.add_argument("--inp", default=OUT)
    n.add_argument("--chains", type=int, default=16, help="chain indices given their own temperature")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "measure":
        return cmd_measure(a)
    if a.cmd == "analyse":
        return cmd_analyse(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
