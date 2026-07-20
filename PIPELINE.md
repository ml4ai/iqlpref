# Reward-model → IQL evaluation pipeline (antmaze)

End-to-end, seed-consistent pipeline for training preference reward models
(`gp_reward-priors`) and evaluating them with offline IQL
(`algorithms/offline/iql_eval.py`). Everything is keyed on a single **global seed
1–10**: the seed selects the data split, fixes the reward-model output directory,
and becomes the IQL training seed — so an IQL run is always evaluated against the
reward model trained on the *same* seed.

- **Methods:** `bnn` (fSGHMC preference BNN), `mr` (MLP reward), `pt` (preference transformer)
- **Variants:** `medium_play`, `medium_diverse`, `large_play`, `large_diverse`
- **Seeds:** `1 … 10`
- **Hardware assumed:** 6× RTX A6000, 255 CPU cores

> **Prerequisite:** activate your conda/virtual env first — every launcher uses the
> `python` and `wandb` on `PATH`. (Override the interpreter for Phase 1 only with
> `PY=/path/to/python ./train_rewards.sh …`.)

---

## Seed → path contract (the thing that makes this work)

The eval training scripts write each run to a **deterministic, per-seed** directory
(no uuid in the path — the uuid stays in the wandb run name only):

```
gp_reward-priors/exp/reward_learning/antmaze_<variant>_<method>_eval_<seed>/
    ├── config.yaml
    ├── sampling_f/chain_*/…          # bnn
    ├── best_model.pt                 # mr, pt
    └── checkpoint_<epoch>.pt         # mr  (snapshot ensemble members)
```

`iql_eval.py` takes `reward_model_root` = that path **without** `_<seed>` and appends
`_<seed>` itself, using `seed` for both the reward model and IQL's own RNG. The sweep
files already point `reward_model_root` at the right base per (variant, method).

> This replaces the old fixed `reward_model_path`. `iql.py` is unchanged; the seed
> logic lives only in the new `iql_eval.py`.

### One reward model → several IQL runs

| Reward model (per variant, seed) | IQL runs it feeds | Sweep dir | Key flags |
|---|---|---|---|
| **BNN** | 2 | `bnn_sweeps` | `bnn_reward_model=true`, `bnn_alpha` = `0.0` (mean) / `0.95` (CVaR) |
| **MR**  | 3 | `mr_sweeps` (best) + `ensemble_sweeps` (mean+CVaR) | best: `best_model.pt`; ensemble: `mr_ensemble=true`, `mr_alpha` = `0.0` / `0.95` over `checkpoint_*.pt` |
| **PT**  | 1 | `pt_sweeps` | `query_length=100` |

`mr_sweeps` and `ensemble_sweeps` both read the **same** `..._mr_eval_<seed>` dir
(best model vs. per-epoch snapshots).

**Run accounting:** Phase 1 = 3×4×10 = **120** reward-model trainings.
Phase 2 = 80 (BNN) + 40 (MR-best) + 80 (MR-ensemble) + 40 (PT) = **240** IQL runs.

---

## Phase 1 — train the reward models

Launcher: `gp_reward-priors/train_rewards.sh METHOD [GPU_LIST] [PACK]`
(run from the submodule root). It sweeps all 4 variants × 10 seeds for one method,
GPU-packed, one deterministic output dir per (variant, seed).

```bash
cd gp_reward-priors

# MR: small nets, pack 3 jobs/GPU  -> 18 concurrent   (~fast)
./train_rewards.sh mr

# PT: transformers, pack 2 jobs/GPU -> 12 concurrent
./train_rewards.sh pt

# BNN: fSGHMC, heaviest. Default 1 GPU/job (8 chains co-located) -> 6 concurrent.
./train_rewards.sh bnn
#   Alternative: spread each run's chains over 3 GPUs -> 2 concurrent, all 6 busy,
#   faster per-run wall-clock if single-GPU chain contention dominates:
./train_rewards.sh bnn "0 1 2 3 4 5" 3
```

Run the three methods **sequentially** — each launcher already saturates all 6 GPUs.
Phase-1 training does *not* spawn the 25-worker eval pool (that's Phase 2), so CPU is
not the limiter here.

Useful env overrides: `SEEDS="1 2 3"`, `VARIANTS="medium_play large_play"`.
Logs: `gp_reward-priors/exp/train_logs/<method>/<variant>_seed<seed>.log`.

Sanity-check completion before Phase 2:

```bash
# expect 40 dirs per method (4 variants × 10 seeds)
ls -d gp_reward-priors/exp/reward_learning/antmaze_*_mr_eval_*  | wc -l
ls -d gp_reward-priors/exp/reward_learning/antmaze_*_bnn_eval_* | wc -l
ls -d gp_reward-priors/exp/reward_learning/antmaze_*_pt_eval_*  | wc -l
```

---

## Phase 2 — IQL evaluation (W&B grid sweeps)

Each sweep group has a `launch.sh SWEEP [GPU_LIST] [AGENTS_PER_GPU]` (run from repo
root). `all` = every antmaze sweep in that group; each sweep grids `seed: 1…10` and
`iql_eval.py` resolves the per-seed reward model automatically.

```bash
# from repo root
./mr_sweeps/launch.sh        all              # 4 sweeps  (MR best model)
./ensemble_sweeps/launch.sh  all              # 8 sweeps  (MR snapshot ensemble: mean+CVaR)
./bnn_sweeps/launch.sh       all              # 8 sweeps  (BNN: mean+CVaR)
./pt_sweeps/launch.sh        all              # 4 sweeps  (PT)
```

**Concurrency / CPU cap.** Every IQL run's evaluation uses `n_envs=25` CPU workers,
so `concurrency × 25 ≤ 255` cores. Defaults are `6 GPUs × 1 agent = 6` concurrent
(150 cores) — safe. The launchers warn if you oversubscribe; `AGENTS_PER_GPU=2`
(12 concurrent, 300 cores) exceeds 255 and the evals will contend when they sync.
Stay at **6–8 concurrent** for best throughput.

You can run the four groups back-to-back, or overlap two groups at 3 GPUs each if you
want to fill the box while keeping the core budget under 255.

---

## Files in this pipeline

| File | Role |
|---|---|
| `gp_reward-priors/scripts_{bnn,mr,pt}/run_*_training_antmaze_eval.py` | Phase-1 training; write deterministic `_<seed>` dirs |
| `gp_reward-priors/scripts_{bnn,mr,pt}/antmaze_<variant>_*_antmaze_eval.yaml` | per-variant training configs |
| `gp_reward-priors/train_rewards.sh` | Phase-1 launcher (GPU-packed, all variants × seeds) |
| `algorithms/offline/iql_eval.py` | Phase-2 IQL, seed-derived `reward_model_root` (iql.py untouched) |
| `{bnn,ensemble,mr,pt}_sweeps/sweep_antmaze_*.yaml` | Phase-2 W&B sweeps (grid over seed) |
| `{bnn,ensemble,mr,pt}_sweeps/launch.sh` | Phase-2 launchers (W&B agents across GPUs) |
