# Reward-model → IQL pipeline (antmaze)

End-to-end pipeline for training preference reward models (`gp_reward-priors`) and
running them through offline IQL (`algorithms/offline/iql_eval.py`). Everything is
keyed on a **global seed** that selects the data split, fixes the reward-model output
directory, *and* becomes the IQL seed — so a run is always paired with the reward
model trained on the same seed.

- **Methods:** `bnn` (fSGHMC preference BNN), `mr` (MLP reward), `pt` (preference transformer)
- **Variants:** `medium_play`, `medium_diverse`, `large_play`, `large_diverse`
- **Seeds:** `0` (selection) + `1 … 10` (evaluation)
- **Hardware assumed:** 6× RTX A6000, 255 CPU cores

> **Prerequisite:** activate your conda/virtual env first — every launcher uses the
> `python` and `wandb` on `PATH`. (Override the interpreter for Phase 1 only with
> `PY=/path/to/python ./train_rewards.sh …`.)

---

## Seed discipline — the load-bearing invariant

**Selection uses seed 0. Evaluation uses seeds 1–10. They never mix.**

`seed` does not only control run-time randomness; it **also selects the data files**
(`{data_root}/{variant}/eval/seed_{seed}/…`). So seed 0 is a genuinely disjoint
partition — a run at `seed: 0` trains on the seed-0 split and is scored on the seed-0
validation split, which no evaluation run ever sees. Every reported number at seeds
1–10 is therefore out-of-sample with respect to the *entire* selection procedure,
including the stage-4 normalization search below.

> If you add a tuning stage, give it a selection seed outside 1–10. The argument
> collapses if any tuning touches an evaluation seed.

Full procedure, all four stages, and the results table:
[`gp_reward-priors/HANDOFF_HP_SELECTION.md`](gp_reward-priors/HANDOFF_HP_SELECTION.md).
Stages 1–3 (architecture, sampler schedule, draw budget) are upstream of this
document and already folded into the production eval configs. **Stage 4 — output
normalization — is the part that runs through this pipeline**, and it is what the
sweep files are currently configured for.

---

## Seed → path contract

The eval training scripts write each run to a **deterministic, per-seed** directory
(no uuid in the path — the uuid stays in the wandb run name only):

```
exp/reward_learning/antmaze_<variant>_<method>_eval_<seed>/
    ├── config.yaml
    ├── sampling_f/chain_*/…          # bnn
    ├── best_model.pt                 # mr, pt
    └── checkpoint_<epoch>.pt         # mr  (snapshot ensemble members)
```

`iql_eval.py` takes `reward_model_root` = that path **without** `_<seed>` and appends
`_<seed>` itself, using `seed` for both the reward model and IQL's own RNG.

The sweeps point `reward_model_root` at `~/iqlpref/exp/reward_learning/…`. The training
configs set `OUT_DIR` / `checkpoints_path` to a *relative* `./exp/reward_learning/…`,
so confirm once that the two agree before launching Phase 2:

```bash
ls -d ~/iqlpref/exp/reward_learning/antmaze_*_eval_0 | head
```

If the dirs are instead under `gp_reward-priors/exp/`, either point the sweeps there or
symlink — the only requirement is that `reward_model_root` resolves to real dirs.

> This replaces the old fixed `reward_model_path`. `iql.py` is unchanged; the seed
> logic lives only in `iql_eval.py`.

### One reward model → several IQL runs

| Reward model (per variant, seed) | IQL runs it feeds | Sweep dir | Key flags |
|---|---|---|---|
| **BNN** | 2 | `bnn_sweeps` | `bnn_reward_model=true`, `bnn_alpha` = `0.0` (mean) / `0.95` (CVaR) |
| **MR**  | 3 | `mr_sweeps` (best) + `ensemble_sweeps` (mean+CVaR) | best: `best_model.pt`; ensemble: `mr_ensemble=true`, `mr_alpha` = `0.0` / `0.95` over `checkpoint_*.pt` |
| **PT**  | 1 | `pt_sweeps` | `query_length=100` |

`mr_sweeps` and `ensemble_sweeps` both read the **same** `..._mr_eval_<seed>` dir
(best model vs. per-epoch snapshots). That's **24 sweep files** = 8 BNN + 8 ensemble +
4 MR + 4 PT.

**Run accounting**

| phase | runs |
|---|---|
| Phase 1 — reward-model training | 3 methods × 4 variants × 11 seeds = **132** |
| Phase 2a — stage-4 selection (seed 0) | 24 sweeps × 8 normalization indices = **192** |
| Phase 2b — evaluation (seeds 1–10) | 24 sweeps × 10 seeds = **240** |

---

## Phase 1 — train the reward models

Launcher: `gp_reward-priors/train_rewards.sh METHOD [GPU_LIST] [PACK]`. It sweeps all
4 variants × 11 seeds (**0–10**) for one method, GPU-packed, one deterministic output
dir per (variant, seed). It resolves its own root, so it runs from anywhere.

```bash
cd gp_reward-priors

# MR: small nets, pack 3 jobs/GPU  -> 18 concurrent
./train_rewards.sh mr

# PT: transformers, pack 2 jobs/GPU -> 12 concurrent
./train_rewards.sh pt

# BNN: fSGHMC, heaviest. Default 1 GPU/job (8 chains co-located) -> 6 concurrent.
./train_rewards.sh bnn
#   Alternative: spread each run's chains over 3 GPUs -> 2 concurrent, all 6 busy:
./train_rewards.sh bnn "0 1 2 3 4 5" 3
```

Run the three methods **sequentially** — each launcher already saturates all 6 GPUs.
Phase-1 does *not* spawn the 25-worker eval pool (that's Phase 2), so CPU is not the
limiter here.

The pool holds `PACK × #GPUs` jobs in flight and refills as each finishes, printing
`[n/44 done]` as slots free. Useful overrides: `SEEDS="0"` (selection lineage only),
`VARIANTS="medium_play large_play"`.
Logs: `gp_reward-priors/exp/train_logs/<method>/<variant>_seed<seed>.log`.

Sanity-check before Phase 2 (**44** dirs per method = 4 variants × 11 seeds):

```bash
ls -d exp/reward_learning/antmaze_*_mr_eval_*  | wc -l
ls -d exp/reward_learning/antmaze_*_bnn_eval_* | wc -l
ls -d exp/reward_learning/antmaze_*_pt_eval_*  | wc -l
```

---

## Phase 2a — stage 4: output-normalization selection (seed 0)

**This is the sweeps' current state.** Each of the 24 sweeps pins `seed: 0` and grids
`normalize_reward` over the **8 normalization functions, indices 0–7**, defined in
`modify_reward()` in `iql.py`. Index 0 is the identity (the call site is guarded by
`if config.normalize_reward:`, so 0 is falsy and nothing is applied).

```bash
# from repo root
./mr_sweeps/launch.sh        all              # 4 sweeps  (MR best model)
./ensemble_sweeps/launch.sh  all              # 8 sweeps  (MR snapshot ensemble: mean+CVaR)
./bnn_sweeps/launch.sh       all              # 8 sweeps  (BNN: mean+CVaR)
./pt_sweeps/launch.sh        all              # 4 sweeps  (PT)
```

**Selection statistic.** One IQL run is 1,000,000 steps with an evaluation every 5,000
steps = **200 evaluation points**, each the mean score over **100 episodes**. The
winning index is the one maximising the **max over those 200 points**. These operative
values come from the IQL run config (`configs/offline/iql/antmaze/<variant>.yaml`), not
from the `iql.py` dataclass defaults.

The index must be selected **per (family × variant)**: indices 2–7 derive their
constants from `min_ret`/`max_ret` of each reward model's own labels, so the same index
means a different transformation for each model.

## Phase 2b — evaluation (seeds 1–10)

Once each (family × variant) has its winning index, flip the sweeps to the evaluation
lineage: set `normalize_reward` to the single winning `value:` and restore
`seed: values: [1,2,3,4,5,6,7,8,9,10]`. Nothing else changes — `iql_eval.py` resolves
each seed's reward model automatically. Reporting uses the identical statistic as
selection; only the seed lineage differs.

**Concurrency / CPU cap.** Every IQL run's evaluation uses `n_envs=25` CPU workers, so
`concurrency × 25 ≤ 255` cores. Defaults are `6 GPUs × 1 agent = 6` concurrent (150
cores) — safe. The launchers warn if you oversubscribe; `AGENTS_PER_GPU=2` (12
concurrent, 300 cores) exceeds 255 and evals will contend when they sync. Stay at
**6–8 concurrent**. Run the four groups back-to-back, or overlap two at 3 GPUs each.

---

## Files in this pipeline

| File | Role |
|---|---|
| `gp_reward-priors/HANDOFF_HP_SELECTION.md` | Authoritative HP-selection procedure (seed discipline, 4 stages, results) |
| `gp_reward-priors/scripts_{bnn,mr,pt}/run_*_training_antmaze_eval.py` | Phase-1 training; write deterministic `_<seed>` dirs |
| `gp_reward-priors/scripts_{bnn,mr,pt}/antmaze_<variant>_*_antmaze_eval.yaml` | per-variant training configs (stage 1–3 winners baked in) |
| `gp_reward-priors/train_rewards.sh` | Phase-1 launcher (GPU-packed, all variants × seeds 0–10) |
| `algorithms/offline/iql_eval.py` | Phase-2 IQL, seed-derived `reward_model_root` (iql.py untouched) |
| `configs/offline/iql/antmaze/<variant>.yaml` | IQL run config (`n_episodes`, `eval_freq`, `max_timesteps`) |
| `{bnn,ensemble,mr,pt}_sweeps/sweep_antmaze_*.yaml` | Phase-2 W&B sweeps (currently: seed 0 × normalize_reward 0–7) |
| `{bnn,ensemble,mr,pt}_sweeps/launch.sh` | Phase-2 launchers (W&B agents across GPUs) |
