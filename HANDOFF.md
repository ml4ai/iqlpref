# Handoff — iqlpref (IQL + preference-learning reward models)

> For the next coding assistant. Covers what was built, the design decisions
> behind it, and how this user likes to work. Read the "Working preferences"
> section first — it will save you from repeating mistakes that already
> happened in this project.

Repo: `ml4ai/iqlpref` (GitHub), default branch `main`.
Date of handoff: 2026-06-07.

---

## 1. Project at a glance

Offline RL (IQL) with three flavors of reward used to relabel the dataset:

- **Task reward** — the environment's own reward (baseline column in results).
- **MLP / PT preference reward models** — trained in the `gp_reward-priors`
  git submodule, loaded by `iql.py` to relabel transitions.
- **BNN posterior reward model** — fSGHMC posterior samples collapsed to a
  single pessimistic reward per transition (see §2.3).

Primary file: **`algorithms/offline/iql.py`** (single-file IQL: `TwinQ`,
`ValueFunction`, `GaussianPolicy`/`DeterministicPolicy`, `ImplicitQLearning`,
plus dataset-relabeling and reward-model loaders, `eval_actor`, and `train()`).

Reward-model training scripts live in the **`gp_reward-priors` submodule**:
- `scripts_mr/run_mr_training.py` — MLP Markovian reward (MR).
- `scripts_bnn/run_bnn_training_f.py`, `run_bnn_full_training_f.py` — BNN (fSGHMC).
- PT (preference transformer) trainer also under `optbnn/`.
Submodule commits must be made/pushed in the submodule first, then the parent
repo updated to point at the new submodule SHA.

Environment constraints: **gym 0.23.1** (old Vector/Env API), ≥25 CPU cores
available, typical `n_episodes=100` for eval.

---

## 2. Work completed this session (most recent first)

Key commits on `main`:

| Commit    | What |
|-----------|------|
| `cffaf9f` | Restore `SWEEPS` edits that a regenerate clobbered (see §4) |
| `3373fa6` | Add bar + box plots to results notebook |
| `b1ad492` | Strip `torch.compile` `_orig_mod.` prefix in reward-model loaders |
| `174b772` | Add antmaze `avg_steps_to_goal` eval metric |
| `e70942b` | Replace β-penalized BNN reward with empirical CVaR |
| `752f025` | Add BNN posterior reward-model support (uncertainty-penalized) |
| `dbd9d0d` | Vectorize `eval_actor` with `AsyncVectorEnv` |

(The user also makes frequent commits titled "updated"/"saved"/"new" — those
are theirs, often notebook/sweep-config edits. Don't assume the working tree
matches your last commit; re-check `git log`/`git status` before regenerating
anything.)

### 2.1 Vectorized `eval_actor` (`dbd9d0d`)

- `gym.vector.AsyncVectorEnv` runs `n_envs=25` episodes in parallel (capped at
  `n_episodes`). Old gym API: `reset()->obs`, `step()->(obs, reward, done, info)`,
  auto-reset on done.
- Picklable worker factory: module-level `_make_eval_env` + `functools.partial`
  (required for the `spawn` start method).
- Each env seeded `seed + i`, so every episode has a distinct initial state;
  the RNG advances naturally across auto-resets.
- Handles both policy output types: `Normal` distribution (`.mean`) or raw
  tensor, then `clamp(... , -max_action, max_action)`.

### 2.2 Antmaze steps-to-goal metric (`174b772`)

- `eval_actor` now returns `(scores, steps_to_goal)`.
- For antmaze (`"antmaze" in env_name.lower()`), a successful episode is one
  with positive return (sparse reward = 1 on reaching the goal, else 0). For
  each success it records the episode's step count.
- `train()` logs `avg_steps_to_goal` to W&B each eval **only for antmaze**:
  mean over successful episodes, or **`-1.0`** when none succeeded.
- Also removed a latent bug: the call site passed `discount=config.discount`,
  which `eval_actor` doesn't accept (would crash on first eval). Eval returns
  are undiscounted.

### 2.3 BNN reward model + empirical CVaR (`752f025`, then `e70942b`)

`qlearning_dataset_bnn()` relabels the dataset from a BNN posterior:

- Loads posterior weight samples from all chains:
  `<dir>/sampling_f/chain_*/sampled_weights/sampled_weights_*` (each a
  `torch.save({"sampled_weights": [...]})`).
- **Architecture inferred from weights**, no config.yaml needed:
  `input_dim = w[0].shape[0]`, `width = w[0].shape[1]`, `depth = (len(w)-2)//2`.
- **First approach (replaced):** uncertainty-penalized reward
  `r̃ = mean_θ − β·std_θ`, Welford streaming (O(N) memory).
- **Current approach — empirical CVaR (`e70942b`):**
  `r̃(s,a) = CVaR_α = mean of the worst (1−α)·S posterior reward samples`.
  Stores the full `(S, N−1)` float32 prediction matrix (~2 GB at S=500,
  N=1M), then one vectorized `np.partition` for the per-transition tail mean.
  Helpers `empirical_cvar()` and `cvar_stability_check()` (compares CVaR at S
  vs S/2; warns if mean relative diff > 0.05).
- Config flags on `TrainConfig`: `bnn_reward_model: bool`, `bnn_alpha: float =
  0.95`, `bnn_n_samples: int = 500` (β removed).
- **HARD CONSTRAINT (from the user's CVaR handoff doc):** do **NOT** add any
  reward normalization / z-scoring inside the CVaR code. Normalization is
  handled elsewhere in the IQL pipeline (`normalize_states`, `modify_reward`).

### 2.4 `torch.compile` prefix fix in reward loaders (`b1ad492`)

- MR models train with `compile_model: true`, so the net is wrapped by
  `torch.compile`, which prefixes every `state_dict` key with `_orig_mod.`.
- `load_mlp_reward_model` / `load_pt_reward_model` looked for bare keys
  (e.g. `layers.0.W`) → `KeyError`.
- Added `_strip_compile_prefix(state)` (no-op when no prefix) and applied it in
  both loaders. Existing checkpoints load without retraining.

### 2.5 Results notebook (`results/results_table.ipynb`)

Two W&B-driven tables + plots, paper-style (rows = datasets, columns =
methods grouped under "IQL with task reward" and "IQL with preference
learning"):

- **Table 1 — Performance:** per run, max eval `mean_score` over training;
  report mean ± std across seeds. `mean_score` is raw (antmaze success rate in
  [0,1]); multiplied by `SCALE = 100` for the 0–100 paper scale.
- **Table 2 — Steps-to-goal:** per run, find the step where `mean_score` peaks
  (`idxmax`) and read `avg_steps_to_goal` at that same step (both logged
  together each eval); report mean ± std. `DROP_FAILED=True` excludes runs
  whose best step never solved (`avg_steps_to_goal == -1`). Not scaled.
- **Plots** (under each table): `plot_means_bars` (grouped bars, std error
  bars) and `plot_box` (grouped box-and-whisker with per-run values as
  jittered dots and the mean marked with an `x`). Both adapt to whatever
  methods/datasets have data.
- **Data fetch:** use `run.history(keys=[...], samples=100000)` (sampled-history
  endpoint, one request per run) — **not** `run.scan_history`, which paginates
  every training step and is dramatically slower (caused an apparent "hang").
  Per-run progress is printed.
- Config cell: `ENTITY="champlin-university-of-arizona"`, `PROJECT="IQL-pref"`,
  `METRIC="mean_score"`, `STEPS_METRIC="avg_steps_to_goal"`.
- `SWEEPS[dataset][method] = sweep_id`; the user maintains the IDs here and
  adds more as results land. The user trimmed columns to
  `task_reward / MR / PT`.

---

## 3. Repo / environment specifics

- **Sweep configs:** `tr_sweeps/` (task reward), `mr_sweeps/` (MR reward),
  and `gp_reward-priors/scripts_mr/*_best.yaml` (multi-seed re-eval). W&B
  project is `IQL-pref`; sweeps grid over `seed: 0..9` and `normalize_reward`.
- **IQL configs:** `configs/offline/iql/antmaze/*.yaml`.
- **W&B metrics logged by `iql.py`:** `mean_score` (raw eval return) each
  `eval_freq`, and `avg_steps_to_goal` (antmaze only). Training metrics logged
  each `log_freq`. Both eval metrics share a step (same `wandb.log` call).

---

## 4. Lessons / gotchas (please honor these)

- **Don't regenerate the results notebook from a script.** It is under active
  manual editing by the user. Once I rebuilt it from a generator and silently
  reverted committed `SWEEPS` edits (`3373fa6`), which had to be recovered
  (`cffaf9f`). **Edit notebooks in place** (targeted cell edits, e.g.
  `NotebookEdit`), and if you must do bulk surgery, first diff against the
  committed version cell-by-cell and preserve user content.
- **CVaR code must not normalize rewards** (see §2.3).
- **Submodule discipline:** commit + push inside `gp_reward-priors` before
  bumping the pointer in the parent repo.

---

## 5. Working preferences (the user)

- **Python env:** runs code in the **`irl` conda env**. In this sandbox the
  `conda` shell function and `conda run -n irl ...` fail non-interactively
  (`__conda_exe: permission denied`). Call the interpreter by absolute path:
  **`/opt/anaconda3/envs/irl/bin/python`**. It has numpy 2.x, matplotlib,
  pandas, etc. (system `python`/`python3` lack these — don't use them).
- **Verify before committing.** Validate notebooks (JSON + `ast`-parse cells),
  smoke-test plotting/logic with synthetic data, and run the actual env when
  possible. The user often runs the code themselves and then says "commit."
- **Commit only when asked.** Don't push proactively. Commit messages: concise
  imperative subject + short body, and **end with**:
  `Co-Authored-By: Claude <model> <noreply@anthropic.com>`.
  Push to `origin main` after committing when asked.
- **Be direct about bugs.** The user appreciates when latent bugs found while
  touching nearby code are fixed and called out (e.g. the `discount` kwarg and
  the `torch.compile` prefix).
- **Handoff docs:** the user has shared `.docx` handoff documents
  (`BNN_IRL_IQL_Handoff.docx`, `CVaR_Implementation_Handoff.docx`) with
  explicit scope/constraint callouts; treat such constraints as hard
  requirements.

---

## 6. Suggested next steps / open threads

- Fill in remaining `SWEEPS` IDs (NMR column was dropped; large-maze PT not yet
  present) as sweeps finish; tables/plots extend automatically.
- If BNN/CVaR is run at high α, ensure `bnn_n_samples` is large enough for a
  stable tail (`cvar_stability_check` warns; rule of thumb S ≥ ceil(30/(1−α))).
- Consider verifying one end-to-end MR run loads with the compile-prefix fix
  (only static/loader checks were done).
