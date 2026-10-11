#!/usr/bin/env python
"""phase2_sweeps.py -- generate the Phase-2 IQL sweep files from ONE canonical spec.

PIPELINE.md's Phase 2 runs the same 24 sweep files in two lineages:

  stage4   seed 0, normalize_reward grid 0-7    (selects the normalization index)
  eval     seeds 1-10, the winning index         (the reported evaluation)

and requires that "nothing else changes" between them.  Hand-flipping 24 files
failed that in practice (gp_reward-priors HANDOFF §4.3.139).  Found on 2026-09-25:

  * `centre_draws` was set in NO deployment file, though §4.3.114 decided
    centring for the deployed reward too, BNN *and* MR-ensemble ("or §3.1's
    cross-family comparability breaks").  `iql_eval` defaults it to False, so
    every BNN and MR-ensemble IQL run would have deployed RAW CVaR while the BNN
    winners were SELECTED on centred CVaR: the selection/deployment mismatch
    §3.2.1 exists to prevent.
  * 16 of the 24 files were still flipped to seeds 1-10 with stage-4 indices from
    the OLD reward models.

So both lineages are generated from the spec below.  Every value a lineage
depends on is written explicitly, including ones equal to an iql_eval default
(`bnn_alpha: 0.95`), so a changed default cannot silently change a run.  The
generator checks that the two lineages differ ONLY in `seed` and
`normalize_reward`.

The oracle `tr_sweeps` (task reward, `iql.py`, no reward model) are covered since
2026-10-07 (HANDOFF §4.3.157).  They were run at a FIXED normalize_reward 1 while
every learned reward had its index selected, so "learned beats oracle" was partly
a statement about normalization selection.  The oracle now gets the same stage 4.
Its grid is indices 0-3 only: for a 0/1 reward whose minimum trajectory return is
0, index 4 is identical to 2, 6 to 2, 5 to 3 and 7 to 3, so 0-3 are the four
DISTINCT transformations and running 4-7 would give indices 2 and 3 two noisy
draws each in the argmax.

Subcommands (run from the repo root):
    python phase2_sweeps.py check                       # audit files vs spec
    python phase2_sweeps.py stage4 [--write]            # seed-0 grid lineage
    python phase2_sweeps.py winners SWEEP_ID [...] --out winners.json
                                                        # score finished stage-4
                                                        # sweeps via iql_score
    python phase2_sweeps.py eval --winners winners.json [--write]
                                                        # seeds 1-10 lineage
    python phase2_sweeps.py register [--write]          # put every COMPLETE
                                                        # evaluation sweep into
                                                        # results/results_table.ipynb
    python phase2_sweeps.py --selftest
Dry-run by default; --write applies.  `eval` writes only the files whose winner
is known, so families can move to evaluation as their stage 4 finishes.
"""

import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
ENTITY, IQL_PROJECT = "champlin-university-of-arizona", "IQL-pref"
VARIANTS = ("medium_play", "medium_diverse", "large_play", "large_diverse")
STAGE4_INDICES = list(range(8))
ORACLE_INDICES = [0, 1, 2, 3]                  # the distinct transformations (above)
DEFAULT_PROGRAM = "algorithms/offline/iql_eval.py"
ORACLE_PROGRAM = "algorithms/offline/iql.py"   # no seed-derived model path needed


def is_oracle(name):
    return name.startswith("tr_sweeps/")


def grid_of(name):
    return ORACLE_INDICES if is_oracle(name) else STAGE4_INDICES


def program_of(name):
    return ORACLE_PROGRAM if is_oracle(name) else DEFAULT_PROGRAM
EVAL_SEEDS = list(range(1, 11))
LINEAGE_KEYS = ("seed", "normalize_reward")    # the ONLY keys that may differ


def spec():
    """{relative_path_without_.yaml: ordered {param: value}} for all 28 files
    (24 learned-reward sweeps + 4 oracle)."""
    out = {}
    for v in VARIANTS:
        base = {"config_path": f"configs/offline/iql/antmaze/{v}_v2.yaml"}
        root = lambda fam: f"~/iqlpref/exp/reward_learning/antmaze_{v}_{fam}_eval"
        for kind, alpha in (("cvar", 0.95), ("mean", 0.0)):
            out[f"bnn_sweeps/sweep_antmaze_{v}_{kind}"] = dict(
                base, reward_model_root=root("bnn"), bnn_reward_model=True,
                bnn_n_samples=-1,          # all draws: label-cache key (4.3.129)
                query_length=1, bnn_alpha=alpha,
                centre_draws=True)         # item F, deployment side (4.3.114)
            out[f"ensemble_sweeps/sweep_antmaze_{v}_{kind}"] = dict(
                base, reward_model_root=root("mr"), mr_ensemble=True,
                query_length=1, mr_alpha=alpha,
                mr_burn_in=100,            # inherited, not revisited
                centre_draws=True)         # 4.3.114: MR-ensemble-CVaR centred too
        out[f"mr_sweeps/sweep_antmaze_{v}"] = dict(
            base, reward_model_root=root("mr"), query_length=1)
        out[f"pt_sweeps/sweep_antmaze_{v}"] = dict(
            base, reward_model_root=root("pt"), query_length=100)
        out[f"tr_sweeps/sweep_antmaze_{v}"] = dict(base)     # oracle: nothing else
    return out


def lineage_params(fixed, lineage, index=None, grid=STAGE4_INDICES):
    p = dict(fixed)
    if lineage == "stage4":
        p["seed"] = {"value": 0}
        p["normalize_reward"] = {"values": list(grid)}
    elif lineage == "eval":
        if index not in grid:
            raise ValueError(f"winning index must be in {list(grid)}, got {index!r}")
        p["seed"] = {"values": EVAL_SEEDS}
        p["normalize_reward"] = {"value": index}
    else:
        raise ValueError(lineage)
    return {k: (v if isinstance(v, dict) else {"value": v}) for k, v in p.items()}


def render(name, params, lineage, index=None):
    g = grid_of(name)
    head = [
        "# GENERATED by phase2_sweeps.py -- do not hand-edit; regenerate instead",
        "# (gp_reward-priors HANDOFF §4.3.139).  Lineage: "
        + (f"STAGE 4, seed 0, normalize_reward grid {g[0]}-{g[-1]} (selection)."
           if lineage == "stage4" else
           f"EVALUATION, seeds 1-10, normalize_reward {index} (stage-4 winner)."),
    ]
    doc = {"project": IQL_PROJECT, "program": program_of(name),
           "method": "grid", "parameters": params}
    return "\n".join(head) + "\n" + yaml.safe_dump(doc, sort_keys=False)


def flat(params):
    """{k: value-or-values} from a sweep 'parameters' block."""
    return {k: (v.get("value", v.get("values")) if isinstance(v, dict) else v)
            for k, v in (params or {}).items()}


def lineage_of(p, grid=STAGE4_INDICES):
    s, n = p.get("seed"), p.get("normalize_reward")
    if s == 0 and n == list(grid):
        return "stage4"
    if s == EVAL_SEEDS and isinstance(n, int):
        return "eval"
    return "INVALID"


def audit(name, doc, fixed):
    """(lineage, [problems]) for one existing file against the spec."""
    p = flat((doc or {}).get("parameters"))
    probs = []
    for k, v in fixed.items():
        if k not in p:
            probs.append(f"{k} missing (spec {v!r})")
        elif p[k] != v:
            probs.append(f"{k}={p[k]!r} (spec {v!r})")
    for k in p:
        if k not in fixed and k not in LINEAGE_KEYS:
            probs.append(f"{k} not in spec")
    if (doc or {}).get("program") != program_of(name):
        probs.append(f"program {(doc or {}).get('program')!r}")
    return lineage_of(p, grid_of(name)), probs


def match_entry(sweep_params, sp):
    """The spec entries whose fixed params EXACTLY equal a sweep's non-lineage
    params.  Exact, not subset: MR best-model's params {config_path, root,
    query_length} are a strict subset of the MR ensemble's, so a subset match
    would file an ensemble sweep under MR best-model as well."""
    p = {k: v for k, v in flat(sweep_params).items() if k not in LINEAGE_KEYS}
    return [n for n, fixed in sp.items() if p == fixed]


GENERATED_MARK = "# GENERATED by phase2_sweeps.py"


def cmd_check(sp):
    bad = 0
    for name, fixed in sp.items():
        path = os.path.join(ROOT, name + ".yaml")
        text = open(path).read() if os.path.exists(path) else ""
        doc = yaml.safe_load(text) if text else None
        lin, probs = audit(name, doc, fixed)
        # A hand-made file cannot show WHICH stage 4 its index came from; an
        # evaluation-lineage file from before the generator carries an index
        # from the old reward models.
        if text and GENERATED_MARK not in text:
            probs = probs + ["not generated (predates phase2_sweeps.py; an eval "
                             "index here is from an unknown stage 4)"]
        bad += bool(probs) or lin == "INVALID"
        print(f"  {name:48s} {lin:8s} {'OK' if not probs else '; '.join(probs)}")
    print(f"\n  {len(sp) - bad}/{len(sp)} files match the spec")
    return 0 if not bad else 1


def write_all(sp, lineage, winners, do_write, only=None):
    n = 0
    for name, fixed in sp.items():
        if only and not name.startswith(only):
            continue
        idx = None
        if lineage == "eval":
            if name not in winners:
                print(f"  {name:48s} SKIP (no stage-4 winner yet)")
                continue
            idx = winners[name]
        params = lineage_params(fixed, lineage, idx, grid_of(name))
        text = render(name, params, lineage, idx)
        back = yaml.safe_load(text)
        # audit() requires every non-lineage key to equal the spec and forbids
        # extras -- which IS the invariant that the two lineages differ only in
        # seed/normalize_reward, since both are the same spec plus those two.
        lin, probs = audit(name, back, fixed)
        assert lin == lineage and not probs, (name, lin, probs)
        path = os.path.join(ROOT, name + ".yaml")
        old = open(path).read() if os.path.exists(path) else ""
        change = "unchanged" if old == text else "CHANGED"
        if do_write and old != text:
            open(path, "w").write(text)
        n += 1
        print(f"  {name:48s} {lineage}{'' if idx is None else f' idx {idx}'}  "
              f"{change}{' (written)' if do_write and old != text else ''}")
    print(f"\n  {n} file(s) {'written' if do_write else 'checked (dry run; --write to apply)'}")
    return 0


def cmd_winners(sweep_ids, sp, out):
    sys.path.insert(0, os.path.join(ROOT, "results"))
    import iql_score
    import wandb
    api = wandb.Api(timeout=120)
    winners = json.load(open(out)) if out and os.path.exists(out) else {}
    for sid in sweep_ids:
        path = sid if sid.count("/") == 2 else f"{ENTITY}/{IQL_PROJECT}/{sid}"
        sw = api.sweep(path)
        hits = match_entry(sw.config.get("parameters"), sp)
        if len(hits) != 1:
            sys.exit(f"{sid}: matches {len(hits)} spec entries {hits} -- "
                     "not a stage-4 sweep generated from this spec?")
        grid = grid_of(hits[0])
        if lineage_of(flat(sw.config.get("parameters")), grid) != "stage4":
            sys.exit(f"{sid}: not a stage-4 (seed 0, grid {grid[0]}-{grid[-1]}) sweep")
        if sw.config.get("program") != program_of(hits[0]):
            sys.exit(f"{sid}: program {sw.config.get('program')!r}, "
                     f"expected {program_of(hits[0])!r}")
        scored = []
        for r in sw.runs:
            s = iql_score.score_run(r)
            s["normalize_reward"] = (r.config or {}).get("normalize_reward")
            scored.append(s)
        win, table = iql_score.select_normalization(scored, expected_indices=grid)
        winners[hits[0]] = int(win)
        print(f"  {hits[0]:48s} winner idx {win}  ({table[0]['score']:.4f}; "
              f"runner-up idx {table[1]['normalize_reward']} {table[1]['score']:.4f})")
    if out:
        json.dump(winners, open(out, "w"), indent=2, sort_keys=True)
        print(f"\n  {len(winners)} winner(s) in {out}")
    return 0


# ---------------------------------------------------------------------------
# register: results/results_table.ipynb's sweep registry, from wandb
# ---------------------------------------------------------------------------
NOTEBOOK = os.path.join("results", "results_table.ipynb")
REGISTRY_CELL, REGISTRY_DOC_CELL = "732f3aed", "cbfd8e4d"     # notebook cell ids
ORACLE_DOC_CELL, ORACLE_CODE_CELL = "0c1e5e1a", "0c1e5e1b"   # appended at the end
ORACLE_CONVENTIONAL = 1                    # r - 1, the index the oracle ran at
TOTAL_STEPS = 1_000_000
# (method key, (column group, subheader)) in table order.  Labels are the user's.
# The oracle column is the STAGE-4-SELECTED index only (user, 2026-10-10); the
# conventional r - 1 sweeps go to ORACLE_CONVENTIONAL and the end-of-notebook
# comparison, never into the main tables or charts.
COLUMNS = [
    ("task_reward", ("IQL with task reward", "")),
    ("MR", ("IQL with preference learning", "MR")),
    ("PT", ("IQL with preference learning", "PT")),
    ("ENS_MEAN", ("IQL with preference learning", "ENS w/ MEAN")),
    ("ENS_CVAR", ("IQL with preference learning", "ENS w/ CVaR")),
    ("BNN_MEAN", ("IQL with preference learning", "BNN w/ MEAN")),
    ("BNN_CVAR", ("IQL with preference learning", "BNN w/ CVaR")),
]


def method_of(name):
    fam, base = name.split("/")
    if fam == "tr_sweeps":
        return "task_reward"
    if fam in ("mr_sweeps", "pt_sweeps"):
        return fam[:2].upper()
    return ("BNN" if fam == "bnn_sweeps" else "ENS") + \
        ("_CVAR" if base.endswith("_cvar") else "_MEAN")


def dataset_of(name):
    base = name.split("/")[1][len("sweep_antmaze_"):]
    for suffix in ("_cvar", "_mean"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    return f"antmaze-{base.replace('_', '-')}-v2"


def classify(name, index, complete, n_done, winners):
    """(status, note) for one evaluation-lineage sweep of spec entry `name`.

    REGISTER   complete, and at the stage-4 winning index
    R1         oracle only: complete at the conventional index when stage 4
               picked another one.  Kept OUT of the main table; it feeds the
               winning-vs-conventional comparison (§4.3.157, §4.3.165)
    PENDING    right index, not all 10 seeds finished yet
    REJECT     wrong index, or no stage-4 winner on record
    """
    win = winners.get(name)
    if is_oracle(name) and win is None:
        if index != ORACLE_CONVENTIONAL:
            return "REJECT", f"idx {index} but oracle stage 4 has no winner yet"
        # Not a winner until stage 4 says so: hold it as the conventional sweep.
        return (("R1", f"idx {index} (conventional; oracle stage 4 pending)")
                if complete else ("PENDING", f"idx {index}, {n_done}/10 seeds"))
    elif win is None:
        return "REJECT", f"idx {index} but no stage-4 winner on record"
    elif index == win:
        note = f"idx {index} (stage-4 winner)"
    elif is_oracle(name) and index == ORACLE_CONVENTIONAL:
        return (("R1", f"idx {index} (conventional; stage 4 picked {win})")
                if complete else ("PENDING", f"idx {index}, {n_done}/10 seeds"))
    else:
        return "REJECT", f"idx {index} but the stage-4 winner is {win}"
    return ("REGISTER", note) if complete else ("PENDING", f"{note}, {n_done}/10 seeds")


def build_registry(found, sp):
    """found: [(name, sweep_id, status, note)].
    -> (registry, notes, problems, conventional).

    registry[dataset][method] = sweep_id or None: the main table, in which the
    oracle cell is the stage-4-selected index ONLY.  conventional[dataset] = the
    oracle's complete index-1 sweep, whether or not index 1 also won.  Two
    complete sweeps for one cell is a problem, never a silent pick."""
    reg = {dataset_of(n): {} for n in sp}
    notes, probs, conventional = {}, [], {}
    for n in sp:
        reg[dataset_of(n)][method_of(n)] = None
        notes[(dataset_of(n), method_of(n))] = "no evaluation sweep yet"
    for name, sid, status, note in found:
        d, m = dataset_of(name), method_of(name)
        if is_oracle(name) and status in ("REGISTER", "R1") \
                and note.startswith(f"idx {ORACLE_CONVENTIONAL} "):
            if conventional.get(d):
                probs.append(f"{d}: two complete index-{ORACLE_CONVENTIONAL} oracle "
                             f"sweeps, {conventional[d]} and {sid}")
            else:
                conventional[d] = sid
        if status == "R1":                 # never in the main table
            if not reg[d].get(m):
                notes[(d, m)] = f"winning-index sweep not there yet; {sid} is {note}"
            continue
        if status == "REGISTER":
            if reg[d].get(m):
                probs.append(f"{d} {m}: two complete sweeps, {reg[d][m]} and {sid}")
                continue
            reg[d][m], notes[(d, m)] = sid, note
        elif status == "PENDING":
            if not reg[d].get(m):
                notes[(d, m)] = f"running: {sid}, {note}"
        else:
            probs.append(f"{name} {sid}: {note}")
    return reg, notes, probs, conventional


def registry_cell(reg, notes, conventional=None, oracle_index=None):
    """Source of the notebook's registry code cell."""
    conventional, oracle_index = conventional or {}, oracle_index or {}
    out = ["# GENERATED by `phase2_sweeps.py register --write` (run from the repo root).",
           "# Do not hand-edit: re-run it.  A sweep is registered only when it is",
           "# COMPLETE (seeds 1-10 all finished) at the stage-4 winning index.",
           "# None = not there yet; the comment says why.",
           "SWEEPS = {"]
    for d in reg:
        out.append(f'    "{d}": {{')
        for m, _ in COLUMNS:
            sid = reg[d].get(m)
            val = f'"{sid}"' if sid else "None"
            out.append(f'        "{m}": {val},  # {notes.get((d, m), "")}')
        out.append("    },")
    out += ["}", "",
            "# (method_key, (column group, column subheader)): table layout and order.",
            "METHOD_COLUMNS = ["]
    out += [f'    ("{m}", ("{g}", "{s}")),' for m, (g, s) in COLUMNS]
    out += ["]", "",
            "# The oracle (task reward).  The tables and charts above use the stage-4",
            "# WINNING index only (SWEEPS[...][\"task_reward\"]).  These two feed the",
            "# winning-vs-conventional comparison at the end of the notebook:",
            "#   ORACLE_INDEX         the stage-4 winning index (None = stage 4 not done)",
            f"#   ORACLE_CONVENTIONAL  the sweep at the conventional index {ORACLE_CONVENTIONAL} (r - 1)",
            "ORACLE_INDEX = {"]
    out += [f'    "{d}": {oracle_index.get(d)!r},' for d in reg]
    out += ["}", "ORACLE_CONVENTIONAL = {"]
    for d in reg:
        sid = conventional.get(d)
        out.append(f'    "{d}": ' + (f'"{sid}"' if sid else "None") + ",")
    out.append("}")
    return "\n".join(out)


def registry_doc(reg, notes, probs):
    done = sum(1 for d in reg for m, _ in COLUMNS if reg[d].get(m))
    todo = [f"- `{d}` **{m}**: {notes[(d, m)]}" for d in reg for m, _ in COLUMNS
            if not reg[d].get(m)]
    txt = ["## Sweep registry", "",
           "`SWEEPS[dataset][method] = sweep_id`, mapped to table columns by",
           "`METHOD_COLUMNS`. **This cell and the next are generated** by",
           "`phase2_sweeps.py register --write`, run from the repo root. It finds",
           "each evaluation sweep on W&B by an exact match on its parameters and",
           "registers it only when all of seeds 1-10 have finished at the stage-4",
           "winning index (`gp_reward-priors/HANDOFF_HP_SELECTION.md` §6).",
           "Unregistered cells render blank.", "",
           "**The task-reward (oracle) column is the stage-4 winning index only.**",
           "Where that differs from the conventional `r - 1`, the two are compared",
           "in the last section of this notebook, not in the main tables.", "",
           f"**{done} of {len(reg) * len(COLUMNS)} cells registered.**"]
    if todo:
        txt += ["", "Not yet registered:", ""] + todo
    if probs:
        txt += ["", "**Problems found (nothing was registered for these):**", ""] + \
               [f"- {x}" for x in probs]
    return "\n".join(txt)


ORACLE_DOC = """# Oracle: winning index vs the conventional index

Every learned reward had its `normalize_reward` index selected on seed 0 (stage
4). The oracle was given the same stage 4, over the four transformations that are
distinct for a 0/1 reward (indices 0-3), so the tables above compare like with
like: **they show the oracle at its stage-4 winning index only.**

The convention for the antmaze task reward is index 1 (`r - 1`). Where stage 4
picked a different index, this section compares the two on the same seeds 1-10.
Variants where index 1 won are listed, not plotted: there the winning and the
conventional oracle are the same sweep."""

ORACLE_CODE = """# Oracle: stage-4 winning index vs the conventional index 1 (r - 1).
# Uses SWEEPS / ORACLE_INDEX / ORACLE_CONVENTIONAL from the registry cell and
# fetch_run_scores / summarize / fmt from the cells above.
oracle_cmp, oracle_status = {}, []
for dataset in SWEEPS:
    sel = SWEEPS[dataset].get("task_reward")
    conv = ORACLE_CONVENTIONAL.get(dataset)
    idx = ORACLE_INDEX.get(dataset)
    if idx is None:
        oracle_status.append((dataset, "stage 4 not finished"))
    elif idx == 1:
        oracle_status.append((dataset, "index 1 won: winning = conventional"))
    elif not sel:
        oracle_status.append((dataset, f"index {idx} won; its seeds 1-10 sweep is not complete yet"))
    elif not conv:
        oracle_status.append((dataset, f"index {idx} won; no conventional sweep registered"))
    else:
        p_sel, r_sel, _ = fetch_run_scores(sel, verbose=False)
        p_conv, r_conv, _ = fetch_run_scores(conv, verbose=False)
        oracle_cmp[dataset] = {
            "index": idx,
            "winning": summarize(p_sel), "conventional": summarize(p_conv),
            "winning_robust": summarize(r_sel), "conventional_robust": summarize(r_conv),
        }
        oracle_status.append((dataset, f"index {idx} won: compared below"))

for dataset, status in oracle_status:
    print(f"{dataset:28s} {status}")

if not oracle_cmp:
    print("\\nNothing to compare yet.")
else:
    oracle_table = pd.DataFrame(
        {
            dataset: {
                "winning index": c["index"],
                f"winning (last-{LAST_N})": fmt(c["winning"]),
                f"conventional r - 1 (last-{LAST_N})": fmt(c["conventional"]),
                "difference of means": round(c["winning"]["mean"] - c["conventional"]["mean"], 2),
                f"winning (last-{ROBUST_N})": fmt(c["winning_robust"]),
                f"conventional r - 1 (last-{ROBUST_N})": fmt(c["conventional_robust"]),
            }
            for dataset, c in oracle_cmp.items()
        }
    ).T
    oracle_table.index.name = "Dataset"
    display(oracle_table)

    # Box-and-whisker with the per-seed scores as dots and the mean as an x,
    # in the style of plot_box above.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {"winning": cycle[0], "conventional": cycle[1]}
    names = list(oracle_cmp)
    rng = np.random.default_rng(0)
    width = 0.34
    fig, ax = plt.subplots(figsize=(max(5.0, 2.6 * len(names) + 2.0), 5))
    for i, dataset in enumerate(names):
        for key, off in (("winning", -width / 2), ("conventional", width / 2)):
            vals = np.asarray(oracle_cmp[dataset][key]["values"], dtype=float)
            bp = ax.boxplot(
                [vals], positions=[i + off], widths=width * 0.85, patch_artist=True,
                showfliers=False, showmeans=True,
                meanprops=dict(marker="x", markeredgecolor="black", markersize=9,
                               markeredgewidth=2),
                medianprops=dict(color="black"),
            )
            bp["boxes"][0].set_facecolor(colors[key])
            bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor(colors[key])
            ax.scatter(i + off + rng.uniform(-0.04, 0.04, size=len(vals)), vals,
                       color=colors[key], edgecolor="white", linewidth=0.5, s=28, zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{d}\\n(winning index {oracle_cmp[d]['index']})" for d in names]
    )
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(f"Score (last-{LAST_N} mean of mean_score × 100)")
    ax.set_title("Oracle (task reward): stage-4 winning index vs conventional r - 1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor=colors["winning"], alpha=0.35, edgecolor=colors["winning"],
                  label="winning index (used in the tables above)"),
            Patch(facecolor=colors["conventional"], alpha=0.35,
                  edgecolor=colors["conventional"], label="conventional index 1 (r - 1)"),
            Line2D([], [], marker="x", color="black", linestyle="none", label="mean"),
        ]
    )
    fig.tight_layout()
    plt.show()"""


def _as_source(text):
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def ensure_cell(nb, cell_id, cell_type, text):
    """Set the source of the cell with this id, appending it at the END of the
    notebook if it does not exist yet."""
    for c in nb["cells"]:
        if c.get("id") == cell_id:
            c["source"] = _as_source(text)
            return
    cell = {"cell_type": cell_type, "id": cell_id, "metadata": {}, "source": _as_source(text)}
    if cell_type == "code":
        cell.update(execution_count=None, outputs=[])
    nb["cells"].append(cell)


def cmd_register(sp, winners_path, do_write, notebook=NOTEBOOK):
    import wandb
    api = wandb.Api(timeout=120)
    winners = json.load(open(winners_path)) if os.path.exists(winners_path) else {}
    found = []
    for sw in api.project(IQL_PROJECT, entity=ENTITY).sweeps():
        params = sw.config.get("parameters")
        hits = match_entry(params, sp)
        if len(hits) != 1:
            print(f"  ?? {sw.id}: matches {len(hits)} spec entries -- ignored")
            continue
        name = hits[0]
        if lineage_of(flat(params), grid_of(name)) != "eval":
            continue                                   # a stage-4 grid
        if sw.config.get("program") != program_of(name):
            found.append((name, sw.id, "REJECT", f"program {sw.config.get('program')!r}"))
            continue
        runs = list(sw.runs)
        done = sorted(r.config.get("seed") for r in runs if r.state == "finished"
                      and (r.summary.get("_step") or 0) >= TOTAL_STEPS)
        complete = done == EVAL_SEEDS and len(runs) == len(EVAL_SEEDS)
        index = flat(params)["normalize_reward"]
        found.append((name, sw.id) + classify(name, index, complete, len(done), winners))
    reg, notes, probs, conventional = build_registry(found, sp)
    oracle_index = {dataset_of(n): winners.get(n) for n in sp if is_oracle(n)}
    for d in reg:
        print(f"  {d}")
        for m, _ in COLUMNS:
            print(f"    {m:15s} {str(reg[d].get(m)):10s} {notes.get((d, m), '')}")
        print(f"    {'(oracle r - 1)':15s} {str(conventional.get(d)):10s} "
              f"stage-4 winning index: {oracle_index.get(d)}")
    for x in probs:
        print(f"  !! {x}")
    path = os.path.join(ROOT, notebook)
    raw = open(path).read()
    nb = json.loads(raw)
    cells = {c.get("id"): c for c in nb["cells"]}
    if REGISTRY_CELL not in cells or REGISTRY_DOC_CELL not in cells:
        sys.exit(f"{notebook}: registry cells {REGISTRY_CELL}/{REGISTRY_DOC_CELL} not found")
    cells[REGISTRY_CELL]["source"] = _as_source(
        registry_cell(reg, notes, conventional, oracle_index))
    cells[REGISTRY_DOC_CELL]["source"] = _as_source(registry_doc(reg, notes, probs))
    ensure_cell(nb, ORACLE_DOC_CELL, "markdown", ORACLE_DOC)
    ensure_cell(nb, ORACLE_CODE_CELL, "code", ORACLE_CODE)
    new = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    n_reg = sum(1 for d in reg for m in reg[d] if reg[d][m])
    state = "unchanged" if new == raw else "CHANGED"
    if do_write and new != raw:
        open(path, "w").write(new)
        state += " (written)"
    print(f"\n  {n_reg} of {len(reg) * len(COLUMNS)} cells registered; {notebook} {state}"
          f"{'' if do_write else ' (dry run; --write to apply)'}")
    return 1 if probs else 0


def selftest():
    sp = spec()
    assert len(sp) == 28, len(sp)
    assert sum(1 for n in sp if n.startswith("tr_")) == 4
    assert sum(1 for n in sp if n.startswith("bnn_")) == 8
    for n, f in sp.items():
        if n.startswith(("bnn_", "ensemble_")):
            assert f["centre_draws"] is True, n            # item F, both families
        if n.startswith("bnn_"):
            assert f["bnn_n_samples"] == -1, n
            assert f["bnn_alpha"] == (0.95 if n.endswith("cvar") else 0.0), n
    # lineages differ only in seed/normalize_reward
    f = sp["bnn_sweeps/sweep_antmaze_large_play_cvar"]
    a, b = flat(lineage_params(f, "stage4")), flat(lineage_params(f, "eval", 3))
    assert {k for k in a if a[k] != b[k]} == {"seed", "normalize_reward"}
    assert a["seed"] == 0 and a["normalize_reward"] == list(range(8))
    assert b["seed"] == list(range(1, 11)) and b["normalize_reward"] == 3
    # render round-trips and audits clean
    t = render("x", lineage_params(f, "eval", 3), "eval", 3)
    lin, probs = audit("x", yaml.safe_load(t), f)
    assert lin == "eval" and not probs, (lin, probs)
    # the pre-fix file (no centre_draws, no bnn_alpha) must FAIL the audit
    old = {"program": "algorithms/offline/iql_eval.py", "parameters": {
        k: {"value": v} for k, v in f.items() if k not in ("centre_draws", "bnn_alpha")}}
    old["parameters"].update(seed={"value": 0}, normalize_reward={"values": list(range(8))})
    lin, probs = audit("x", old, f)
    assert any("centre_draws missing" in p for p in probs), probs
    # a sweep maps to exactly one entry; mean and cvar are distinguishable
    hits = match_entry(lineage_params(f, "stage4"), sp)
    assert hits == ["bnn_sweeps/sweep_antmaze_large_play_cvar"], hits
    assert match_entry(lineage_params(sp["ensemble_sweeps/sweep_antmaze_medium_play_mean"],
                                      "stage4"), sp) == \
        ["ensemble_sweeps/sweep_antmaze_medium_play_mean"]
    # MR best-model and MR ensemble share a root but must not collide
    assert match_entry(lineage_params(sp["mr_sweeps/sweep_antmaze_medium_play"],
                                      "stage4"), sp) == ["mr_sweeps/sweep_antmaze_medium_play"]
    try:
        lineage_params(f, "eval", 9)
        raise AssertionError("accepted index 9")
    except ValueError:
        pass
    # oracle: iql.py, grid 0-3, config_path only, and it maps to itself alone
    o = "tr_sweeps/sweep_antmaze_large_play"
    fo = sp[o]
    assert set(fo) == {"config_path"} and program_of(o).endswith("/iql.py")
    ts = render(o, lineage_params(fo, "stage4", None, grid_of(o)), "stage4")
    lin, probs = audit(o, yaml.safe_load(ts), fo)
    assert lin == "stage4" and not probs, (lin, probs)
    assert "Lineage: STAGE 4" in ts and "grid 0-3" in ts and "iql.py" in ts
    assert flat(yaml.safe_load(ts)["parameters"])["normalize_reward"] == [0, 1, 2, 3]
    te = render(o, lineage_params(fo, "eval", 1, grid_of(o)), "eval", 1)
    lin, probs = audit(o, yaml.safe_load(te), fo)
    assert lin == "eval" and not probs and "Lineage: EVALUATION" in te
    assert match_entry(lineage_params(fo, "stage4", None, grid_of(o)), sp) == [o]
    # an 8-index grid is NOT the oracle's stage 4, and index 5 is not a valid winner
    assert audit(o, {"program": ORACLE_PROGRAM, "parameters": lineage_params(
        fo, "stage4")}, fo)[0] == "INVALID"
    try:
        lineage_params(fo, "eval", 5, grid_of(o))
        raise AssertionError("accepted oracle index 5")
    except ValueError:
        pass
    # the oracle file audited as a learned-reward program must fail
    bad = yaml.safe_load(ts); bad["program"] = DEFAULT_PROGRAM
    assert any("program" in x for x in audit(o, bad, fo)[1])
    # ---- register ----
    assert method_of("bnn_sweeps/sweep_antmaze_large_play_cvar") == "BNN_CVAR"
    assert method_of("ensemble_sweeps/sweep_antmaze_medium_diverse_mean") == "ENS_MEAN"
    assert method_of("mr_sweeps/sweep_antmaze_medium_play") == "MR"
    assert method_of("pt_sweeps/sweep_antmaze_large_diverse") == "PT"
    assert method_of(o) == "task_reward"
    assert dataset_of("bnn_sweeps/sweep_antmaze_medium_diverse_cvar") == \
        "antmaze-medium-diverse-v2" == dataset_of("mr_sweeps/sweep_antmaze_medium_diverse")
    b = "bnn_sweeps/sweep_antmaze_large_play_cvar"
    W = {b: 3}
    assert classify(b, 3, True, 10, W)[0] == "REGISTER"
    assert classify(b, 3, False, 6, W)[0] == "PENDING"
    assert classify(b, 2, True, 10, W)[0] == "REJECT"           # not the winner
    assert classify(b, 3, True, 10, {})[0] == "REJECT"          # no winner on record
    assert classify(o, 1, True, 10, {})[0] == "R1"       # stage 4 pending: not a winner yet
    assert classify(o, 2, True, 10, {})[0] == "REJECT"
    assert classify(o, 1, True, 10, {o: 1})[0] == "REGISTER"
    assert classify(o, 1, True, 10, {o: 2})[0] == "R1"          # kept as conventional
    assert classify(o, 2, True, 10, {o: 2})[0] == "REGISTER"
    assert classify(o, 3, True, 10, {o: 2})[0] == "REJECT"
    L, M = "antmaze-large-play-v2", "antmaze-medium-play-v2"
    om = "tr_sweeps/sweep_antmaze_medium_play"
    # large_play: stage 4 picked 2 and both sweeps are complete; medium_play: index 1 won
    reg, notes, probs, conv = build_registry(
        [(b, "aaa", "REGISTER", "idx 3 (stage-4 winner)"),
         (o, "ooo", "R1", "idx 1 (conventional; stage 4 picked 2)"),
         (o, "nnn", "REGISTER", "idx 2 (stage-4 winner)"),
         (om, "ppp", "REGISTER", "idx 1 (stage-4 winner)"),
         ("mr_sweeps/sweep_antmaze_large_play", "mmm", "PENDING", "idx 3, 4/10 seeds")], sp)
    assert not probs
    assert reg[L]["BNN_CVAR"] == "aaa" and reg[L]["MR"] is None
    assert "running: mmm" in notes[(L, "MR")]
    # the main table carries the WINNING index only; index 1 never enters it
    assert reg[L]["task_reward"] == "nnn" and reg[M]["task_reward"] == "ppp"
    assert all("task_reward_r1" not in v for v in reg.values())
    assert conv == {L: "ooo", M: "ppp"}                # index-1 sweeps, both variants
    ns = {}
    exec(registry_cell(reg, notes, conv, {L: 2, M: 1}), ns)      # valid Python
    assert ns["SWEEPS"][L]["task_reward"] == "nnn" and ns["SWEEPS"][L]["MR"] is None
    assert len(ns["SWEEPS"]) == 4 and len(ns["METHOD_COLUMNS"]) == 7   # never an r1 column
    assert [m for m, _ in ns["METHOD_COLUMNS"]].count("task_reward") == 1
    assert ns["ORACLE_CONVENTIONAL"][L] == "ooo" and ns["ORACLE_CONVENTIONAL"][M] == "ppp"
    assert ns["ORACLE_INDEX"][L] == 2 and ns["ORACLE_INDEX"][M] == 1
    assert ns["ORACLE_INDEX"]["antmaze-large-diverse-v2"] is None
    # winning index picked but its sweep not complete: the main cell is BLANK,
    # the conventional sweep is still available for the comparison section
    reg5, _, _, conv5 = build_registry(
        [(o, "ooo", "R1", "idx 1 (conventional; stage 4 picked 3)")], sp)
    assert reg5[L]["task_reward"] is None and conv5 == {L: "ooo"}
    # oracle stage 4 still pending: the index-1 sweep is NOT in the main table
    reg6, notes6, _, conv6 = build_registry(
        [(o,) + ("ooo",) + classify(o, 1, True, 10, {})], sp)
    assert reg6[L]["task_reward"] is None and conv6 == {L: "ooo"}
    assert "ooo" in notes6[(L, "task_reward")]
    _, _, probs, _ = build_registry(
        [(b, "aaa", "REGISTER", ""), (b, "bbb", "REGISTER", "")], sp)
    assert probs and "two complete sweeps" in probs[0]
    assert _as_source("a\nb") == ["a\n", "b"]
    compile(ORACLE_CODE, "<oracle cell>", "exec")               # the cell parses
    nbt = {"cells": [{"id": "x", "cell_type": "code", "source": []}]}
    ensure_cell(nbt, ORACLE_DOC_CELL, "markdown", "a\nb")
    ensure_cell(nbt, ORACLE_CODE_CELL, "code", "c")
    ensure_cell(nbt, ORACLE_CODE_CELL, "code", "d")              # update, not a 2nd copy
    assert [c["id"] for c in nbt["cells"]] == ["x", ORACLE_DOC_CELL, ORACLE_CODE_CELL]
    assert nbt["cells"][-1]["source"] == ["d"] and nbt["cells"][-1]["outputs"] == []
    assert "outputs" not in nbt["cells"][1]
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?",
                    choices=("check", "stage4", "eval", "winners", "register"))
    ap.add_argument("sweep_ids", nargs="*")
    ap.add_argument("--winners", default=None, help="JSON {file: index} for eval")
    ap.add_argument("--out", default="phase2_winners.json")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", default=None, metavar="PREFIX",
                    help="restrict stage4/eval to files starting with PREFIX "
                         "(e.g. tr_sweeps).  `stage4 --write` with no --only "
                         "rewrites EVERY file to the stage-4 lineage.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sp = spec()
    if a.cmd == "check":
        return cmd_check(sp)
    if a.cmd == "stage4":
        return write_all(sp, "stage4", {}, a.write, a.only)
    if a.cmd == "eval":
        if not a.winners:
            ap.error("eval needs --winners winners.json (from `winners`)")
        return write_all(sp, "eval", json.load(open(a.winners)), a.write, a.only)
    if a.cmd == "register":
        return cmd_register(sp, a.winners or a.out, a.write)
    if a.cmd == "winners":
        if not a.sweep_ids:
            ap.error("winners needs the stage-4 sweep ids launch.sh printed")
        return cmd_winners(a.sweep_ids, sp, a.out)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
