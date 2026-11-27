#!/usr/bin/env python
"""replication.py -- the replication's IQL sweep files (gp_reward-priors HANDOFF §14.83).

The replication re-runs §14.5's main hypothesis on NEW reward models (selected and trained at
seeds 100-110, gp_reward-priors/scripts_bnn/make_repl_selection.py) under the CORRECTED
normalization: return_range_split "length" at every index, in BOTH protocols, so the
normalized reward never depends on the terminals (results/leak_measure.py --audit: PASS).
Study 1's files, specs and readers are untouched; every file here lives in repl_sweeps/.

  THE RULE (§14.104): the normalization test decides the index.  `replication.py rule <N|F|O3|S>
  --write` records the adopted rule in results/repl_rule.json, once, from the test's registered
  reading.  Until then only the model-free trivial cells are written.
    N, F, O3  a FIXED index for every learned family (N 0, F 2, O3 the test's one index per
              dataset), both protocols: NO grids at seed 100 (§14.104's decision rule); the
              task reward at index 1, r - 1, no grid (§14.108 item 2, §14.163)
    S         the leak-free selection at seed 100, the grids below (§14.83 decision 4)

  GRIDS  (rule S only; seed 100, full data; Study 1's stage-4 rule picks the index)
    repl_sweeps/grid_{std,cand}_antmaze_<v>_<fam>   fam in mr, pt, nmr, ens_mean: indices 0-7
    repl_sweeps/grid_cand_antmaze_<v>_task          the task reward, candidate: indices 0-3
  CELLS  (seeds 101-110, at the rule's index or the grid winner; nothing re-selected under a condition)
    repl_sweeps/{std,cand}_<level>_antmaze_<v>_<fam>   level in full, r16-r128, n01-n045
    repl_sweeps/cand_full_antmaze_<v>_task             usability (§14.5)
    repl_sweeps/cand_{zero,mean}_antmaze_<v>           usability and test (c), index 0

  Every file is Study 1's for the same cell with these changes and no others (checked on
  the rendered text): return_range_split "length"; seq_labeller 3 for PT and NMR (the head
  count read correctly for ANY PT, §14.28 -- 2 was kept in Study 1 only so old runs stayed
  matched, and the replication's PT winners are not known in advance); the seeds.  The
  reward_model_root is Study 1's (iql_eval appends _<seed>, so models 100-110 are new).
  All run iql_eval.py, the task reward's included, as Study 1's no-terminals cells did.

  Every file carries return_range_split, which no Study 1 collector's spec has (the
  leak control's two cells, §14.82, are told apart by their seeds 1-10), and its seeds
  are 100 or 101-110.  A learned family's files are written only once its replication
  production config is SELECTED (no UNSELECTED-BASE), a cell only once its grid has a
  winner in results/repl_grid_winners.json.

Run from the repo root:
    python replication.py rule N [--write]           # record the adopted rule (once)
    python replication.py write [--write]          # every file that can be written now
    python replication.py winners [--write]         # score the finished grids (wandb)
    python replication.py --selftest
Launch with stage4_queue.sh items  repl:<file>@repl  (one sweep each; repl_sweeps/launch.sh).
"""

import argparse
import json
import os
import sys

import yaml

import phase2_sweeps as P

ROOT = P.ROOT
REPL_DIR = "repl_sweeps"
HELDOUT = 100
EVAL_SEEDS = list(range(101, 111))
SPLIT = "length"
PROTOCOLS = ("std", "cand")
ANCHORS = {"mr": "mr_sweeps/sweep_antmaze_{v}", "pt": "pt_sweeps/sweep_antmaze_{v}",
           "nmr": "nmr_sweeps/sweep_antmaze_{v}", "ens_mean": "ensemble_sweeps/sweep_antmaze_{v}_mean"}
MODEL_FAM = {"mr": "mr", "pt": "pt", "nmr": "nmr", "ens_mean": "mr"}     # the ensemble reads MR's models
METHOD = {"mr": "MR", "pt": "PT", "nmr": "NMR", "ens_mean": "ENS_MEAN"}
SEQ_LABELLER = 3
TASK_GRID = list(P.ORACLE_INDICES)                  # 0-3, as Study 1's no-terminals task grid
WINNERS = os.path.join("results", "repl_grid_winners.json")
MARK = "Lineage: REPLICATION"
RULE_FILE = os.path.join("results", "repl_rule.json")
NORMTEST_WINNERS = os.path.join("results", "normtest_grid_winners.json")
FIXED_RULES = {"N": 0, "F": 2}                      # §14.104; O3's index is the test's, per dataset
TASK_FIXED_INDEX = 1                                # r - 1 under every fixed rule (§14.108 item 2, §14.163)


def adopted_rule(root=ROOT):
    """The adopted rule's record ({"rule", "index": {variant: k} or None, ...}), or None if not yet."""
    path = os.path.join(root, RULE_FILE)
    return json.load(open(path)) if os.path.exists(path) else None


def rule_record(rule, root=ROOT):
    """The record `rule --write` stores: the fixed index per dataset (None under S)."""
    if rule in FIXED_RULES:
        index = {v: FIXED_RULES[rule] for v in P.VARIANTS}
    elif rule == "O3":
        path = os.path.join(root, NORMTEST_WINNERS)
        w = json.load(open(path)) if os.path.exists(path) else {}
        missing = [v for v in P.VARIANTS if f"o3_{v}" not in w]
        if missing:
            raise ValueError(f"O3 index missing in {NORMTEST_WINNERS} for {missing}")
        index = {v: int(w[f"o3_{v}"]) for v in P.VARIANTS}
    elif rule == "S":
        index = None
    else:
        raise ValueError(f"unknown rule {rule!r}; one of N, F, O3, S")
    return {"rule": rule, "index": index,
            "task_index": TASK_FIXED_INDEX if index is not None else None,
            "source": "the normalization test's registered reading (HANDOFF §14.104, §14.108, §14.110)"}


def _rule(rule):
    """rule argument -> record or None: "file" reads results/repl_rule.json."""
    return adopted_rule() if rule == "file" else rule


def levels(E):
    return ("full",) + tuple(E.SPARSITY) + tuple(E.NOISE_PRIMARY)


def family_fixed(sp, E, fam, v, level, prot):
    f = dict(sp[ANCHORS[fam].format(v=v)])
    if fam in ("pt", "nmr"):
        f["seq_labeller"] = SEQ_LABELLER
    if level != "full":
        f["reward_model_root"] = f"{P.MODEL_PARENT}/{E.model_base(v, MODEL_FAM[fam], level)}"
    f["return_range_split"] = SPLIT
    if prot == "cand":
        f["drop_terminals"] = True
    return f


def task_fixed(sp, v):
    return dict(sp[f"tr_sweeps/sweep_antmaze_{v}"], drop_terminals=True, return_range_split=SPLIT)


def grid_spec(sp, E, rule="file"):
    """{grid name: dict(fam, v, prot, fixed, grid)}; fam "task" for the task reward.
    Empty under a fixed rule (N, F, O3): nothing is selected at seed 100 (§14.104)."""
    rec = _rule(rule)
    out = {}
    if rec is not None and rec["index"] is not None:
        return out
    for v in P.VARIANTS:
        for fam in ANCHORS:
            for prot in PROTOCOLS:
                out[f"{REPL_DIR}/grid_{prot}_antmaze_{v}_{fam}"] = dict(
                    fam=fam, v=v, prot=prot, fixed=family_fixed(sp, E, fam, v, "full", prot),
                    grid=list(P.STAGE4_INDICES))
        out[f"{REPL_DIR}/grid_cand_antmaze_{v}_task"] = dict(
            fam="task", v=v, prot="cand", grid=TASK_GRID,
            fixed=task_fixed(sp, v))
    return out


def cell_spec(sp, E, rule="file"):
    """{cell name: dict(fam, v, level, prot, fixed, grid name or None, index or None, src)}.
    Under a fixed rule every learned cell is at the rule's index and the task reward at index 1,
    with no grid; under S (or before the rule is adopted) they take their grid's winner."""
    rec = _rule(rule)
    fixed_idx = rec["index"] if rec is not None else None
    out = {}
    for v in P.VARIANTS:
        for fam in ANCHORS:
            for prot in PROTOCOLS:
                for lv in levels(E):
                    g = None if fixed_idx else f"{REPL_DIR}/grid_{prot}_antmaze_{v}_{fam}"
                    out[f"{REPL_DIR}/{prot}_{lv}_antmaze_{v}_{fam}"] = dict(
                        fam=fam, v=v, level=lv, prot=prot, fixed=family_fixed(sp, E, fam, v, lv, prot),
                        grid=g, index=fixed_idx[v] if fixed_idx else None,
                        src=f"the adopted rule {rec['rule']} (§14.104)" if fixed_idx else None)
        out[f"{REPL_DIR}/cand_full_antmaze_{v}_task"] = dict(
            fam="task", v=v, level="full", prot="cand",
            grid=None if fixed_idx else f"{REPL_DIR}/grid_cand_antmaze_{v}_task",
            index=TASK_FIXED_INDEX if fixed_idx else None, fixed=task_fixed(sp, v),
            src=f"r - 1 under the adopted rule {rec['rule']} (§14.163)" if fixed_idx else None)
        for mode in P.TRIVIAL_MODES:
            out[f"{REPL_DIR}/cand_{mode}_antmaze_{v}"] = dict(
                fam=f"trivial_{mode}", v=v, level="full", prot="cand", grid=None, index=P.TRIVIAL_INDEX,
                fixed={"config_path": sp[f"tr_sweeps/sweep_antmaze_{v}"]["config_path"],
                       "trivial_reward": mode, "drop_terminals": True, "return_range_split": SPLIT})
    return out


def lineage(fixed, kind, index=None, grid=None):
    p = dict(fixed)
    if kind == "grid":
        p["seed"], p["normalize_reward"] = {"value": HELDOUT}, {"values": list(grid)}
    else:
        p["seed"], p["normalize_reward"] = {"values": EVAL_SEEDS}, {"value": index}
    return {k: (x if isinstance(x, dict) else {"value": x}) for k, x in p.items()}


def render(name, entry, kind, index=None):
    what = ({"task": "the task reward", "trivial_zero": "the zero reward",
             "trivial_mean": "the mean-constant reward"}.get(entry["fam"])
            or f"{METHOD[entry['fam']]} ({ANCHORS[entry['fam']].format(v=entry['v'])})")
    prot = "CANDIDATE (no terminals)" if entry["prot"] == "cand" else "STANDARD (with terminals)"
    if kind == "grid":
        lin = f"{MARK} grid, seed {HELDOUT}, normalize_reward {entry['grid']} (selection)."
    else:
        src = entry.get("src") or (f"the winner of {entry['grid']}" if entry["grid"] else "fixed in advance (§14.2)")
        lin = f"{MARK} eval, seeds 101-110, normalize_reward {index} ({src})."
    head = ["# GENERATED by replication.py -- do not hand-edit; regenerate instead",
            f"# (gp_reward-priors HANDOFF §14.83).  {lin}",
            f"# {what}, {prot}" + (f", level {entry['level']}" if kind != "grid" else ", full data") + ".",
            f"# return_range_split {SPLIT!r}: the normalization never reads the terminals (leak audit PASS)."]
    doc = {"project": P.IQL_PROJECT, "program": P.DEFAULT_PROGRAM, "method": "grid",
           "parameters": lineage(entry["fixed"], kind, index, entry.get("grid") if kind == "grid" else None)}
    return "\n".join(head) + "\n" + yaml.safe_dump(doc, sort_keys=False)


def study1_base(sp, E, entry):
    """The Study 1 params this file must equal apart from the replication's changes."""
    fam, v = entry["fam"], entry["v"]
    if fam.startswith("trivial_"):
        return P.trivial_spec(sp)[f"{P.CONTROL_DIR}/trivial_{fam[8:]}_noterm_antmaze_{v}"]
    if fam == "task":
        return dict(sp[f"tr_sweeps/sweep_antmaze_{v}"], drop_terminals=True)
    lv = entry.get("level", "full")
    f = dict(sp[ANCHORS[fam].format(v=v)]) if lv == "full" else \
        dict(P.exp_spec(sp, lv, E)[P.exp_name(ANCHORS[fam].format(v=v), lv)][1])
    if entry["prot"] == "cand":
        f["drop_terminals"] = True
    return f


def check(sp, E, name, entry, text, kind, index):
    """Problems with one rendered file (an empty list = it is what §14.83 says)."""
    doc = yaml.safe_load(text)
    got = P.flat(doc["parameters"])
    probs = []
    if doc["program"] != P.DEFAULT_PROGRAM:
        probs.append(f"program {doc['program']}")
    base = study1_base(sp, E, entry)
    diff = {k for k in set(got) | set(base) if got.get(k) != base.get(k)} - {"seed", "normalize_reward"}
    allowed = {"return_range_split"} | ({"seq_labeller"} if entry["fam"] in ("pt", "nmr") else set())
    if not diff <= allowed or got.get("return_range_split") != SPLIT:
        probs.append(f"differs from Study 1's cell in {sorted(diff)} (allowed {sorted(allowed)})")
    if entry["fam"] in ("pt", "nmr") and got.get("seq_labeller") != SEQ_LABELLER:
        probs.append(f"seq_labeller {got.get('seq_labeller')}")
    if (entry["prot"] == "cand") != (got.get("drop_terminals") is True):
        probs.append(f"drop_terminals {got.get('drop_terminals')!r} under {entry['prot']}")
    want_seed = HELDOUT if kind == "grid" else EVAL_SEEDS
    want_idx = entry["grid"] if kind == "grid" else index
    if got.get("seed") != want_seed or got.get("normalize_reward") != want_idx:
        probs.append(f"seed {got.get('seed')} index {got.get('normalize_reward')}")
    if MARK not in text:
        probs.append("no lineage mark")
    return probs


def selected(fam, v, cfg_dir=None):
    """True once the family's replication production config holds a seed-100 winner."""
    if fam not in MODEL_FAM:
        return True                                  # task / trivial: no reward model
    m = MODEL_FAM[fam]
    path = os.path.join(cfg_dir or os.path.join(ROOT, "gp_reward-priors"), f"scripts_{m}",
                        f"antmaze_{v}_{m}_antmaze_eval_repl.yaml")
    return os.path.exists(path) and "UNSELECTED-BASE" not in open(path).read()


def plan(sp, E, gwin, cfg_dir=None, rule="file"):
    """[(name, entry, kind, index or None, skip reason or None)] for every file.  Before the rule
    is adopted only the trivial cells (fixed in advance, no reward model) can be written."""
    rec = _rule(rule)
    pending = "waiting on the normalization test's rule (replication.py rule)"
    out = []
    for name, g in grid_spec(sp, E, rec).items():
        ok = selected(g["fam"], g["v"], cfg_dir)
        why = pending if rec is None else (None if ok else "replication config not selected yet")
        out.append((name, g, "grid", None, why))
    for name, c in cell_spec(sp, E, rec).items():
        idx = c["index"] if c["grid"] is None else gwin.get(c["grid"])
        why = (None if selected(c["fam"], c["v"], cfg_dir) else "replication config not selected yet")
        if rec is None and not c["fam"].startswith("trivial_"):
            why = pending
        if why is None and idx is None:
            why = "no grid winner yet"
        out.append((name, c, "eval", idx, why))
    return out


def cmd_write(do_write, root=ROOT, cfg_dir=None):
    sp, E = P.spec(), P._conditions()
    rec = adopted_rule(root)
    print(f"  rule: {'NOT ADOPTED YET (only the trivial cells)' if rec is None else rec['rule']}"
          + (f", index {rec['index']}, task reward {rec['task_index']}" if rec and rec["index"] else ""))
    path = os.path.join(root, WINNERS)
    gwin = json.load(open(path)) if os.path.exists(path) else {}
    n = skipped = bad = 0
    pl = plan(sp, E, gwin, cfg_dir, rec)
    names = {name for name, *_ in pl}
    d = os.path.join(root, REPL_DIR)
    obsolete = sorted(f"{REPL_DIR}/{f[:-5]}" for f in (os.listdir(d) if os.path.isdir(d) else [])
                      if f.endswith(".yaml") and f"{REPL_DIR}/{f[:-5]}" not in names)
    for name in obsolete:
        print(f"  OBSOLETE under rule {rec['rule'] if rec else '?'}: {name}.yaml -- not in the plan; "
              f"git rm it (never launch it)")
    for name, entry, kind, idx, why in pl:
        if why:
            skipped += 1
            continue
        text = render(name, entry, kind, idx)
        probs = check(sp, E, name, entry, text, kind, idx)
        if probs:
            bad += 1
            print(f"  !! {name}: {'; '.join(probs)}")
            continue
        fpath = os.path.join(root, name + ".yaml")
        old = open(fpath).read() if os.path.exists(fpath) else ""
        change = "unchanged" if old == text else ("CHANGED" if old else "new")
        if do_write and old != text:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            open(fpath, "w").write(text)
        n += 1
        print(f"  {name:56s} {kind}{'' if idx is None else f' idx {idx}'}  {change}"
              f"{' (written)' if do_write and old != text else ''}")
    print(f"\n  {n} file(s) {'written' if do_write else 'checked (dry run; --write to apply)'}; "
          f"{skipped} not writable yet (no rule adopted, config unselected or no grid winner); {bad} FAILED")
    return 1 if bad else 0


def find_grids(api, sp, E):
    """{grid name: [sweeps]} -- seed-100 sweeps whose params are exactly a grid's."""
    gs = grid_spec(sp, E)
    fixed_of = {n: g["fixed"] for n, g in gs.items()}
    out = {}
    for sw in P.iql_sweeps(api):
        params = sw.config.get("parameters")
        hits = P.match_entry(params, fixed_of)
        flat = P.flat(params)
        if len(hits) == 1 and flat.get("seed") == HELDOUT and flat.get("normalize_reward") == gs[hits[0]]["grid"]:
            out.setdefault(hits[0], []).append(sw)
    return out


def score_grid(sw, entry):
    """(winner, table, notes) by Study 1's stage-4 rule (iql_score.select_normalization)."""
    sys.path.insert(0, os.path.join(ROOT, "results"))
    import eval_readout as R
    import iql_score
    runs, notes = [], []
    oracle = not entry["fam"] in MODEL_FAM
    for r in sw.runs:
        cfg = r.config or {}
        bad = R.audit_run(cfg, "", HELDOUT, entry["fixed"], cfg.get("normalize_reward"), oracle)
        if bad:
            notes.append(f"run {r.id}: AUDIT {bad[0]}")
        runs.append(iql_score.score_run(r))
    if sw.config.get("program") != P.DEFAULT_PROGRAM:
        notes.append(f"program {sw.config.get('program')!r}")
    if notes:
        return None, None, notes
    win, table = iql_score.select_normalization(runs, entry["grid"])
    return int(win), table, []


def cmd_rule(rule, do_write, root=ROOT):
    """Record the adopted rule, once; refuses to change it."""
    new = rule_record(rule, root)
    old = adopted_rule(root)
    if old is not None and (old["rule"], old["index"]) != (new["rule"], new["index"]):
        sys.exit(f"!! {RULE_FILE} already records rule {old['rule']} {old['index']}: nothing written")
    print(f"  rule {new['rule']}: learned-family index {new['index'] or 'per grid winner (seed 100)'}; "
          f"task reward {new['task_index'] if new['task_index'] is not None else 'per its grid'}")
    if do_write:
        json.dump(new, open(os.path.join(root, RULE_FILE), "w"), indent=1)
        print(f"  written to {RULE_FILE}")
    return 0


def cmd_winners(do_write):
    import wandb
    sp, E = P.spec(), P._conditions()
    path = os.path.join(ROOT, WINNERS)
    gwin = json.load(open(path)) if os.path.exists(path) else {}
    found = find_grids(wandb.Api(timeout=120), sp, E)
    new = dict(gwin)
    for name, entry in grid_spec(sp, E).items():
        sws = found.get(name, [])
        if not sws:
            print(f"  {name:48s} no sweep")
            continue
        if len(sws) > 1:
            print(f"  {name:48s} {len(sws)} sweeps ({', '.join(s.id for s in sws)}) -- NOT read")
            continue
        try:
            win, table, notes = score_grid(sws[0], entry)
        except ValueError as e:
            print(f"  {name:48s} {sws[0].id}  INCOMPLETE: {e}")
            continue
        if notes:
            print(f"  {name:48s} {sws[0].id}  NOT READ -- {notes[0]}")
            continue
        by = {s["normalize_reward"]: s["score"] for s in table}
        row = " ".join(f"{by[i]:.3f}" for i in entry["grid"])
        if name in gwin and gwin[name] != win:
            sys.exit(f"!! {name}: winner changed {gwin[name]} -> {win}: nothing written")
        new[name] = win
        print(f"  {name:48s} {sws[0].id}  {row}   winner {win}")
    if do_write:
        json.dump(dict(sorted(new.items())), open(path, "w"), indent=1)
        print(f"\n  {len(new)} replication grid winner(s) in {WINNERS}")
    return 0


def selftest():
    import tempfile
    sp, E = P.spec(), P._conditions()
    S = rule_record("S")                                   # the selection arm: grids at seed 100
    gs, cs = grid_spec(sp, E, S), cell_spec(sp, E, S)
    assert len(gs) == 36 and len(cs) == 4 * 4 * 2 * 10 + 12, (len(gs), len(cs))   # 10 levels incl. full
    # every file renders to what §14.83 says, and differs from Study 1 only as allowed
    for name, g in gs.items():
        assert not check(sp, E, name, g, render(name, g, "grid"), "grid", None), name
    for name, c in cs.items():
        idx = c["index"] if c["index"] is not None else 3
        assert not check(sp, E, name, c, render(name, c, "eval", idx), "eval", idx), name
    # a std/cand pair differs only in drop_terminals; a ladder cell only in its model root
    a = P.flat(lineage(cs[f"{REPL_DIR}/std_r32_antmaze_large_play_pt"]["fixed"], "eval", 2))
    b = P.flat(lineage(cs[f"{REPL_DIR}/cand_r32_antmaze_large_play_pt"]["fixed"], "eval", 2))
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {"drop_terminals"}
    c = P.flat(lineage(cs[f"{REPL_DIR}/cand_full_antmaze_large_play_pt"]["fixed"], "eval", 2))
    assert {k for k in set(b) | set(c) if b.get(k) != c.get(k)} == {"reward_model_root"}
    assert b["reward_model_root"].endswith("/antmaze_large_play_pt_r32_eval") and b["seq_labeller"] == 3
    e = P.flat(cs[f"{REPL_DIR}/cand_n04_antmaze_medium_play_ens_mean"]["fixed"])
    assert e["mr_ensemble"] is True and e["reward_model_root"].endswith("/antmaze_medium_play_mr_n04_eval")
    assert gs[f"{REPL_DIR}/grid_cand_antmaze_medium_play_task"]["grid"] == [0, 1, 2, 3]
    # never a Study 1 cell: no Study 1 spec this repo reads has return_range_split ...
    for spec_ in (sp, P.noterm_spec(sp), P.trivial_spec(sp),
                  {n: f for n, (_, f, _) in P.control_spec(sp).items()},
                  {n: f for n, (_, f) in P.control_grid_spec(sp).items()}):
        for name, x in list(gs.items()) + list(cs.items()):
            assert not P.match_entry(lineage(x["fixed"], "grid", grid=[0]), spec_), name
    # ... except the leak control's (§14.82), which its seeds 1-10 keep apart from ours
    lf = P.leakfree_spec()
    if lf:
        hits = [n for n, x in cs.items() if P.match_entry(lineage(x["fixed"], "eval", 6),
                                                           {k: f for k, (_, f, _, _, _) in lf.items()})]
        assert hits == [f"{REPL_DIR}/cand_full_antmaze_large_diverse_mr"], hits
    # write gating: unselected configs -> only the model-free files; a grid winner -> its cell
    with tempfile.TemporaryDirectory() as d:
        for m in ("mr", "pt", "nmr"):
            os.makedirs(os.path.join(d, "cfg", f"scripts_{m}"))
            for v in P.VARIANTS:
                open(os.path.join(d, "cfg", f"scripts_{m}", f"antmaze_{v}_{m}_antmaze_eval_repl.yaml"), "w").write(
                    "# UNSELECTED-BASE\nlr: 1\n")
        pl = plan(sp, E, {}, os.path.join(d, "cfg"), S)
        ok = [n for n, _, _, _, why in pl if why is None]
        assert sorted(ok) == sorted([f"{REPL_DIR}/grid_cand_antmaze_{v}_task" for v in P.VARIANTS] +
                                    [f"{REPL_DIR}/cand_{m}_antmaze_{v}" for v in P.VARIANTS for m in ("zero", "mean")]), ok
        open(os.path.join(d, "cfg", "scripts_mr", "antmaze_large_play_mr_antmaze_eval_repl.yaml"), "w").write(
            "# Production config (REPLICATION...)\nlr: 1\n")
        gw = {f"{REPL_DIR}/grid_cand_antmaze_large_play_mr": 5, f"{REPL_DIR}/grid_cand_antmaze_large_play_task": 1}
        ok = {n: idx for n, _, _, idx, why in plan(sp, E, gw, os.path.join(d, "cfg"), S) if why is None}
        assert ok[f"{REPL_DIR}/grid_std_antmaze_large_play_mr"] is None
        assert ok[f"{REPL_DIR}/grid_cand_antmaze_large_play_ens_mean"] is None          # reads MR's models
        assert ok[f"{REPL_DIR}/cand_r16_antmaze_large_play_mr"] == 5
        assert ok[f"{REPL_DIR}/cand_full_antmaze_large_play_task"] == 1
        assert f"{REPL_DIR}/std_r16_antmaze_large_play_mr" not in ok                       # no std winner yet
        assert f"{REPL_DIR}/grid_cand_antmaze_large_play_pt" not in ok                     # PT unselected
        assert f"{REPL_DIR}/cand_r16_antmaze_large_play_ens_mean" not in ok                # no ens_mean winner
        # BEFORE the rule is adopted: only the trivial cells, even with every config selected
        for m in ("mr", "pt", "nmr"):
            for v in P.VARIANTS:
                open(os.path.join(d, "cfg", f"scripts_{m}", f"antmaze_{v}_{m}_antmaze_eval_repl.yaml"), "w").write(
                    "# Production config (REPLICATION...)\nlr: 1\n")
        ok = [n for n, _, _, _, why in plan(sp, E, gw, os.path.join(d, "cfg"), None) if why is None]
        assert sorted(ok) == sorted(f"{REPL_DIR}/cand_{m}_antmaze_{v}" for v in P.VARIANTS for m in ("zero", "mean")), ok
        # THE FIXED RULES (§14.104, §14.163): no grid; every learned cell, both protocols, at the
        # rule's index; the task reward at index 1; the trivial cells unchanged at index 0
        o3w = {f"o3_{v}": k for v, k in zip(P.VARIANTS, (0, 3, 3, 3))}
        os.makedirs(os.path.join(d, "results"))
        json.dump(o3w, open(os.path.join(d, NORMTEST_WINNERS), "w"))
        for rule, want in (("N", dict.fromkeys(P.VARIANTS, 0)), ("F", dict.fromkeys(P.VARIANTS, 2)),
                           ("O3", dict(zip(P.VARIANTS, (0, 3, 3, 3))))):
            rec = rule_record(rule, d)
            assert rec["index"] == want and rec["task_index"] == 1, (rule, rec)
            assert grid_spec(sp, E, rec) == {}, rule
            fcs = cell_spec(sp, E, rec)
            assert set(fcs) == set(cs), rule                       # the same cells as under S
            pl = plan(sp, E, {}, os.path.join(d, "cfg"), rec)      # no grid winners needed
            assert all(kind == "eval" for _, _, kind, _, _ in pl), rule
            ok = {n: idx for n, _, _, idx, why in pl if why is None}
            assert len(ok) == len(cs), (rule, len(ok))
            for n, c in fcs.items():
                k = (1 if c["fam"] == "task" else P.TRIVIAL_INDEX if c["fam"].startswith("trivial_") else want[c["v"]])
                assert ok[n] == k and c["grid"] is None, (rule, n, ok[n])
                text = render(n, c, "eval", k)
                assert not check(sp, E, n, c, text, "eval", k), (rule, n)
                assert c["fixed"] == cs[n]["fixed"], (rule, n)         # only the index differs from S
            t = render(f"{REPL_DIR}/cand_full_antmaze_medium_play_task", fcs[f"{REPL_DIR}/cand_full_antmaze_medium_play_task"], "eval", 1)
            assert f"normalize_reward 1 (r - 1 under the adopted rule {rule}" in t, t[:300]
        # the rule is recorded once and never changed
        assert cmd_rule("F", True, d) == 0 and adopted_rule(d)["rule"] == "F"
        assert cmd_rule("F", True, d) == 0                         # the same rule again: fine
        try:
            cmd_rule("N", True, d)
            raise AssertionError("rule change accepted")
        except SystemExit:
            pass
        assert adopted_rule(d)["rule"] == "F"
        try:
            rule_record("O3", os.path.join(d, "cfg"))              # no normtest winners there
            raise AssertionError("O3 without its indices accepted")
        except ValueError:
            pass
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("write", "winners", "rule"))
    ap.add_argument("rule", nargs="?", choices=("N", "F", "O3", "S"), help="for `rule`: the adopted rule")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "write":
        return cmd_write(a.write)
    if a.cmd == "winners":
        return cmd_winners(a.write)
    if a.cmd == "rule":
        if not a.rule:
            ap.error("rule needs N, F, O3 or S")
        return cmd_rule(a.rule, a.write)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
