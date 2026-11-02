#!/usr/bin/env python
"""replication.py -- the replication's IQL sweep files (gp_reward-priors HANDOFF §14.83).

The replication re-runs §14.5's main hypothesis on NEW reward models (selected and trained at
seeds 100-110, gp_reward-priors/scripts_bnn/make_repl_selection.py) under the CORRECTED
normalization: return_range_split "length" at every index, in BOTH protocols, so the
normalized reward never depends on the terminals (results/leak_measure.py --audit: PASS).
Study 1's files, specs and readers are untouched; every file here lives in repl_sweeps/.

  GRIDS  (seed 100, full data; Study 1's stage-4 rule picks the index)
    repl_sweeps/grid_{std,cand}_antmaze_<v>_<fam>   fam in mr, pt, nmr, ens_mean: indices 0-7
    repl_sweeps/grid_cand_antmaze_<v>_task          the task reward, candidate: indices 0-3
  CELLS  (seeds 101-110, at the grid winner; nothing re-selected under a condition)
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


def grid_spec(sp, E):
    """{grid name: dict(fam, v, prot, fixed, grid)}; fam "task" for the task reward."""
    out = {}
    for v in P.VARIANTS:
        for fam in ANCHORS:
            for prot in PROTOCOLS:
                out[f"{REPL_DIR}/grid_{prot}_antmaze_{v}_{fam}"] = dict(
                    fam=fam, v=v, prot=prot, fixed=family_fixed(sp, E, fam, v, "full", prot),
                    grid=list(P.STAGE4_INDICES))
        out[f"{REPL_DIR}/grid_cand_antmaze_{v}_task"] = dict(
            fam="task", v=v, prot="cand", grid=TASK_GRID,
            fixed=dict(sp[f"tr_sweeps/sweep_antmaze_{v}"], drop_terminals=True, return_range_split=SPLIT))
    return out


def cell_spec(sp, E):
    """{cell name: dict(fam, v, level, prot, fixed, grid name or None, index or None)}."""
    out = {}
    for v in P.VARIANTS:
        for fam in ANCHORS:
            for prot in PROTOCOLS:
                for lv in levels(E):
                    out[f"{REPL_DIR}/{prot}_{lv}_antmaze_{v}_{fam}"] = dict(
                        fam=fam, v=v, level=lv, prot=prot, fixed=family_fixed(sp, E, fam, v, lv, prot),
                        grid=f"{REPL_DIR}/grid_{prot}_antmaze_{v}_{fam}", index=None)
        out[f"{REPL_DIR}/cand_full_antmaze_{v}_task"] = dict(
            fam="task", v=v, level="full", prot="cand", grid=f"{REPL_DIR}/grid_cand_antmaze_{v}_task",
            index=None, fixed=grid_spec(sp, E)[f"{REPL_DIR}/grid_cand_antmaze_{v}_task"]["fixed"])
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
        src = (f"the winner of {entry['grid']}" if entry["grid"] else "fixed in advance (§14.2)")
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


def plan(sp, E, gwin, cfg_dir=None):
    """[(name, entry, kind, index or None, skip reason or None)] for every file."""
    out = []
    for name, g in grid_spec(sp, E).items():
        ok = selected(g["fam"], g["v"], cfg_dir)
        out.append((name, g, "grid", None, None if ok else "replication config not selected yet"))
    for name, c in cell_spec(sp, E).items():
        idx = c["index"] if c["grid"] is None else gwin.get(c["grid"])
        why = (None if selected(c["fam"], c["v"], cfg_dir) else "replication config not selected yet")
        if why is None and idx is None:
            why = "no grid winner yet"
        out.append((name, c, "eval", idx, why))
    return out


def cmd_write(do_write, root=ROOT, cfg_dir=None):
    sp, E = P.spec(), P._conditions()
    path = os.path.join(root, WINNERS)
    gwin = json.load(open(path)) if os.path.exists(path) else {}
    n = skipped = bad = 0
    for name, entry, kind, idx, why in plan(sp, E, gwin, cfg_dir):
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
          f"{skipped} not writable yet (config unselected or no grid winner); {bad} FAILED")
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
    gs, cs = grid_spec(sp, E), cell_spec(sp, E)
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
        pl = plan(sp, E, {}, os.path.join(d, "cfg"))
        ok = [n for n, _, _, _, why in pl if why is None]
        assert sorted(ok) == sorted([f"{REPL_DIR}/grid_cand_antmaze_{v}_task" for v in P.VARIANTS] +
                                    [f"{REPL_DIR}/cand_{m}_antmaze_{v}" for v in P.VARIANTS for m in ("zero", "mean")]), ok
        open(os.path.join(d, "cfg", "scripts_mr", "antmaze_large_play_mr_antmaze_eval_repl.yaml"), "w").write(
            "# Production config (REPLICATION...)\nlr: 1\n")
        gw = {f"{REPL_DIR}/grid_cand_antmaze_large_play_mr": 5, f"{REPL_DIR}/grid_cand_antmaze_large_play_task": 1}
        ok = {n: idx for n, _, _, idx, why in plan(sp, E, gw, os.path.join(d, "cfg")) if why is None}
        assert ok[f"{REPL_DIR}/grid_std_antmaze_large_play_mr"] is None
        assert ok[f"{REPL_DIR}/grid_cand_antmaze_large_play_ens_mean"] is None          # reads MR's models
        assert ok[f"{REPL_DIR}/cand_r16_antmaze_large_play_mr"] == 5
        assert ok[f"{REPL_DIR}/cand_full_antmaze_large_play_task"] == 1
        assert f"{REPL_DIR}/std_r16_antmaze_large_play_mr" not in ok                       # no std winner yet
        assert f"{REPL_DIR}/grid_cand_antmaze_large_play_pt" not in ok                     # PT unselected
        assert f"{REPL_DIR}/cand_r16_antmaze_large_play_ens_mean" not in ok                # no ens_mean winner
    print("selftest OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("write", "winners"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "write":
        return cmd_write(a.write)
    if a.cmd == "winners":
        return cmd_winners(a.write)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
