#!/usr/bin/env python
"""wandb_backup.py -- a local, self-describing copy of the W&B results (handoff §14.137).

If W&B were lost, every hypothesis test, readout and notebook plot in this project could be
recomputed from this copy.  Run it ON THE BOX (leviathan); it writes to a physical drive:

    python results/wandb_backup.py                              # every project of the entity
    python results/wandb_backup.py --projects IQL-pref MR-training
    python results/wandb_backup.py --dry-run                    # count only, write nothing
    python results/wandb_backup.py --index-only                 # rebuild index/ from disk, no W&B

It is INCREMENTAL and RESUMABLE: a run whose fingerprint (state, heartbeat, last history step,
summary step) is unchanged since its last copy is skipped, so a rerun copies only new or
still-changing runs.  Every file is written to a temporary name and renamed, so an interrupted
backup leaves no half-written file.  It is read-only on W&B.

LAYOUT under --dest (default /media/mule/scratch/loren/wandb_backup):

    README.md                  this layout and naming scheme, written on every backup
    MANIFEST.json              the last backup: when, entity, projects, counts, errors, the
                               script's commit, the wandb version
    index/                     what each file contains (CSV, one row per thing)
      projects.csv             project, runs, sweeps, finished runs, last backup
      sweeps.csv               one row per sweep: project, sweep id, state, method, created,
                               runs, its normalization index, the iqlpref sweep file(s) it
                               was launched from (see NAMING), superseded or not, path
      runs.csv                 one row per run: project, run id, name, state, created,
                               heartbeat, sweep id, sweep file, group, job type, the config
                               fields the readouts key on (seed, dataset, normalization index,
                               terminals dropped, trajectory cut, reward model), history rows
                               and keys, path, sha256 of the history file
      history_keys.csv         project, logged key, number of runs that log it
      local_files.csv          the box-only files copied beside W&B (see --extra)
      errors.csv               runs or sweeps that failed in the last backup, with the reason
    <project>/                 one directory per W&B project, named exactly as in W&B
      project.json             the project's name, entity, and when it was backed up
      sweeps/<sweep_id>.json   the sweep: id, name, state, method, its full config (the YAML
                               it was launched with), its run ids
      runs/<run_id>/           one directory per run, named by its W&B run id
        run.json               id, name, state, created/heartbeat times, url, sweep, group,
                               job type, tags, notes, config, summary, and the fingerprint
        history.jsonl.gz       EVERY logged row, one JSON object per line, sorted by _step
                               (read with a sample budget far above the row count, which
                               returns every row; run.json "history.method" says how)
        files/                 the run's small text files (output.log, config.yaml,
                               requirements.txt, wandb-metadata.json, wandb-summary.json),
                               unless --no-files
    local/                     copies of box-only inputs the readouts need (--extra), each
                               under its path relative to the iqlpref checkout

NAMING.  W&B ids are unique within a project and never change, so they name the files; the
index maps them to what they are.  A run's sweep file is the iqlpref YAML whose `program` and
`parameters` equal the sweep's config exactly (index/sweeps.csv "sweep_files"; several when
one file was launched more than once).  When none matches exactly, "sweep_files_other_index_or_seeds"
gives the file(s) that match once normalize_reward and seed are set aside: the same file launched
at another normalization index (a seed-0 grid, a re-run at a grid winner, the oracle's index 1)
or other seeds; runs.csv marks those with "~".  A sweep with neither was launched from a file
that has since changed (e.g. the pre-fix PT sweeps, §14.24) or that lives outside the repository;
its full config is still in sweeps/<sweep_id>.json.  The file's path says what the sweep is, by the
repository's own convention, e.g. mr_sweeps/n01/noterm_sweep_antmaze_medium_play.yaml is MR,
label noise 0.1, without terminals, medium_play; results/superseded_sweeps.json marks
relaunched sweeps whose first launch is not read.

The script prints counts only, never a result (the BNN/CVaR hold, §14.1: held families' runs
are copied byte for byte like any other, and nothing of them is read or shown).
"""

import argparse
import concurrent.futures as cf
import csv
import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENTITY = "champlin-university-of-arizona"
DEST = "/media/mule/scratch/loren/wandb_backup"
# Box-only, git-ignored inputs a readout needs (copied to <dest>/local/ when present).
EXTRA = ("exp/rq1_accuracy.json", "exp/rq1_accuracy_repl.json", "exp/leak_audit.txt", "exp/goal_check.txt")
SMALL_FILES = ("output.log", "config.yaml", "requirements.txt", "wandb-metadata.json", "wandb-summary.json")
MAX_FILE_BYTES = 50 * 2 ** 20
# The history is read through W&B's sampled-history endpoint with a budget far above any
# run's row count, which returns EVERY row (iql_score.run_curve does the same).  It is about
# 500 times faster than scan_history on the IQL runs (0.4 s against 287 s for a 4,000-row
# run, 2026-11-16).  A run that fills the budget falls back to scan_history; --verify
# compares the two on a sample of runs.
SAMPLES = 1_000_000
# The config fields the readouts key on, in runs.csv (absent fields stay empty).
CONFIG_COLUMNS = ("seed", "env", "dataset", "normalize_reward", "drop_terminals", "return_range_split",
                  "reward_model_path", "reward_model_root", "reward_model_seed_offset", "checkpoints_path",
                  "group", "program")
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def _atomic(path, write):
    """write(f) into a temporary file beside path, then rename it over path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json(path, obj):
    _atomic(path, lambda f: f.write((json.dumps(obj, indent=1, sort_keys=True, default=_jsonable) + "\n").encode()))


def _jsonable(x):
    """W&B summary and config values that json cannot encode by itself."""
    if hasattr(x, "isoformat"):
        return x.isoformat()
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "_json_dict"):
        return x._json_dict
    if isinstance(x, (set, tuple)):
        return list(x)
    return repr(x)


def write_csv(path, rows, columns):
    def w(f):
        import io
        t = io.TextIOWrapper(f, encoding="utf-8", newline="")
        cw = csv.DictWriter(t, fieldnames=columns, extrasaction="ignore")
        cw.writeheader()
        for r in rows:
            cw.writerow({k: ("" if r.get(k) is None else r[k]) for k in columns})
        t.flush()
        t.detach()
    _atomic(path, w)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------
def retry(fn, *a, tries=4, **k):
    """W&B reads time out or reset now and then; back off and try again."""
    for i in range(tries):
        try:
            return fn(*a, **k)
        except Exception as e:                       # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(15 * (i + 1))


def _plain(d):
    return {k: v for k, v in dict(d or {}).items()}


def fingerprint(run):
    """What changes while a run is alive; unchanged = nothing new to copy."""
    s = _plain(run.summary)
    return {"state": run.state, "heartbeat_at": getattr(run, "heartbeat_at", None),
            "last_history_step": getattr(run, "lastHistoryStep", None), "summary_step": s.get("_step")}


def history_rows(run):
    """(every logged row, how it was read)."""
    rows = retry(run.history, samples=SAMPLES, pandas=False)
    if len(rows) < SAMPLES:
        return list(rows), f"history(samples={SAMPLES})"
    return list(retry(lambda: list(run.scan_history(page_size=1000)))), "scan_history"


def verify_history(run):
    """None if the sampled history equals scan_history row for row, else the difference."""
    a = sorted(json.dumps(r, sort_keys=True, default=_jsonable) for r in history_rows(run)[0])
    b = sorted(json.dumps(r, sort_keys=True, default=_jsonable) for r in retry(lambda: list(run.scan_history(page_size=1000))))
    return None if a == b else f"sampled {len(a)} rows, scan {len(b)} rows, {len(set(a) ^ set(b))} differ"


def backup_run(run, project_dir, files):
    """Copy one run.  Returns (status, info) with status 'copied' or 'skipped'."""
    rdir = os.path.join(project_dir, "runs", run.id)
    meta_path = os.path.join(rdir, "run.json")
    fp = fingerprint(run)
    if os.path.exists(meta_path) and os.path.exists(os.path.join(rdir, "history.jsonl.gz")):
        try:
            old = json.load(open(meta_path))
            if old.get("fingerprint") == json.loads(json.dumps(fp, default=_jsonable)):
                return "skipped", old
        except (OSError, ValueError):
            pass
    rows, method = history_rows(run)
    rows.sort(key=lambda r: (r.get("_step") is None, r.get("_step")))
    keys = sorted({k for r in rows for k in r})

    def w(f):
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
            for r in rows:
                g.write((json.dumps(r, sort_keys=True, default=_jsonable) + "\n").encode())
    hist = os.path.join(rdir, "history.jsonl.gz")
    _atomic(hist, w)
    copied_files = []
    if files:
        for fobj in retry(lambda: list(run.files())):
            if os.path.basename(fobj.name) in SMALL_FILES and (fobj.size or 0) <= MAX_FILE_BYTES:
                try:
                    tmpd = tempfile.mkdtemp(dir=rdir, prefix=".dl_")
                    retry(fobj.download, root=tmpd, replace=True)
                    os.makedirs(os.path.join(rdir, "files"), exist_ok=True)
                    os.replace(os.path.join(tmpd, fobj.name), os.path.join(rdir, "files", os.path.basename(fobj.name)))
                    copied_files.append(os.path.basename(fobj.name))
                except Exception:                    # noqa: BLE001  a missing log is not fatal
                    pass
                finally:
                    shutil.rmtree(tmpd, ignore_errors=True)
    meta = {"id": run.id, "name": run.name, "display_name": getattr(run, "display_name", None),
            "state": run.state, "created_at": run.created_at, "heartbeat_at": getattr(run, "heartbeat_at", None),
            "url": run.url, "entity": run.entity, "project": run.project,
            "sweep": run.sweep.id if run.sweep else None, "group": run.group, "job_type": run.job_type,
            "tags": list(run.tags or []), "notes": run.notes,
            "config": {k: v for k, v in _plain(run.config).items() if not k.startswith("_")},
            "summary": _plain(run.summary), "fingerprint": fp,
            "history": {"file": "history.jsonl.gz", "rows": len(rows), "keys": keys, "sha256": sha256(hist),
                        "method": method},
            "files": copied_files,
            "backed_up_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    write_json(meta_path, meta)
    return "copied", json.loads(json.dumps(meta, default=_jsonable))


def backup_sweep(sw, project_dir, run_ids):
    """run_ids: the sweep's runs, from the project's run listing (one call per sweep saved)."""
    cfg = _plain(sw.config)
    obj = {"id": sw.id, "name": getattr(sw, "name", None) or cfg.get("name"), "state": sw.state,
           "method": cfg.get("method"), "created_at": getattr(sw, "created_at", None), "config": cfg,
           "runs": sorted(run_ids), "url": sw.url,
           "backed_up_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    write_json(os.path.join(project_dir, "sweeps", f"{sw.id}.json"), obj)
    return obj


# --------------------------------------------------------------------------
# The index (built from disk only, so it can be rebuilt without W&B)
# --------------------------------------------------------------------------
def _num(x):
    """W&B stores a YAML 0.0 as 0: compare integral floats as ints (booleans untouched)."""
    if isinstance(x, dict):
        return {k: _num(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_num(v) for v in x]
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return x


def _canon(d):
    return json.dumps(_num(d), sort_keys=True, default=str)


LOOSE = ("normalize_reward", "seed")       # what a relaunch of one file changes (another index, other seeds)


def _loose(params):
    return {k: v for k, v in (params or {}).items() if k not in LOOSE}


def repo_sweep_files(repo):
    """({canonical (program, parameters): [paths]}, the same with LOOSE parameters dropped) for
    every sweep YAML in the iqlpref checkout."""
    import yaml
    out, loose = {}, {}
    if not repo or not os.path.isdir(repo):
        return out, loose
    for dirpath, dirs, names in os.walk(repo):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("wandb", "exp", "node_modules")]
        for n in names:
            if not n.endswith((".yaml", ".yml")):
                continue
            p = os.path.join(dirpath, n)
            try:
                y = yaml.safe_load(open(p))
            except Exception:                        # noqa: BLE001
                continue
            if isinstance(y, dict) and isinstance(y.get("parameters"), dict):
                rel = os.path.relpath(p, repo)
                out.setdefault(_canon((y.get("program"), y["parameters"])), []).append(rel)
                loose.setdefault(_canon((y.get("program"), _loose(y["parameters"]))), []).append(rel)
    return out, loose


def build_index(dest, repo, errors=()):
    files_of, loose_of = repo_sweep_files(repo)
    superseded = set()
    sp = os.path.join(repo or "", "results", "superseded_sweeps.json")
    if repo and os.path.exists(sp):
        try:
            s = json.load(open(sp))
            superseded = set(s if isinstance(s, list) else s.keys())
        except (OSError, ValueError):
            pass
    projects, sweeps, runs, keys = [], [], [], {}
    for proj in sorted(d for d in os.listdir(dest) if os.path.isfile(os.path.join(dest, d, "project.json"))):
        pdir = os.path.join(dest, proj)
        sweep_file = {}
        for fn in sorted(os.listdir(os.path.join(pdir, "sweeps")) if os.path.isdir(os.path.join(pdir, "sweeps")) else []):
            s = json.load(open(os.path.join(pdir, "sweeps", fn)))
            c = s.get("config") or {}
            f = files_of.get(_canon((c.get("program"), c.get("parameters"))), [])
            fl = [] if f else loose_of.get(_canon((c.get("program"), _loose(c.get("parameters")))), [])
            sweep_file[s["id"]] = ";".join(f) or ";".join(f"~{x}" for x in fl)
            sweeps.append({"project": proj, "sweep_id": s["id"], "name": s.get("name"), "state": s.get("state"),
                           "method": s.get("method"), "created_at": s.get("created_at"), "runs": len(s.get("runs", [])),
                           "sweep_files": ";".join(f), "sweep_files_other_index_or_seeds": ";".join(fl),
                           "index": json.dumps((c.get("parameters") or {}).get("normalize_reward")),
                           "superseded": s["id"] in superseded,
                           "path": os.path.relpath(os.path.join(pdir, "sweeps", fn), dest)})
        n_runs = n_fin = 0
        rdir = os.path.join(pdir, "runs")
        for rid in sorted(os.listdir(rdir) if os.path.isdir(rdir) else []):
            mp = os.path.join(rdir, rid, "run.json")
            if not os.path.exists(mp):
                continue
            m = json.load(open(mp))
            n_runs += 1
            n_fin += m.get("state") == "finished"
            h = m.get("history") or {}
            for k in h.get("keys", []):
                keys[(proj, k)] = keys.get((proj, k), 0) + 1
            cfg = m.get("config") or {}
            row = {"project": proj, "run_id": m["id"], "name": m.get("name"), "state": m.get("state"),
                   "created_at": m.get("created_at"), "heartbeat_at": m.get("heartbeat_at"), "sweep_id": m.get("sweep"),
                   "sweep_files": sweep_file.get(m.get("sweep"), ""), "group": m.get("group"),
                   "job_type": m.get("job_type"), "history_rows": h.get("rows"), "history_keys": len(h.get("keys", [])),
                   "history_sha256": h.get("sha256"), "files": ";".join(m.get("files", [])),
                   "path": os.path.relpath(os.path.join(rdir, rid), dest), "backed_up_utc": m.get("backed_up_utc")}
            for c in CONFIG_COLUMNS:
                v = cfg.get(c)
                row[f"cfg_{c}"] = json.dumps(v) if isinstance(v, (list, dict)) else v
            runs.append(row)
        pj = json.load(open(os.path.join(pdir, "project.json")))
        projects.append({"project": proj, "runs": n_runs, "finished_runs": n_fin,
                         "sweeps": sum(1 for s in sweeps if s["project"] == proj), "backed_up_utc": pj.get("backed_up_utc")})
    idx = os.path.join(dest, "index")
    write_csv(os.path.join(idx, "projects.csv"), projects, ["project", "runs", "finished_runs", "sweeps", "backed_up_utc"])
    write_csv(os.path.join(idx, "sweeps.csv"), sweeps, ["project", "sweep_id", "name", "state", "method", "created_at",
                                                        "runs", "index", "sweep_files", "sweep_files_other_index_or_seeds",
                                                        "superseded", "path"])
    write_csv(os.path.join(idx, "runs.csv"), runs,
              ["project", "run_id", "name", "state", "created_at", "heartbeat_at", "sweep_id", "sweep_files", "group",
               "job_type"] + [f"cfg_{c}" for c in CONFIG_COLUMNS] +
              ["history_rows", "history_keys", "history_sha256", "files", "path", "backed_up_utc"])
    write_csv(os.path.join(idx, "history_keys.csv"),
              [{"project": p, "key": k, "runs": n} for (p, k), n in sorted(keys.items())], ["project", "key", "runs"])
    if errors:
        write_csv(os.path.join(idx, "errors.csv"), list(errors), ["project", "what", "id", "error"])
    elif os.path.exists(os.path.join(idx, "errors.csv")):
        os.remove(os.path.join(idx, "errors.csv"))
    return {"projects": len(projects), "sweeps": len(sweeps), "runs": len(runs)}


def copy_extra(dest, repo, extra):
    rows = []
    for rel in extra:
        src = os.path.join(repo, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dest, "local", rel)
        _atomic(dst, lambda f, s=src: f.write(open(s, "rb").read()))
        st = os.stat(src)
        rows.append({"file": rel, "bytes": st.st_size,
                     "modified_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(timespec="seconds"),
                     "sha256": sha256(dst), "path": os.path.relpath(dst, dest)})
    if rows:
        write_csv(os.path.join(dest, "index", "local_files.csv"), rows, ["file", "bytes", "modified_utc", "sha256", "path"])
    return rows


def write_readme(dest):
    doc = __doc__.split("LAYOUT under --dest", 1)[1]
    text = ("# W&B backup\n\nWritten by iqlpref `results/wandb_backup.py` (handoff §14.137); "
            "regenerated on every backup.\n\nLAYOUT" + doc.split("\n", 1)[1].split("The script prints counts", 1)[0] +
            "\n## Reading it back\n\n```python\nimport gzip, json\n"
            "meta = json.load(open('<project>/runs/<run_id>/run.json'))          # config, summary, state\n"
            "rows = [json.loads(l) for l in gzip.open('<project>/runs/<run_id>/history.jsonl.gz', 'rt')]\n"
            "curve = [(r['_step'], r[k]) for r in rows if r.get(k) is not None]    # k: a logged key\n```\n\n"
            "Use `index/runs.csv` to find runs (filter by project, sweep, sweep file or config "
            "columns) and `index/history_keys.csv` to see what each project logs.\n")
    _atomic(os.path.join(dest, "README.md"), lambda f: f.write(text.encode()))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEST)
    ap.add_argument("--entity", default=ENTITY)
    ap.add_argument("--projects", nargs="*", help="W&B projects (default: every project of the entity)")
    ap.add_argument("--repo", default=ROOT, help="the iqlpref checkout, for sweep-file names and --extra")
    ap.add_argument("--extra", nargs="*", default=list(EXTRA), help="box-only files to copy, relative to --repo")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-files", action="store_true", help="skip the runs' small text files (output.log, ...)")
    ap.add_argument("--limit", type=int, help="at most this many runs per project (for testing)")
    ap.add_argument("--verify", type=int, default=0, metavar="K",
                    help="per project, check K finished runs' sampled history against scan_history (slow: "
                         "minutes per IQL run)")
    ap.add_argument("--dry-run", action="store_true", help="count projects, sweeps and runs; write nothing")
    ap.add_argument("--index-only", action="store_true", help="rebuild index/ and README.md from disk; no W&B")
    a = ap.parse_args()
    if a.index_only:
        write_readme(a.dest)
        print("index:", build_index(a.dest, a.repo))
        return 0
    import wandb
    api = wandb.Api(timeout=120)
    projects = a.projects or sorted(p.name for p in retry(lambda: list(api.projects(a.entity))))
    if not a.dry_run:
        os.makedirs(a.dest, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    errors, counts = [], {}
    for proj in projects:
        pdir = os.path.join(a.dest, proj)
        sweeps = retry(lambda: list(api.project(proj, entity=a.entity).sweeps()))
        runs = retry(lambda: list(api.runs(f"{a.entity}/{proj}", per_page=500)))
        if a.limit:
            runs = runs[:a.limit]
        if a.dry_run:
            print(f"{proj:40s} sweeps {len(sweeps):5d}  runs {len(runs):6d}")
            continue
        write_json(os.path.join(pdir, "project.json"), {"entity": a.entity, "project": proj, "backed_up_utc": started})
        of_sweep = {}
        for r in runs:
            if r.sweep:
                of_sweep.setdefault(r.sweep.id, []).append(r.id)
        for sw in sweeps:
            try:
                backup_sweep(sw, pdir, of_sweep.get(sw.id, []))
            except Exception as e:                   # noqa: BLE001
                errors.append({"project": proj, "what": "sweep", "id": sw.id, "error": f"{type(e).__name__}: {e}"})
        c = {"copied": 0, "skipped": 0, "failed": 0}
        done = 0

        def one(run):
            try:
                st, _ = backup_run(run, pdir, not a.no_files)
                return st, None
            except Exception as e:                   # noqa: BLE001
                return "failed", {"project": proj, "what": "run", "id": run.id, "error": f"{type(e).__name__}: {e}"}
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for st, err in ex.map(one, runs):
                c[st] += 1
                done += 1
                if err:
                    errors.append(err)
                if done % 500 == 0:
                    print(f"  {proj}: {done}/{len(runs)} runs ({c['copied']} copied, {c['skipped']} unchanged, "
                          f"{c['failed']} failed)", flush=True)
        counts[proj] = dict(c, sweeps=len(sweeps), runs=len(runs))
        if a.verify:
            fin = [r for r in runs if r.state == "finished"][:a.verify]
            bad = [(r.id, verify_history(r)) for r in fin]
            bad = [(i, d) for i, d in bad if d]
            counts[proj]["verified"] = {"runs": len(fin), "mismatches": len(bad)}
            errors += [{"project": proj, "what": "verify", "id": i, "error": d} for i, d in bad]
            print(f"  {proj}: history verified on {len(fin)} run(s), {len(bad)} mismatch(es)", flush=True)
        print(f"{proj:40s} sweeps {len(sweeps):5d}  runs {len(runs):6d}  copied {c['copied']}  unchanged "
              f"{c['skipped']}  failed {c['failed']}", flush=True)
    if a.dry_run:
        return 0
    extra = copy_extra(a.dest, a.repo, a.extra)
    write_readme(a.dest)
    idx = build_index(a.dest, a.repo, errors)
    commit = subprocess.run(["git", "-C", a.repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    write_json(os.path.join(a.dest, "MANIFEST.json"), {
        "started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "entity": a.entity, "projects": counts, "errors": len(errors), "local_files": [r["file"] for r in extra],
        "index": idx, "script": "results/wandb_backup.py", "iqlpref_commit": commit, "wandb_version": wandb.__version__})
    print(f"done: {idx['runs']} runs, {idx['sweeps']} sweeps in {idx['projects']} projects; {len(errors)} error(s)"
          + (" -- see index/errors.csv; rerun to retry them" if errors else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
