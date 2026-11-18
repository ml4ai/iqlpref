#!/usr/bin/env python
"""wandb_offline.py -- the W&B public API, served from the backup (handoff §14.141).

The registered readouts talk to W&B through a small part of `wandb.Api`.  OfflineApi answers
the same calls from the copy that results/wandb_backup.py writes, so the readouts run
UNCHANGED without W&B:

    import wandb_offline
    wandb_offline.install("/media/mule/scratch/loren/wandb_backup")   # wandb.Api -> OfflineApi
    ... any readout ...

What it serves (everything the readouts call; anything else raises NotImplementedError, so a
readout that needs more fails loudly instead of quietly reading less):
    Api(...)                       any arguments, ignored
    api.project(name, entity).sweeps()
    api.sweep("entity/project/id"), api.run("entity/project/id")
    api.runs("entity/project", filters={"created_at": {"$gt": t}} or None, per_page, order)
    Sweep: id, config, state, url, runs, name
    Run:   id, name, state, config, summary, sweep (.id), created_at, heartbeat_at, url,
           entity, project, group, job_type, tags, notes, lastHistoryStep,
           history(keys=None, samples=..., pandas=False), scan_history(keys=None)
Run.history with keys returns the rows that log every key, each holding _step and those
keys, as W&B's sampled endpoint does when the budget exceeds the rows (the readouts pass
100,000; a run with more rows than the budget is refused, never silently sampled).

`install(dest, block_network=True)` also makes every socket connect raise, so nothing can
reach W&B behind the backup's back.
"""

import gzip
import json
import os
import socket


def _load(path):
    with open(path) as f:
        return json.load(f)


class OfflineSweep:
    def __init__(self, api, project, sid):
        self._api, self._project = api, project
        self._d = _load(os.path.join(api.dest, project, "sweeps", f"{sid}.json"))
        self.id = sid
        self.config = self._d.get("config") or {}
        self.state = self._d.get("state")
        self.url = self._d.get("url")
        self.name = self._d.get("name")

    @property
    def runs(self):
        return [self._api._run(self._project, rid) for rid in self._d.get("runs", [])]


class _SweepRef:
    def __init__(self, sid):
        self.id = sid


class OfflineRun:
    def __init__(self, api, project, rid):
        self._api, self._project = api, project
        self._dir = os.path.join(api.dest, project, "runs", rid)
        m = _load(os.path.join(self._dir, "run.json"))
        self._m = m
        self.id = m["id"]
        self.name = m.get("name")
        self.display_name = m.get("display_name")
        self.state = m.get("state")
        self.config = m.get("config") or {}
        self.summary = m.get("summary") or {}
        self.sweep = _SweepRef(m["sweep"]) if m.get("sweep") else None
        self.created_at = m.get("created_at")
        self.heartbeat_at = m.get("heartbeat_at")
        self.url = m.get("url")
        self.entity = m.get("entity")
        self.project = m.get("project")
        self.group = m.get("group")
        self.job_type = m.get("job_type")
        self.tags = m.get("tags") or []
        self.notes = m.get("notes")
        self.lastHistoryStep = (m.get("fingerprint") or {}).get("last_history_step")

    def _rows(self):
        with gzip.open(os.path.join(self._dir, "history.jsonl.gz"), "rt") as f:
            return [json.loads(line) for line in f]

    def scan_history(self, keys=None, page_size=None, min_step=None, max_step=None):
        for r in self._rows():
            s = r.get("_step")
            if (min_step is not None and s is not None and s < min_step) or \
                    (max_step is not None and s is not None and s >= max_step):
                continue
            if keys is None:
                yield r
            elif all(r.get(k) is not None for k in keys):
                yield {k: r[k] for k in ("_step", *keys) if k in r}

    def history(self, samples=500, keys=None, x_axis="_step", pandas=True, stream="default"):
        if stream != "default" or x_axis != "_step":
            raise NotImplementedError("OfflineRun.history: only the default stream on _step is backed up")
        rows = list(self.scan_history(keys=keys))
        if len(rows) > samples:
            raise NotImplementedError(f"run {self.id}: {len(rows)} rows exceed samples={samples}; W&B would "
                                      "sample them, and the backup will not imitate that")
        if pandas:
            import pandas as pd
            return pd.DataFrame(rows)
        return rows

    def files(self):
        raise NotImplementedError("OfflineRun.files: read <run>/files/ in the backup directly")


class _Project:
    def __init__(self, api, name):
        self._api, self.name = api, name

    def sweeps(self):
        d = os.path.join(self._api.dest, self.name, "sweeps")
        ids = sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []
        return [self._api._sweep(self.name, s) for s in ids]


class OfflineApi:
    def __init__(self, dest, *args, **kwargs):
        if not os.path.isfile(os.path.join(dest, "MANIFEST.json")):
            raise FileNotFoundError(f"{dest}: no MANIFEST.json -- not a backup written by wandb_backup.py")
        self.dest = dest
        self.api_key = None
        self._runs, self._sweeps = {}, {}

    @staticmethod
    def _split(path):
        parts = path.strip("/").split("/")
        return parts[-2], parts[-1]                  # entity/project/id -> (project, id)

    def _run(self, project, rid):
        k = (project, rid)
        if k not in self._runs:
            self._runs[k] = OfflineRun(self, project, rid)
        return self._runs[k]

    def _sweep(self, project, sid):
        k = (project, sid)
        if k not in self._sweeps:
            self._sweeps[k] = OfflineSweep(self, project, sid)
        return self._sweeps[k]

    def project(self, name, entity=None):
        return _Project(self, name)

    def projects(self, entity=None, per_page=None):
        return [_Project(self, p) for p in sorted(os.listdir(self.dest))
                if os.path.isfile(os.path.join(self.dest, p, "project.json"))]

    def sweep(self, path):
        project, sid = self._split(path)
        return self._sweep(project, sid)

    def run(self, path):
        project, rid = self._split(path)
        return self._run(project, rid)

    def runs(self, path, filters=None, order="-created_at", per_page=None, include_sweeps=True, **kw):
        project = path.strip("/").split("/")[-1]
        since = None
        if filters:
            if set(filters) != {"created_at"} or set(filters["created_at"]) != {"$gt"}:
                raise NotImplementedError(f"OfflineApi.runs: filters {filters!r}")
            since = filters["created_at"]["$gt"]
        d = os.path.join(self.dest, project, "runs")
        out = [self._run(project, rid) for rid in (sorted(os.listdir(d)) if os.path.isdir(d) else [])
               if os.path.isfile(os.path.join(d, rid, "run.json"))]
        if since is not None:
            out = [r for r in out if str(r.created_at) > str(since)]
        key = (order or "-created_at").lstrip("+-")
        out.sort(key=lambda r: str(getattr(r, key, "") or ""), reverse=(order or "").startswith("-"))
        return out


def install(dest, block_network=True):
    """wandb.Api -> OfflineApi(dest); optionally refuse every network connection."""
    import wandb
    wandb.Api = lambda *a, **k: OfflineApi(dest)
    if block_network:
        def refuse(*a, **k):
            raise OSError("network blocked: this process reads the W&B backup only")
        socket.socket.connect = refuse
        socket.create_connection = refuse
    return dest
