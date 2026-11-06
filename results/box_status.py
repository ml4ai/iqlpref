"""Box status from wandb: running IQL runs and their speed; sweeps active in the last day."""
import datetime as dt
import statistics
from collections import defaultdict

import wandb

api = wandb.Api(timeout=120)
P = "champlin-university-of-arizona/IQL-pref"
now = dt.datetime.now(dt.timezone.utc)
# The project-level state filter can lag the run itself: on 2026-11-05 it still returned two
# runs that had FINISHED on 23 Oct (aqszat8y seeds 7 and 8; handoff §14.90).  Each run's own
# state is authoritative, so the filter's mismatches are dropped and counted.
listed = api.runs(P, filters={"state": "running"})
runs = [r for r in listed if r.state == "running"]
index_stale = [r for r in listed if r.state != "running"]
speeds, by_sweep = [], defaultdict(list)
for r in runs:
    step = r.summary.get("_step") or 0
    rt = r.summary.get("_runtime") or 0
    sp = step / (rt / 3600) if rt > 600 else None
    hb = r.heartbeatAt
    stale = ""
    if hb:
        age = (now - dt.datetime.fromisoformat(hb.replace("Z", "+00:00")).replace(tzinfo=dt.timezone.utc)).total_seconds() / 60
        if age > 20:
            stale = f" STALE heartbeat {age:.0f} min"
    if sp:
        speeds.append(sp)
    by_sweep[r.sweep.id if r.sweep else "-"].append((r.config.get("seed"), step, sp, stale))
print(f"{len(runs)} running; median {statistics.median(speeds) / 1e3:.0f}k steps/h" if speeds else f"{len(runs)} running")
if index_stale:
    print(f"  ({len(index_stale)} run(s) listed as running by wandb's filter but {sorted({r.state for r in index_stale})} "
          f"themselves -- dropped: " + ", ".join(f"{r.sweep.id if r.sweep else '-'}/{r.id}" for r in index_stale) + ")")
for sid, lst in sorted(by_sweep.items()):
    sw = api.sweep(f"{P}/{sid}") if sid != "-" else None
    name = sw.config.get("name", "") if sw else ""
    p = sw.config.get("parameters", {}) if sw else {}
    root = (p.get("reward_model_root") or {}).get("value", "") if sw else ""
    tags = [k for k in ("drop_terminals", "gauge_reward", "trivial_reward", "heuristic_reward") if k in p]
    idx = (p.get("normalize_reward") or {})
    n_fin = sum(1 for r in sw.runs if r.state == "finished") if sw else 0
    print(f"  {sid}  {str(root).split('/')[-1] or p.get('config_path', {}).get('value', '')}  idx {idx.get('value', idx.get('values'))}  "
          f"{tags}  finished {n_fin}  running seeds "
          + ", ".join(f"{s}@{st // 1000}k{'' if v is None else f' {v / 1e3:.0f}k/h'}{x}" for s, st, v, x in sorted(lst, key=lambda t: str(t[0]))))
