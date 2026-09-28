#!/usr/bin/env bash
# Run Phase-2 sweeps ONE AT A TIME on a fixed GPU set, in the order given.
#
# Why this exists (gp_reward-priors HANDOFF §4.3.145): `<family>_sweeps/launch.sh all`
# round-robins agents over sweeps and each wandb agent stays on its own sweep, so
# with fewer agents than sweeps the extra sweeps get NO agent and never run.
# Launching one sweep at a time with every agent on it avoids that, and finishes
# each sweep (and so each family's stage 4) as early as possible.
#
# Run from the REPO ROOT, under nohup:
#   nohup ./stage4_queue.sh "0 1" 2 bnn:sweep_antmaze_large_play_cvar \
#         mr:sweep_antmaze_medium_play ... > stage4_queue.log 2>&1 &
#
#   GPU_LIST        space-separated GPU ids (quote it)
#   AGENTS_PER_GPU  agents per GPU; concurrency x 25 eval cores must fit
#                   alongside whatever else is on the box
#   ITEM            <family>:<sweep>, family in bnn|ensemble|mr|pt|tr
#
# DRY_RUN=1 prints the plan and checks every sweep file exists, launching nothing.
# A failed launch is logged and the queue moves on.
set -uo pipefail

if [[ ! -f algorithms/offline/iql.py ]]; then
  echo "ERROR: run this from the iqlpref repo root" >&2
  exit 1
fi
if (( $# < 3 )); then
  echo "usage: $0 \"GPU_LIST\" AGENTS_PER_GPU family:sweep [family:sweep ...]" >&2
  exit 1
fi

GPU_LIST="$1"; AGENTS_PER_GPU="$2"; shift 2

# Validate the whole queue before launching anything.
for item in "$@"; do
  fam="${item%%:*}"; sw="${item#*:}"; sw="${sw%.yaml}"
  if [[ ! -f "${fam}_sweeps/${sw}.yaml" || ! -x "${fam}_sweeps/launch.sh" ]]; then
    echo "ERROR: ${fam}_sweeps/${sw}.yaml or its launch.sh not found ($item)" >&2
    exit 1
  fi
done

n=0
for item in "$@"; do
  n=$(( n + 1 ))
  fam="${item%%:*}"; sw="${item#*:}"; sw="${sw%.yaml}"
  echo "[$(date -u +%FT%TZ)] ($n/$#) START $fam $sw  gpus=\"$GPU_LIST\" x$AGENTS_PER_GPU"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    continue
  fi
  if "./${fam}_sweeps/launch.sh" "$sw" "$GPU_LIST" "$AGENTS_PER_GPU"; then
    echo "[$(date -u +%FT%TZ)] ($n/$#) DONE  $fam $sw"
  else
    echo "[$(date -u +%FT%TZ)] ($n/$#) FAILED $fam $sw (exit $?) -- continuing" >&2
  fi
done
echo "[$(date -u +%FT%TZ)] queue finished ($# sweeps)"
