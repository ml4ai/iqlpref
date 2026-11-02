#!/usr/bin/env bash
# Launch repl_sweeps (the replication, gp_reward-priors HANDOFF 14.83) as W&B grid sweeps.
#
# Run from the REPO ROOT.
#
# Usage:
#   ./repl_sweeps/launch.sh [SWEEP] [GPU_LIST] [AGENTS_PER_GPU]
#
#   SWEEP           a single replication sweep (handoff 14.83; no "all"): full path,
#                   basename with/without .yaml.
#   GPU_LIST        space-separated GPU ids (quote it).      Default: "0 1 2 3 4 5"
#   AGENTS_PER_GPU  wandb agents to launch on each GPU.      Default: 1
#
# Concurrency = (#GPUs) * AGENTS_PER_GPU. Each concurrent run's eval uses
# n_envs=25 CPU workers, so the script warns if concurrency*25 exceeds
# TOTAL_CORES (default 255). Agent slots are round-robined across the sweeps.
#
# Examples:
#   ./repl_sweeps/launch.sh grid_cand_antmaze_medium_play_task "0 1" 2   # one sweep, 4 concurrent
#   Normally through the queue:  ./stage4_queue.sh "0 1 2 3 4" 1 repl:grid_cand_antmaze_medium_play_task@repl
set -euo pipefail

if [[ ! -d repl_sweeps || ! -f algorithms/offline/iql.py ]]; then
  echo "ERROR: run this from the iqlpref repo root:  ./repl_sweeps/launch.sh" >&2
  exit 1
fi

SWEEP_ARG="${1:-all}"
GPU_LIST="${2:-0 1 2 3 4 5}"
AGENTS_PER_GPU="${3:-1}"
TOTAL_CORES="${TOTAL_CORES:-255}"
CPU_PER_RUN=25
LOGDIR="repl_sweeps/logs"; mkdir -p "$LOGDIR"

# --- resolve sweep yaml(s) ---
SWEEP_YAMLS=()
if [[ "$SWEEP_ARG" == "all" ]]; then
  echo "ERROR: repl_sweeps has no 'all': queue each file with stage4_queue.sh (repl:<file>@repl)" >&2
  exit 1
else
  y="$SWEEP_ARG"
  [[ "$y" == *.yaml ]] || y="${y}.yaml"
  [[ "$y" == */* ]] || y="repl_sweeps/${y}"
  if [[ ! -f "$y" ]]; then
    echo "ERROR: sweep yaml not found: $y" >&2
    exit 1
  fi
  SWEEP_YAMLS=("$y")
fi

# --- concurrency / CPU check ---
read -ra GPUS <<< "$GPU_LIST"
CONCURRENCY=$(( ${#GPUS[@]} * AGENTS_PER_GPU ))
CORES_USED=$(( CONCURRENCY * CPU_PER_RUN ))
echo "Sweeps:      ${#SWEEP_YAMLS[@]}   GPUs: ${GPUS[*]}   agents/GPU: $AGENTS_PER_GPU"
echo "Concurrency: $CONCURRENCY runs  ->  ~$CORES_USED / $TOTAL_CORES CPU cores at eval"
if (( CORES_USED > TOTAL_CORES )); then
  echo "WARNING: $CONCURRENCY concurrent evals need $CORES_USED cores (> $TOTAL_CORES);" >&2
  echo "         CPUs will be oversubscribed when evals sync. Lower AGENTS_PER_GPU or #GPUs." >&2
fi

# --- create the sweeps, capture their ids (entity/project/id) ---
SWEEP_IDS=()
for y in "${SWEEP_YAMLS[@]}"; do
  id=$(wandb sweep "$y" 2>&1 | tee /dev/stderr \
        | grep -oE 'wandb agent [^[:space:]]+' | tail -1 | awk '{print $3}')
  if [[ -z "${id:-}" ]]; then
    echo "ERROR: failed to parse sweep id for $y" >&2
    exit 1
  fi
  SWEEP_IDS+=("$id")
done

echo "Created sweeps:"; printf '  %s\n' "${SWEEP_IDS[@]}"

# --- launch agents: AGENTS_PER_GPU per GPU, round-robin sweeps across slots ---
slot=0
for gpu in "${GPUS[@]}"; do
  for ((a=0; a<AGENTS_PER_GPU; a++)); do
    id="${SWEEP_IDS[$(( slot % ${#SWEEP_IDS[@]} ))]}"
    logf="$LOGDIR/agent_${id##*/}_g${gpu}_a${a}.log"
    echo "GPU $gpu  <-  $id  ($logf)"
    CUDA_VISIBLE_DEVICES=$gpu nohup wandb agent "$id" > "$logf" 2>&1 &
    slot=$(( slot + 1 ))
  done
done

echo "Launched $slot agents."
echo "Monitor:  watch -n5 nvidia-smi   |   tail -f $LOGDIR/*.log"
wait
