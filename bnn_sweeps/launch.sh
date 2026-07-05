#!/usr/bin/env bash
# Launch all 4 bnn_sweeps as W&B grid sweeps across 6 GPUs.
#
# Run from the REPO ROOT (paths below are relative):
#     ./bnn_sweeps/launch.sh
#
# Bottleneck is CPU, not GPU: each run trains a tiny IQL MLP on 1 GPU but each
# eval (every eval_freq=5000 steps, 100 episodes) uses n_envs=25 CPU workers.
# 255 cores / 25 = ~10 concurrent runs max. AGENTS_PER_SWEEP=2 -> 8 concurrent
# (200 cores, comfortable). Bump toward 10 only if CPU headroom looks fine.
set -euo pipefail

# --- must run from repo root so the relative sweep paths resolve ---
if [[ ! -d bnn_sweeps || ! -f algorithms/offline/iql.py ]]; then
  echo "ERROR: run this from the iqlpref repo root:  ./bnn_sweeps/launch.sh" >&2
  exit 1
fi

AGENTS_PER_SWEEP=2          # 2 -> 8 concurrent (200 CPU cores). Set 3 on two sweeps for 10.
NGPU=6
LOGDIR="bnn_sweeps/logs"; mkdir -p "$LOGDIR"

SWEEP_YAMLS=(
  bnn_sweeps/sweep_antmaze_medium_play.yaml
  bnn_sweeps/sweep_antmaze_medium_diverse.yaml
  bnn_sweeps/sweep_antmaze_large_play.yaml
  bnn_sweeps/sweep_antmaze_large_diverse.yaml
)

# --- create the 4 sweeps, capture their IDs (entity/project/id) ---
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

# --- launch the agent pool, round-robin GPU assignment ---
slot=0
for id in "${SWEEP_IDS[@]}"; do
  for ((a=0; a<AGENTS_PER_SWEEP; a++)); do
    gpu=$(( slot % NGPU ))
    logf="$LOGDIR/agent_${id//\//_}_${a}.log"
    echo "GPU $gpu  <-  $id  ($logf)"
    CUDA_VISIBLE_DEVICES=$gpu nohup wandb agent "$id" > "$logf" 2>&1 &
    slot=$(( slot + 1 ))
  done
done

echo "Launched $slot agents."
echo "Monitor:  watch -n5 nvidia-smi   |   tail -f $LOGDIR/*.log"
wait
