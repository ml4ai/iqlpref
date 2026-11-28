#!/usr/bin/env bash
# clite_waiter.sh -- start C-lite (results/calibration_clite.py) as soon as ONE replication lane frees
# (handoff §14.164 as amended, §14.177).
#
# The waiter watches the nine replication lanes EA-EI (queues/repl_launch.sh, §14.176).  When the
# first one's queue script exits (its last cell has ended), it runs C-lite's four arms, one GPU
# each, on the first four of that lane's GPUs, then measures them:
#   python results/calibration_clite.py run --gpus <4 of the lane's GPUs>
# C-lite takes the freed lane's place, so the box's load does not grow.
#
# Checks first and changes NOTHING if any check fails:
#   - each of the nine lanes' queue scripts matches exactly one process;
#   - C-lite is not launched already (no waiter, no calibration_clite.py process, no clite_waiter.log);
#   - C-lite's self-test passes.
# Run from the iqlpref repo root on the box, AFTER ./queues/repl_launch.sh:
#   DRY_RUN=1 ./queues/clite_waiter.sh     # checks only
#   ./queues/clite_waiter.sh               # checks, then launches the waiter (log clite_waiter.log)
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: needs bash >= 4 (associative arrays); found $BASH_VERSION" >&2; exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f algorithms/offline/iql.py ]] || { echo "ERROR: not the iqlpref repo" >&2; exit 1; }
PY="${PY:-python}"

LANES=(EA EB EC ED EE EF EG EH EI)
declare -A GPUS=([EA]="0 1 2 3 4" [EB]="1 2 3 4 5" [EC]="0 1 3 4 5" [ED]="5 0 1 2 3" [EE]="0 1 2 4 5"
                 [EF]="2 3 4 5 0" [EG]="4 5 0 1 2" [EH]="3 4 5 0 1" [EI]="0 2 3 4 5")
# Each lane's queue script, by its GPU list and its first item; the bracket keeps this script's
# own command line (and any other waiter's) from matching.
pat() {
  local first; first=$(grep -v '^#' "queues/repl_$1.txt" | head -1)
  echo "stage4_queue[.]sh ${GPUS[$1]} 1 repl[:]${first#repl:}"
}

if [[ "${1:-}" == "--wait" ]]; then
  echo "[$(date -u +%FT%TZ)] C-lite waiter: watching EA-EI"
  free=""
  while [[ -z "$free" ]]; do
    for L in "${LANES[@]}"; do
      if ! pgrep -f "$(pat "$L")" > /dev/null; then free="$L"; break; fi
    done
    [[ -z "$free" ]] && sleep 300
  done
  read -ra g <<< "${GPUS[$free]}"
  echo "[$(date -u +%FT%TZ)] lane $free has freed; C-lite on GPUs ${g[*]:0:4}"
  exec "$PY" results/calibration_clite.py run --gpus "${g[@]:0:4}"
fi

fail=0
for L in "${LANES[@]}"; do
  n=$(pgrep -fc "$(pat "$L")")
  if [[ "$n" != 1 ]]; then echo "CHECK FAILED: lane $L: $n queue scripts match (expected 1) -- launch the replication first"; fail=1; fi
done
if pgrep -f 'clite_waiter[.]sh --wait' > /dev/null || pgrep -f 'calibration_clite[.]py' > /dev/null || [[ -e clite_waiter.log ]]; then
  echo "CHECK FAILED: C-lite looks launched already (a waiter or calibration_clite.py is running, or clite_waiter.log exists)"; fail=1
fi
if ! "$PY" results/calibration_clite.py --selftest > /dev/null; then
  echo "CHECK FAILED: results/calibration_clite.py --selftest (git pull?)"; fail=1
fi
if (( fail )); then echo "Nothing changed."; exit 1; fi
echo "Checks passed: EA-EI are each running; C-lite is not launched; its self-test passes."
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY_RUN: would launch the C-lite waiter."; exit 0; fi

nohup ./queues/clite_waiter.sh --wait > clite_waiter.log 2>&1 &
sleep 2
echo "C-lite waiter launched (log clite_waiter.log); waiters now: $(pgrep -fc 'clite_waiter[.]sh --wait') (expect 1)"
