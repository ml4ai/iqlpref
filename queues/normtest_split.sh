#!/usr/bin/env bash
# normtest_split.sh -- split the normalization test's four lanes into eight (handoff §14.152).
#
# Lanes BS BT BU BV each run their 26 items one at a time.  This stops each lane's QUEUE
# SCRIPT (its running sweep, launch.sh and agents, finish normally) and relaunches the
# lane's not-yet-launched items as two queues:
#   continuation  BW BX BY BZ  the lane's remaining grids, then about half its cells, on the
#                              lane's own GPUs; it WAITS until the lane's running sweep ends
#   new lane      CA CB CC CD  the other cells, on a new GPU rotation, starting at once
# Nothing is launched twice: the item lists were built from W&B on 2026-11-20 (the running
# item and everything before it are left out), and this script checks first that each
# lane is still running exactly that item.  If any check fails it changes NOTHING.
#
# Run from the iqlpref repo root on the box:
#   DRY_RUN=1 ./queues/normtest_split.sh     # checks only
#   ./queues/normtest_split.sh               # checks, then stops the four queues and launches eight
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: needs bash >= 4 (associative arrays); found $BASH_VERSION" >&2; exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f algorithms/offline/iql.py ]] || { echo "ERROR: not the iqlpref repo" >&2; exit 1; }

LANES=(BS BT BU BV)
declare -A GPUS=([BS]="0 1 2 3 4" [BT]="1 2 3 4 5" [BU]="0 1 3 4 5" [BV]="5 0 1 2 3")
declare -A FIRST=([BS]=grid_std_antmaze_large_diverse_ens_mean [BT]=grid_std_antmaze_large_diverse_mr
                  [BU]=grid_std_antmaze_large_diverse_nmr [BV]=grid_std_antmaze_large_diverse_pt)
declare -A CUR=([BS]=grid_std_antmaze_medium_play_ens_mean [BT]=grid_std_antmaze_medium_play_mr
                [BU]=grid_std_antmaze_medium_diverse_nmr [BV]=grid_std_antmaze_large_play_pt)
declare -A CONT=([BS]=BW [BT]=BX [BU]=BY [BV]=BZ)
declare -A NEWL=([BS]=CA [BT]=CB [BU]=CC [BV]=CD)
declare -A NEWG=([BS]="0 1 2 4 5" [BT]="2 3 4 5 0" [BU]="4 5 0 1 2" [BV]="3 4 5 0 1")

qpat() { echo "stage4_queue[.]sh ${GPUS[$1]} 1 norm:${FIRST[$1]}@"; }      # the lane's queue script
lpat() { echo "norm_sweeps/launch[.]sh [^ ]+ ${GPUS[$1]} 1\$"; }           # the lane's running sweep

fail=0
for L in "${LANES[@]}"; do
  n=$(pgrep -fc "$(qpat "$L")")
  if [[ "$n" != 1 ]]; then echo "CHECK FAILED: lane $L: $n queue scripts match (expected 1)"; fail=1; fi
  cur=$(pgrep -af "$(lpat "$L")" | awk '{print $4}')
  if [[ "$cur" != "${CUR[$L]}" ]]; then
    echo "CHECK FAILED: lane $L is running '${cur:-nothing}', the split expects '${CUR[$L]}' -- re-plan it"; fail=1
  fi
done
for L in "${LANES[@]}"; do
  for Q in "${CONT[$L]}" "${NEWL[$L]}"; do
    if ! DRY_RUN=1 ./stage4_queue.sh "${GPUS[$L]}" 1 $(grep -v '^#' "queues/normtest_$Q.txt") > /dev/null; then
      echo "CHECK FAILED: queues/normtest_$Q.txt does not validate"; fail=1
    fi
  done
done
if (( fail )); then echo "Nothing changed."; exit 1; fi
echo "Checks passed: BS BT BU BV are each on the expected sweep; the eight queue files validate."
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY_RUN: would stop BS-BV's queue scripts and launch BW-BZ (waiting) and CA-CD."; exit 0; fi

for L in "${LANES[@]}"; do
  pkill -f "$(qpat "$L")"
done
sleep 3
for L in "${LANES[@]}"; do
  if pgrep -f "$(qpat "$L")" > /dev/null; then echo "WARNING: lane $L's queue script is still alive"; fi
  if ! pgrep -f "$(lpat "$L")" > /dev/null; then echo "NOTE: lane $L's sweep has already ended; ${CONT[$L]} starts at once"; fi
done
for L in "${LANES[@]}"; do
  C="${CONT[$L]}"; N="${NEWL[$L]}"
  nohup bash -c "while pgrep -f '$(lpat "$L")' > /dev/null; do sleep 300; done; ./stage4_queue.sh \"${GPUS[$L]}\" 1 \$(grep -v '^#' queues/normtest_$C.txt)" > "stage4_queue_$C.log" 2>&1 &
  nohup ./stage4_queue.sh "${NEWG[$L]}" 1 $(grep -v '^#' "queues/normtest_$N.txt") > "stage4_queue_$N.log" 2>&1 &
  echo "lane $L: $C waits for its sweep (log stage4_queue_$C.log); $N started on \"${NEWG[$L]}\" (stage4_queue_$N.log)"
done
sleep 2
echo "Queue scripts now: $(pgrep -fc 'stage4_queue[.]sh') ; waiters: $(pgrep -fc 'while pgrep -f')"
