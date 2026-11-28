#!/usr/bin/env bash
# repl_launch.sh -- launch the replication's IQL cells on nine lanes (handoff §14.176).
#
# Rule N was adopted by the normalization test (§14.175).  So there are no seed-100 grids; the
# 332 cells (replication.py write, rule N) run at index 0, with the task reward at index 1.
# They are dealt round-robin over nine lanes, EA-EI, in this order:
#   1. the sparsity block: full data (both protocols), the task and trivial cells, then
#      r16-r128 (both protocols);
#   2. the noise block: n01-n045 (both protocols).
# The registered reading of a ladder needs both protocols, so each block ends with every
# family's ladder complete.  Nine lanes of 5 GPUs x 1 agent is about 45 IQL runs, the speed
# rule's limit (§14.162).
#
# Checks first and changes NOTHING if any check fails:
#   - no other queue script is running (the box is meant to be idle);
#   - no lane has been launched already (a log file or a process names its queue file);
#   - all nine queue files validate (stage4_queue.sh DRY_RUN: every file exists and is in
#     the REPLICATION lineage).
# Run from the iqlpref repo root on the box:
#   DRY_RUN=1 ./queues/repl_launch.sh     # checks only
#   ./queues/repl_launch.sh               # checks, then launches EA-EI
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: needs bash >= 4 (associative arrays); found $BASH_VERSION" >&2; exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f algorithms/offline/iql.py ]] || { echo "ERROR: not the iqlpref repo" >&2; exit 1; }

LANES=(EA EB EC ED EE EF EG EH EI)
declare -A GPUS=([EA]="0 1 2 3 4" [EB]="1 2 3 4 5" [EC]="0 1 3 4 5" [ED]="5 0 1 2 3" [EE]="0 1 2 4 5"
                 [EF]="2 3 4 5 0" [EG]="4 5 0 1 2" [EH]="3 4 5 0 1" [EI]="0 2 3 4 5")

fail=0
n=$(pgrep -fc 'stage4_queue[.]sh')
if [[ "$n" != 0 ]]; then
  echo "CHECK FAILED: $n queue script(s) already running -- the box is not idle:"; pgrep -af 'stage4_queue[.]sh' | cut -c1-160
  fail=1
fi
for L in "${LANES[@]}"; do
  if pgrep -f "repl_$L[.]txt" > /dev/null || [[ -e "stage4_queue_$L.log" ]]; then
    echo "CHECK FAILED: lane $L looks launched already (a process names queues/repl_$L.txt, or stage4_queue_$L.log exists)"; fail=1
  fi
  if ! DRY_RUN=1 ./stage4_queue.sh "${GPUS[$L]}" 1 $(grep -v '^#' "queues/repl_$L.txt") > /dev/null; then
    echo "CHECK FAILED: queues/repl_$L.txt does not validate (git pull?)"; fail=1
  fi
done
if (( fail )); then echo "Nothing changed."; exit 1; fi
total=$(cat queues/repl_E?.txt | grep -vc '^#')
echo "Checks passed: the box is idle; the nine queue files ($total cells) validate."
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY_RUN: would launch EA-EI."; exit 0; fi

for L in "${LANES[@]}"; do
  nohup ./stage4_queue.sh "${GPUS[$L]}" 1 $(grep -v '^#' "queues/repl_$L.txt") > "stage4_queue_$L.log" 2>&1 &
  echo "lane $L: $(grep -vc '^#' "queues/repl_$L.txt") cells on \"${GPUS[$L]}\" (log stage4_queue_$L.log)"
done
sleep 2
echo "Queue scripts now: $(pgrep -fc 'stage4_queue[.]sh') (expect 9)"
