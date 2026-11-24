#!/usr/bin/env bash
# normtest_stage2.sh -- queue the normalization test's stage-2 S cells (handoff §14.162).
#
# S's grid winners are in (results/normtest_grid_winners.json).  Where S's winner is index 0
# or 2, S's cell IS the N or F cell already queued; the other 18 S cells are new files.  They
# go to eight WAITERS, each behind one running lane, so the box's load does not grow (nine
# lanes, about 45 IQL runs, is the speed rule's limit):
#   DA  behind BQ (NMR's last sensitivity companion, ends first)  7 cells
#   DB  behind BW   2      DC  behind BX   1      DD  behind BY   2
#   DE  behind CA   2      DF  behind CB   2      DG  behind CC   1      DH  behind CD   1
# (none behind BZ, which has the most left).  Sized so all nine end at about the same time.
# Each waiter starts its queue on its predecessor's GPUs once the predecessor's QUEUE SCRIPT
# has exited (stage4_queue.sh blocks on each sweep, so that is when its last sweep has ended).
#
# Checks first and changes NOTHING if any check fails.
# Run from the iqlpref repo root on the box:
#   DRY_RUN=1 ./queues/normtest_stage2.sh     # checks only
#   ./queues/normtest_stage2.sh               # checks, then launches the eight waiters
set -uo pipefail
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: needs bash >= 4 (associative arrays); found $BASH_VERSION" >&2; exit 1
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f algorithms/offline/iql.py ]] || { echo "ERROR: not the iqlpref repo" >&2; exit 1; }

QUEUES=(DA DB DC DD DE DF DG DH)
declare -A PRED=([DA]=BQ [DB]=BW [DC]=BX [DD]=BY [DE]=CA [DF]=CB [DG]=CC [DH]=CD)
declare -A GPUS=([DA]="0 2 3 4 5" [DB]="0 1 2 3 4" [DC]="1 2 3 4 5" [DD]="0 1 3 4 5"
                 [DE]="0 1 2 4 5" [DF]="2 3 4 5 0" [DG]="4 5 0 1 2" [DH]="3 4 5 0 1")
# Each predecessor's queue script, matched by its GPU list and first item (normtest lanes) or
# its last item (BQ).  The bracket keeps a waiter's own command line from matching.
declare -A PAT=(
  [BQ]="stage4_queue[.]sh .*n045o[/]rescale_noterm_sweep_antmaze_medium_play"
  [BW]="stage4_queue[.]sh 0 1 2 3 4 1 norm[:]grid_cand_antmaze_large_diverse_ens_mean@"
  [BX]="stage4_queue[.]sh 1 2 3 4 5 1 norm[:]grid_cand_antmaze_large_diverse_mr@"
  [BY]="stage4_queue[.]sh 0 1 3 4 5 1 norm[:]grid_std_antmaze_medium_play_nmr@"
  [CA]="stage4_queue[.]sh 0 1 2 4 5 1 norm[:]eval_std_antmaze_medium_play_nmr_idx0@"
  [CB]="stage4_queue[.]sh 2 3 4 5 0 1 norm[:]eval_std_antmaze_medium_play_nmr_idx2@"
  [CC]="stage4_queue[.]sh 4 5 0 1 2 1 norm[:]eval_std_antmaze_medium_play_mr_idx0@"
  [CD]="stage4_queue[.]sh 3 4 5 0 1 1 norm[:]eval_std_antmaze_medium_play_mr_idx2@")

fail=0
for Q in "${QUEUES[@]}"; do
  P="${PRED[$Q]}"
  n=$(pgrep -fc "${PAT[$P]}")
  if [[ "$P" == BQ && "$n" == 0 ]]; then
    # BQ may already have finished; that is fine only if its last sweep is not running.
    if pgrep -f "launch[.]sh .*n045o[/]rescale_noterm_sweep_antmaze_medium_play" > /dev/null; then
      echo "CHECK FAILED: BQ's queue script not found but its n045o sweep is running -- re-plan DA"; fail=1
    else
      echo "note: BQ has finished; DA will start at once"
    fi
  elif [[ "$n" != 1 ]]; then
    echo "CHECK FAILED: lane $P: $n queue scripts match (expected 1)"; fail=1
  fi
  if pgrep -f "normtest_$Q[.]txt" > /dev/null || [[ -e "stage4_queue_$Q.log" ]]; then
    echo "CHECK FAILED: $Q looks launched already (a process names queues/normtest_$Q.txt, or stage4_queue_$Q.log exists)"; fail=1
  fi
  if ! DRY_RUN=1 ./stage4_queue.sh "${GPUS[$Q]}" 1 $(grep -v '^#' "queues/normtest_$Q.txt") > /dev/null; then
    echo "CHECK FAILED: queues/normtest_$Q.txt does not validate (git pull?)"; fail=1
  fi
done
if (( fail )); then echo "Nothing changed."; exit 1; fi
echo "Checks passed: each predecessor is running (or BQ has ended); the eight queue files validate."
if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "DRY_RUN: would launch waiters DA-DH."; exit 0; fi

for Q in "${QUEUES[@]}"; do
  P="${PRED[$Q]}"
  nohup bash -c "while pgrep -f '${PAT[$P]}' > /dev/null; do sleep 300; done; ./stage4_queue.sh \"${GPUS[$Q]}\" 1 \$(grep -v '^#' queues/normtest_$Q.txt)" > "stage4_queue_$Q.log" 2>&1 &
  echo "$Q waits for $P, then runs $(grep -vc '^#' "queues/normtest_$Q.txt") cell(s) on \"${GPUS[$Q]}\" (log stage4_queue_$Q.log)"
done
sleep 2
echo "Queue scripts now: $(pgrep -fc 'stage4_queue[.]sh') ; waiters: $(pgrep -fc 'while pgrep -f')"
