# Held: outputs covered by the BNN/CVaR hold

Moved here on 2026-10-21 under the BNN/CVaR hold (`gp_reward-priors/HANDOFF_HP_SELECTION.md`
§14.1, decided by the author on 2026-10-20). These are generated readout outputs that print
BNN CVaR, BNN mean or ensemble-CVaR results. They are out of the reporting path and are
not quoted anywhere until the author lifts the hold. Nothing was deleted, and raw runs,
wandb, caches and chains are where they were.

| file | from | covered content |
|---|---|---|
| `eval_readout_latest.txt` | `results/` | the §12 readout: BNN CVaR's eight verdicts, every anchor cell |
| `control_steps_latest.txt` | `results/` | steps-to-goal, MR **and** BNN CVaR (mixed; a filtered run of `control_steps.py` regenerates the MR part) |

The inner repo's outputs are stashed in `gp_reward-priors/calibration_results/held_bnn_cvar/`.
