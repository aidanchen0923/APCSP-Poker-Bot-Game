"""DECISION 0 pre-registered A/B test.

PRE-REGISTRATION (locked before run):
- N = 200 paired matches (same seed -> same deals/opps/seat for both bots)
- new bot: chip-EV mid-game, P(1st) rollout only in last K=2 hands
- old bot: P(1st) rollout everywhere (current behavior)
- PASS criterion: paired-difference 95% CI EXCLUDES ZERO on positive side
- FAIL: CI includes zero -> spine reframe is not adopted, confront variance ceiling

runtime estimate: ~10-20 min depending on your machine
(each match runs both bots, so ~2x normal time)
copy back: everything from '=== DECISION 0 RESULT ===' down."""
import Sim
Sim.paired_ab(matches=200, base_seed=2025, total_hands=10, regime_K=2)