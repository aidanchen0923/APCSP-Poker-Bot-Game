"""horizon sweep: does winrate degrade as match length grows?
tests dilution hypothesis (long matches -> rollout tail dominates -> baseline drift).
copy back: the 4 summary lines."""
import Sim
for th in [3, 8, 12, 20]:
    Sim.winrate(matches=40, total_hands=th, seed=11, label=f'h{th}')