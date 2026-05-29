"""per-archetype: our winrate vs a table of only that archetype.
shows where the exploit is working vs failing.
copy back: the 6 summary lines."""
import Sim
for arch in sim.ZOO:
    Sim.winrate(matches=30, total_hands=10, seed=7, mix=[arch], label=f'vs-{arch}')