"""confirm horizon-10 winrate at N=120 across two seeds.
expected runtime: ~5-10 min depending on machine.
copy back: the last 3 lines of output."""
import Sim
Sim.confirm(matches=120, total_hands=10, seeds=(42, 137))