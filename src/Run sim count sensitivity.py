"""does cranking sim count change the result, or is the cut-down rollout fine?
if results are within CI of each other, low knobs are good enough -> cheap experiments.
copy back: the 3 summary lines."""
import Sim
for label, knobs in [
    ('low  2500/200/350',  dict(eq_iters=2500, base_sims=200,  last_sims=350)),
    ('mid  5000/500/800',  dict(eq_iters=5000, base_sims=500,  last_sims=800)),
    ('high 12000/1200/1800', dict(eq_iters=12000, base_sims=1200, last_sims=1800)),
]:
    Sim.winrate(matches=20, total_hands=10, seed=21, label=label, **knobs)