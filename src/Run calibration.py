"""diagnose rollout calibration. each decision logs predicted P(1st);
we compare predicted vs actual win-rate across buckets, actions, streets.
gap > 0.10 in a bucket = rollout systematically lies about that class of decisions.
runtime ~3-5 min for N=40."""
import Sim
Sim.calibration_run(matches=40, total_hands=10, seed=99)