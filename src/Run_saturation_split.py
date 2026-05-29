"""test the two-regime reframe (chip-EV mid-game vs P1st deadline).
classifies our decisions saturated vs non-saturated, checks where wins come from.
runtime ~4-6 min for N=60.
copy back: everything from 'N=60' to the end."""
import Sim
Sim.saturation_split(matches=60, total_hands=10, seed=77)