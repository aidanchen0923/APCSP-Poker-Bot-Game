# Poker Bot Architecture

## Data Structures

### Cards
```python
holeCards    # list or tuple of 2 cards — your private cards
commonCards  # list of 0, 3, 4, or 5 cards — community cards
```

### Hand Storage
```python
handRankings  # dict or list storing poker hands with their rank
              # e.g. { "royal_flush": 10, "straight_flush": 9, ... }
```

### Players
```python
playerList   # list of player names in acting order, index 0 acts first
selfVar      # string — your name as it appears in playerList
startingStack  # int — chip count at the start of the match
```

### Game History
```python
gameHistList  # 2D list — each element is one player action
              # format: [playerName, move, value]
              # value = chip amount if raise or call, -1 if fold or check
              # e.g. [["Alice", "raise", 50], ["Bob", "fold", -1]]
```

### Moves
```python
moves  # list of valid moves e.g. ["fold", "check", "call", "raise"]
```

---

## Core Functions

### `selfHandEval(holeCards, commonCards)`
Computes your hand equity across all street states.

- Runs Monte Carlo simulation over remaining community cards using:

$$\binom{52 - 2n - c}{5 - c}$$

where $c$ is the number of community cards already revealed

- Returns probability of holding each winning hand rank, for every possible final board state
- Gracefully handles all streets — pre-flop through river
- At the river ($c = 5$), returns deterministic hand rank with no simulation needed

**Returns:** `dict` mapping hand rank → probability

---

### `elseHandEval(holeCards, commonCards, playerCount, aggressionLabels)`
Estimates the probability that any opponent holds each winning hand rank.

- Samples from opponent hand space: $\dfrac{52!}{(52-2n)! \cdot 2^n}$ reduced by known cards
- Accounts for nonlinear scaling — probability that *someone* beats you is not simply `n × singleOpponentProbability` due to deck correlation
- `aggressionLabels` is a dict mapping player name → aggression profile, used to tighten or loosen assumed range per opponent
- Early hands (1–3): treats all opponents as uniform random range
- Later hands (4+): feeds aggression label as a multiplier on range prior
- Any showdown: hard overrides prior with observed cards

**Returns:** `dict` mapping hand rank → probability that at least one opponent holds it

---

### `gameHistEval(gameHistList, startingStack)`
Preprocesses game history into derived values for the decision function.

- **Pot size** — sum of all non `-1` values across all actions
- **Stack sizes** — `startingStack` minus each player's total contributed values
- **Pot odds** — given current call amount:

$$\text{potOdds} = \frac{\text{callAmount}}{\text{potSize} + \text{callAmount}}$$

- **Aggression labels** — per-player profile derived from raise frequency, bet sizing patterns, and fold-to-pressure rate
  - Hands 1–3: no signal, return neutral label
  - Hands 4–6: coarse label (loose/tight, aggressive/passive)
  - Any showdown: hard update on that player's label

**Returns:** `dict` containing `potSize`, `stackSizes`, `potOdds`, `aggressionLabels`

---

### `decisionFunction(selfEval, elseEval, histEval)`
Takes outputs of all three eval functions and returns the optimal move.

- Compares your win probability from `selfEval` against `potOdds` from `histEval`
- If `P(win) > potOdds` → calling has positive EV
- Weights `elseEval` to adjust for how many opponents are live and their aggression labels
- Accounts for hand type relative to player count:
  - Drawing hands (flushes, straights) — EV more resilient in multiway pots
  - Strong made hands (top pair etc.) — EV degrades faster with more players
- Returns a move from `moves` and a raise value if applicable

**Returns:** `(move, value)` where `value` is `-1` if move is fold or check

---

## Monte Carlo Simulation

Configuration count at any street:

$$\frac{52!}{(52-2n)! \cdot 2^n} \cdot \binom{52 - 2n - c}{5 - c}$$

| Street | $c$ | Community Factor |
|--------|-----|-----------------|
| Pre-flop | 0 | $\binom{52-2n}{5}$ |
| Post-flop | 3 | $\binom{49-2n}{2}$ |
| Post-turn | 4 | $\binom{48-2n}{1}$ |
| Post-river | 5 | $\binom{47-2n}{0} = 1$ |

- Community cards are **never trimmed** — no signal exists to bias them
- Opponent hand space is trimmed via aggression labels from `gameHistEval`
- Bluff detection is **coarse and binary** — trustworthy vs untrustworthy — not full range modeling, given the limited hand count per match

---

## Opponent Modeling Strategy

Given a short match (~10 hands), full Bayesian convergence is not feasible.

| Hands Played | Strategy |
|--------------|----------|
| 1–3 | Pure equity, ignore all opponent signals |
| 4–6 | Coarse aggression label, adjust range prior |
| 7–10 | 1–2 showdowns expected, hard update on priors |

**Volatile bots** (random aggression per match) are handled naturally — within a single match they are consistent, so within-match learning still holds. Bots randomizing within a match are effectively noise and opponent modeling is abandoned in favor of pure equity for those players.

**Zero-sum note:** More players does not increase aggregate EV. Strong made hands degrade in win probability faster than drawing hands as player count increases — `elseHandEval` must account for this nonlinearity.
