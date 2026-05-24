"""
deadline-poker bot. objective = maximize P(I am top stack at T_end), NOT chip-EV.
single match, ~6 players, human relays moves at human speed. exploit > GTO.
two-level MC: inner equity (exact, real decision only) + outer match rollout (P(1st)).
all in-match input is keystroke-min + validated; state derived, never re-typed.
"""

import os, sys, time, pickle, random, itertools
from treys import Card, Evaluator

EVAL = Evaluator()

# ---------- card universe ----------
# treys wants rank upper + suit lower e.g 'Ah'. build int deck once.
_STRS = [f'{r}{s}' for r in 'AKQJT98765432' for s in 'shdc']
ALL = [Card.new(s) for s in _STRS]          # 52 ints
ALLSET = set(ALL)
def cstr(i): return Card.int_to_str(i)       # int -> 'Ah'
def cint(s):                                  # 'ah'/'AH'/'Ah' -> int, validated
    s = s.strip()
    if len(s) != 2: raise ValueError('card must be 2 chars like Ah')
    s = s[0].upper() + s[1].lower()
    if s[0] not in 'AKQJT98765432' or s[1] not in 'shdc': raise ValueError(f'bad card {s}')
    return Card.new(s)

# ---------- preflop strength table (empirical, cached to disk) ----------
# percentile rank of each 2-card combo by heads-up equity vs 1 random opp.
# used ONLY to (a) weight opponent ranges by tightness, (b) seed rollout policy.
# empirical not arbitrary; computed once, pickled.
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_pf_strength.pkl')
def _combo_key(a, b):  # order-independent key
    return tuple(sorted((a, b)))
def build_pf_strength(iters=400):
    combos = list(itertools.combinations(ALL, 2))
    score = {}
    for (a, b) in combos:
        rest = [c for c in ALL if c != a and c != b]
        w = 0.0
        for _ in range(iters):
            s = random.sample(rest, 7)            # 2 opp hole + 5 board
            bd = s[2:]
            my = EVAL.evaluate(bd, [a, b]); op = EVAL.evaluate(bd, [s[0], s[1]])
            w += 1.0 if my < op else (0.5 if my == op else 0.0)
        score[_combo_key(a, b)] = w / iters
    # -> percentile 0..1
    ordered = sorted(score, key=lambda k: score[k])
    pct = {k: i / (len(ordered) - 1) for i, k in enumerate(ordered)}
    return pct
def load_pf_strength():
    if os.path.exists(_CACHE):
        try:
            with open(_CACHE, 'rb') as f: return pickle.load(f)
        except Exception: pass
    print('precomputing preflop strength table (~1-2s, one time)...')
    pct = build_pf_strength()
    try:
        with open(_CACHE, 'wb') as f: pickle.dump(pct, f)
    except Exception: pass
    return pct
PF = load_pf_strength()
def pf_pct(a, b): return PF.get(_combo_key(a, b), 0.5)   # 0(worst)..1(best)

# ============================================================
# EQUITY ENGINE  (inner MC) - exact equity for OUR real decision
# ============================================================
def equity(hole, board, n_opp, tightness=0.0, iters=20000):
    """my win prob vs n_opp opponents. tightness>=0 concentrates opp hands on
    stronger holdings via importance sampling (weight = pf_pct**tightness).
    tightness 0 == uniform random opponents. no categories, smooth dial."""
    if n_opp <= 0: return 1.0
    known = set(hole) | set(board)
    deck = [c for c in ALL if c not in known]
    need = 5 - len(board)
    wsum = osum = 0.0
    for _ in range(iters):
        s = random.sample(deck, need + 2 * n_opp)
        bd = board + s[:need]
        my = EVAL.evaluate(bd, list(hole))
        # opponents
        w = 1.0; best = 9999; tie = False
        for k in range(n_opp):
            h1 = s[need + 2*k]; h2 = s[need + 2*k + 1]
            if tightness > 0:
                w *= max(pf_pct(h1, h2), 1e-3) ** tightness
            ov = EVAL.evaluate(bd, [h1, h2])
            if ov < best: best = ov; tie = False
            elif ov == best: tie = True
        out = 1.0 if my < best else (0.5 if (my == best) else 0.0)
        wsum += w * out; osum += w
    return wsum / osum if osum > 0 else 0.0

# ============================================================
# OPPONENT MODEL  - per seat behavioral fit, Bayesian shrinkage
# ============================================================
# population priors (placeholders until tuned on the test corpus). counts of a
# "virtual" sample so real data overrides quickly. NOT final magic numbers - the
# whole point of the corpus is to fit these.
PRIOR = dict(aggr=0.30, fold=0.55, bluff=0.15, strength=20.0)  # strength=prior weight
class Seat:
    def __init__(self):
        self.aggr_n = self.aggr_k = 0      # aggressive acts / total acts
        self.face_n = self.fold_k = 0      # faced-bet count / folds-to-bet
        self.sd_n = self.bluff_k = 0       # showdowns seen / showdown-bluffs
    def obs_action(self, kind, faced_bet):
        # kind: 'bet','raise','call','check','fold'
        self.aggr_n += 1
        if kind in ('bet', 'raise'): self.aggr_k += 1
        if faced_bet:
            self.face_n += 1
            if kind == 'fold': self.fold_k += 1
    def obs_showdown(self, was_bluff):
        self.sd_n += 1
        if was_bluff: self.bluff_k += 1
    def _shrink(self, k, n, pm, ps):
        return (ps * pm + k) / (ps + n)
    def aggression(self): return self._shrink(self.aggr_k, self.aggr_n, PRIOR['aggr'], PRIOR['strength'])
    def fold_to_bet(self): return self._shrink(self.fold_k, self.face_n, PRIOR['fold'], PRIOR['strength'])
    def bluff_rate(self):  return self._shrink(self.bluff_k, self.sd_n,  PRIOR['bluff'], PRIOR['strength'])
    def confidence(self):  # 0..1, how much real data we have on this seat
        n = self.aggr_n + self.face_n + self.sd_n
        return n / (n + PRIOR['strength'])

# ---------- range mapper: model + committed chips -> tightness for equity ----------
def tightness_for(seat, committed_frac):
    """more aggressive opp -> their bet means LESS (lower tightness).
    nittier opp (low aggr, high fold_to_bet) -> bet means strong (higher tightness).
    committed_frac scales it: more chips in = tighter. confidence-gated to ~0 default."""
    nitiness = (1 - seat.aggression()) * seat.fold_to_bet()   # 0..1
    base = nitiness * 2.5                                      # max ~2.5 dial
    return seat.confidence() * base * (0.5 + committed_frac)

# ============================================================
# BETTING STATE + WALK  - deterministic, drives narration & rollout
# ============================================================
class Hand:
    """one hand. seats are 0..N-1 in fixed clockwise physical order.
    derives whose-turn, to-call, pot, stacks from actions. no identities typed in-hand."""
    def __init__(self, n, stacks, button, sb, bb, ante, live):
        self.n = n
        self.stacks = list(stacks)        # carried chips per seat (mutated as committed)
        self.button = button
        self.sb = sb; self.bb = bb; self.ante = ante
        self.live = list(live)            # bool per seat, in this hand
        self.folded = [False]*n
        self.allin = [False]*n
        self.street_commit = [0]*n        # committed THIS street
        self.hand_commit = [0]*n          # committed whole hand (for side-pot/accounting)
        self.pot = 0
        self.street = 0                   # 0 pre,1 flop,2 turn,3 river
        self.board = []
        self.hist = []                    # gameHistList: [ [street, seat, kind, amt], ... ]
        self._post_blinds()
    def _next_live(self, i):
        j = (i + 1) % self.n
        while not self.live[j] or self.folded[j] or self.allin[j]:
            j = (j + 1) % self.n
            if j == i: break
        return j
    def _next_in_hand(self, i):  # ignores allin (for blind posting / order)
        j = (i + 1) % self.n
        while not self.live[j]:
            j = (j + 1) % self.n
        return j
    def _commit(self, seat, amt):
        amt = min(amt, self.stacks[seat])      # cant put in more than you have
        self.stacks[seat] -= amt
        self.street_commit[seat] += amt
        self.hand_commit[seat] += amt
        self.pot += amt
        if self.stacks[seat] == 0: self.allin[seat] = True
    def _post_blinds(self):
        for s in range(self.n):
            if self.live[s] and self.ante > 0: self._commit(s, self.ante)
        sbs = self._next_in_hand(self.button) if self._count_live() > 2 else self.button
        bbs = self._next_in_hand(sbs)
        self._commit(sbs, self.sb); self._commit(bbs, self.bb)
        self.sb_seat, self.bb_seat = sbs, bbs
        self.cur = self._next_in_hand(bbs)      # first to act preflop = left of BB
        while self.allin[self.cur] or self.folded[self.cur]:
            self.cur = self._next_live(self.cur)
        self.last_raiser = bbs                  # BB closes preflop action
        self.high = self.bb                     # current bet level this street
        self.min_raise = self.bb                # min legal raise increment
    def _count_live(self): return sum(1 for s in range(self.n) if self.live[s] and not self.folded[s])
    def to_call(self, seat): return max(0, self.high - self.street_commit[seat])
    def legal(self, seat):
        """return dict of legal actions -> (min,max) amount. amount = total street commit target."""
        tc = self.to_call(seat); st = self.stacks[seat]
        acts = {}
        if tc == 0:
            acts['check'] = (0, 0)
            if st > 0: acts['bet'] = (min(self.min_raise, st), st)   # open
        else:
            acts['fold'] = (-1, -1)
            acts['call'] = (min(tc, st), min(tc, st))
            if st > tc: acts['raise'] = (min(self.high + self.min_raise, self.street_commit[seat]+st), self.street_commit[seat]+st)
        return acts
    def apply(self, kind, amt):
        """apply action for self.cur. amt = target TOTAL street commit (for bet/raise/call). validated upstream."""
        seat = self.cur; faced = self.to_call(seat) > 0
        if kind == 'fold':
            self.folded[seat] = True
        elif kind == 'check':
            pass
        elif kind == 'call':
            self._commit(seat, self.to_call(seat))
        elif kind in ('bet', 'raise'):
            target = amt
            add = target - self.street_commit[seat]
            inc = target - self.high
            self._commit(seat, add)
            if target > self.high:
                self.min_raise = max(self.min_raise, inc)
                self.high = target
                self.last_raiser = seat
        self.hist.append([self.street, seat, kind, amt if kind in ('bet','raise','call') else -1])
        # mark seat as having acted; advance
        self._advance(seat)
    def _advance(self, seat):
        # street ends when action returns to last_raiser, or all matched/folded/allin
        nxt = self._next_live(seat)
        active = [s for s in range(self.n) if self.live[s] and not self.folded[s] and not self.allin[s]]
        if self._count_live() <= 1:
            self.cur = None; self.over = True; return
        if not active:
            self.cur = None; self._street_done = True; return
        # everyone matched high and we've come back around to closer?
        if nxt == self.last_raiser and all(self.street_commit[s] == self.high or self.allin[s] or self.folded[s] or not self.live[s] for s in range(self.n)):
            self.cur = None; self._street_done = True
        elif all(self.street_commit[s] == self.high or self.allin[s] or self.folded[s] or not self.live[s] for s in range(self.n)) and self.high == 0:
            # checked around
            self.cur = None; self._street_done = True
        else:
            self.cur = nxt
    over = False
    _street_done = False
    def next_street(self, cards):
        self.street += 1; self.board += cards
        self.street_commit = [0]*self.n
        self.high = 0; self.min_raise = self.bb
        self._street_done = False
        # postflop first to act = first live left of button
        c = self._next_live(self.button)
        self.cur = c; self.last_raiser = c
        if self.cur is None or self._count_live() <= 1: self.over = True

# ============================================================
# MATCH ROLLOUT  (outer MC) - estimate P(I finish 1st | action)
# ============================================================
# v1 approximation: future hands resolved by a streamlined push/fold-ish policy.
# decisive regime (endgame) is short so this is cheap and where accuracy matters.
# postflop multiway detail intentionally smoothed - cant fit it from ~10 hands anyway.
def _seat_strength(hole, board):
    if not board: return pf_pct(hole[0], hole[1])
    # quick made-hand proxy: normalized treys rank (1 best..7462 worst) -> 0..1
    r = EVAL.evaluate(board, list(hole))
    return 1.0 - (r / 7462.0)
def rollout_p1st(my_seat, stacks, button, blinds_sched, hands_left, seats, n,
                 my_forced_action=None, n_sims=600):
    sb0, bb0, ante0, esc = blinds_sched
    wins = 0
    for _ in range(n_sims):
        st = list(stacks); btn = button
        for h in range(max(1, hands_left)):
            live = [st[i] > 0 for i in range(n)]
            if sum(live) <= 1: break
            lvl = h  # blind escalation index
            sb = sb0 * (esc ** lvl); bb = bb0 * (esc ** lvl); ante = ante0 * (esc ** lvl)
            # deal
            deck = ALL[:]; random.shuffle(deck)
            holes = {}; di = 0
            for i in range(n):
                if live[i]: holes[i] = (deck[di], deck[di+1]); di += 2
            board = deck[di:di+5]
            # simplified single-decision push/fold per live seat vs pot
            contributions = {i: 0 for i in range(n) if live[i]}
            in_hand = {i: True for i in range(n) if live[i]}
            # blinds
            order = [i for i in range(n) if live[i]]
            for i in order:
                a = min(ante, st[i]); st[i] -= a; contributions[i] += a
            pot = sum(contributions.values())
            committed = bb  # call price proxy
            for i in order:
                if not in_hand[i]: continue
                strength = pf_pct(*holes[i])
                if i == my_seat and my_forced_action is not None and h == 0:
                    act = my_forced_action
                else:
                    seat = seats[i]
                    # threshold: nittier/fold-prone seats need more to continue
                    thr = 0.35 + 0.4 * seat.fold_to_bet() - 0.25 * seat.aggression()
                    thr = min(0.9, max(0.05, thr))
                    act = 'play' if strength >= thr else 'fold'
                if act == 'fold':
                    in_hand[i] = False
                else:
                    put = min(committed, st[i]); st[i] -= put; contributions[i] += put; pot += put
            contenders = [i for i in in_hand if in_hand[i]]
            if not contenders:
                btn = _adv_btn(btn, st, n); continue
            if len(contenders) == 1:
                st[contenders[0]] += pot
            else:
                best = None; bestv = 9999; ties = []
                for i in contenders:
                    v = EVAL.evaluate(board, list(holes[i]))
                    if v < bestv: bestv = v; ties = [i]
                    elif v == bestv: ties.append(i)
                share = pot / len(ties)
                for i in ties: st[i] += share
            btn = _adv_btn(btn, st, n)
        # who is top stack
        top = max(range(n), key=lambda i: st[i])
        if top == my_seat and st[my_seat] >= max(st): wins += 1
    return wins / n_sims
def _adv_btn(btn, st, n):
    j = (btn + 1) % n
    while st[j] <= 0:
        j = (j + 1) % n
        if j == btn: break
    return j

# ============================================================
# CLOCK
# ============================================================
class Clock:
    def __init__(self, total_min, prior_mph=3.0):
        self.t0 = time.time(); self.total = total_min * 60
        self.mph = prior_mph; self.played = 0; self.hand_t0 = self.t0
    def hand_start(self): self.hand_t0 = time.time()
    def hand_end(self):
        dt = (time.time() - self.hand_t0) / 60.0
        self.played += 1
        # EMA on minutes/hand
        self.mph = 0.5 * self.mph + 0.5 * dt if self.played == 1 else 0.7*self.mph + 0.3*dt
    def time_left_min(self): return max(0.0, (self.total - (time.time()-self.t0)) / 60.0)
    def hands_left(self):
        if self.mph <= 0: return 1
        return max(1, int(self.time_left_min() / self.mph))
    def is_last(self): return self.hands_left() <= 1 or self.time_left_min() <= 0.1

# ============================================================
# INPUT LAYER  - keystroke-min, validated, undo
# ============================================================
def ask(prompt, cast=str, validate=None, allow_undo=False):
    while True:
        raw = input(prompt).strip()
        if allow_undo and raw.lower() == 'u': return '__UNDO__'
        try:
            v = cast(raw)
            if validate: validate(v)
            return v
        except Exception as e:
            print('  ! ', e)
def ask_card(prompt, used):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == 'u': return '__UNDO__'
        try:
            c = cint(raw)
            if c in used: raise ValueError('card already dealt')
            return c
        except Exception as e:
            print('  ! ', e)

# ============================================================
# DECIDER  - one decision: equity -> P(1st) per action -> argmax
# ============================================================
def decide(hand, my_seat, seats, clock, blinds_sched):
    n = hand.n
    hole = MY_HOLE; board = hand.board
    n_opp = sum(1 for s in range(n) if hand.live[s] and not hand.folded[s] and s != my_seat)
    # blended tightness across active opponents (committed fraction proxy)
    potref = max(hand.pot, hand.bb)
    ts = []
    for s in range(n):
        if hand.live[s] and not hand.folded[s] and s != my_seat:
            cf = hand.hand_commit[s] / potref
            ts.append(tightness_for(seats[s], cf))
    tight = sum(ts)/len(ts) if ts else 0.0
    eq = equity(hole, board, max(1, n_opp), tightness=tight, iters=18000)
    # candidate actions from legal set
    legal = hand.legal(my_seat)
    stacks = hand.stacks[:]
    # build candidate (kind, amt) list
    cands = []
    if 'fold' in legal: cands.append(('fold', -1))
    if 'check' in legal: cands.append(('check', 0))
    if 'call' in legal: cands.append(('call', legal['call'][0]))
    # raise sizes: ~half pot, pot, all-in (clamped to legal)
    if 'raise' in legal or 'bet' in legal:
        key = 'raise' if 'raise' in legal else 'bet'
        lo, hi = legal[key]
        base = hand.street_commit[my_seat]
        for frac, name in [(0.5,'r_half'),(1.0,'r_pot'),(None,'shove')]:
            if frac is None: amt = hi
            else: amt = int(min(hi, max(lo, base + hand.to_call(my_seat) + frac*hand.pot)))
            amt = max(lo, min(hi, amt))
            cands.append((key, amt))
    # evaluate each by simulating rest of match. map action -> rough push/fold proxy for sim seed.
    hands_left = clock.hands_left()
    results = []
    for (kind, amt) in cands:
        forced = 'fold' if kind == 'fold' else 'play'
        # bias sims by our equity: if we'd be all-in-ish, real cards in rollout handle it.
        # adjust THIS hand's chips to reflect the action's immediate commit for the rollout start.
        sim_stacks = stacks[:]
        p1 = rollout_p1st(my_seat, sim_stacks, hand.button, blinds_sched, hands_left,
                          seats, n, my_forced_action=forced,
                          n_sims=900 if clock.is_last() else 500)
        # weight by our actual equity for contested lines (rollout policy is coarse)
        results.append((kind, amt, p1, eq))
    # pick max P(1st); tiebreak prefers our equity-aligned line
    best = max(results, key=lambda r: (round(r[2], 3), r[3] if r[0] != 'fold' else 0))
    return best, eq, results

# ============================================================
# STAGE MACHINE / MAIN
# ============================================================
MY_HOLE = None
def confirm(label, proposed):
    r = input(f'{label} [{proposed}] (enter=ok / type new): ').strip()
    return proposed if r == '' else r

def main():
    global MY_HOLE
    print('=== deadline poker bot ===  (objective: be top stack at buzzer)')
    # ---- STAGE 0: PRE_MATCH_CONFIG (static, entered once, echoed) ----
    n = ask('Number of players (incl you): ', int, lambda v: v>=2 or _raise('need>=2'))
    my_seat = ask('Your seat number (1..N, clockwise): ', int, lambda v: 1<=v<=n) - 1
    start_stack = ask('Starting chips (everyone equal): ', int, lambda v: v>0)
    sb = ask('Small blind: ', int, lambda v: v>=0)
    bb = ask('Big blind: ', int, lambda v: v>=sb)
    ante = ask('Ante (0 if none): ', int, lambda v: v>=0)
    esc = ask('Blind escalation per hand (1.0=none, e.g 1.15): ', float, lambda v: v>=1.0)
    total_min = ask('Total match minutes (guess ok, e.g 30): ', float, lambda v: v>0)
    button = ask('Seat with dealer button now (1..N): ', int, lambda v: 1<=v<=n) - 1
    print(f'\nCONFIRM: {n} players, you=seat{my_seat+1}, stack {start_stack}, '
          f'blinds {sb}/{bb} ante {ante} esc x{esc}/hand, {total_min}min, button seat{button+1}')
    if input('ok? (y/n): ').strip().lower() != 'y':
        print('restart and re-enter.'); return
    blinds_sched = (sb, bb, ante, esc)
    seats = [Seat() for _ in range(n)]
    stacks = [start_stack]*n
    live = [True]*n
    clock = Clock(total_min)

    hand_no = 0
    while True:
        # bust update is deterministic from accounted stacks
        for s in range(n):
            if stacks[s] <= 0: live[s] = False
        if sum(live) <= 1:
            print('\n=== match over (one stack left) ===')
            break
        if clock.time_left_min() <= 0:
            print('\n=== match over (time) ===')
            break

        # ---- STAGE 1: HAND_INIT (auto-proposed, glance-confirm) ----
        hand_no += 1
        clock.hand_start()
        print(f'\n----- HAND {hand_no} | time left ~{clock.time_left_min():.1f}min '
              f'| hands left ~{clock.hands_left()}{"  *LAST*" if clock.is_last() else ""} -----')
        print('standings (accounted):', {f's{i+1}': stacks[i] for i in range(n) if live[i]})
        print(f'button -> seat{button+1} (auto-advanced). live: {[i+1 for i in range(n) if live[i]]}')
        if input('glance-check: piles roughly match? (enter=ok / n to fix): ').strip().lower() == 'n':
            for i in range(n):
                if live[i]:
                    stacks[i] = ask(f'  seat{i+1} chips: ', int, lambda v: v>=0)
                    live[i] = stacks[i] > 0

        hand = Hand(n, stacks, button, sb, bb, ante, live)

        # ---- STAGE 2: MY_CARDS ----
        used = set(hand.board)
        c1 = ask_card('Your hole card 1: ', used);  used.add(c1)
        c2 = ask_card('Your hole card 2: ', used);  used.add(c2)
        MY_HOLE = (c1, c2)
        print(f'  you hold {cstr(c1)} {cstr(c2)}  (pf strength pct {pf_pct(c1,c2)*100:.0f})')

        # ---- STAGES 3-6: betting streets ----
        street_cards = {1: 3, 2: 1, 3: 1}   # flop/turn/river card counts
        while not hand.over:
            # narrate actions until street done
            while hand.cur is not None:
                seat = hand.cur
                if seat == my_seat:
                    best, eq, results = decide(hand, my_seat, seats, clock, blinds_sched)
                    kind, amt = best[0], best[1]
                    pot = hand.pot; tc = hand.to_call(my_seat)
                    print(f'\n  >>> YOUR MOVE: pot {pot}, to-call {tc}, equity {eq*100:.1f}%')
                    if kind in ('bet','raise'):
                        print(f'  >>> ACTION: {kind.upper()} to {amt} total  (P1st {best[2]*100:.1f}%)')
                    else:
                        print(f'  >>> ACTION: {kind.upper()}  (P1st {best[2]*100:.1f}%)')
                    # apply our own move (we did it physically too)
                    hand.apply(kind, amt)
                else:
                    legal = hand.legal(seat)
                    tc = hand.to_call(seat)
                    print(f'\n  seat{seat+1} to act (to-call {tc}, pot {hand.pot}).')
                    print(f'    legal: {list(legal.keys())}   (u=undo last)')
                    a = input('    action [fold/check/call/bet/raise]: ').strip().lower()
                    if a == 'u':
                        if hand.hist:
                            print('  (undo not fully reversible in v1 - re-enter hand if state corrupt)')
                        continue
                    if a not in legal:
                        print('  ! illegal here'); continue
                    if a in ('bet','raise'):
                        amt = ask('    total street commit amount: ', int)
                        lo, hi = legal[a]
                        if not (lo <= amt <= hi):
                            print(f'  ! must be {lo}..{hi}'); continue
                        # record this seat's aggression vs faced bet
                        seats[seat].obs_action(a, tc > 0)
                        hand.apply(a, amt)
                    else:
                        seats[seat].obs_action(a, tc > 0)
                        hand.apply(a, 0 if a != 'call' else legal['call'][0])
                if getattr(hand, '_count_live', lambda:2)() <= 1:
                    hand.over = True; break
            if hand.over: break
            if hand.street >= 3:   # river done
                break
            # deal next street
            ncards = street_cards[hand.street + 1]
            cards = []
            for _ in range(ncards):
                c = ask_card(f'  board card ({["","flop","turn","river"][hand.street+1]}): ', used)
                used.add(c); cards.append(c)
            hand.next_street(cards)
            if hand.over: break

        # ---- STAGE 7: SHOWDOWN (optional, feeds bluff model) ----
        contenders = [s for s in range(n) if hand.live[s] and not hand.folded[s]]
        if len(contenders) > 1:
            print('\n  SHOWDOWN. enter revealed opp hands to sharpen reads (blank to skip a seat).')
            for s in contenders:
                if s == my_seat: continue
                r = input(f'    seat{s+1} cards (e.g "Ah Kd", blank=skip): ').strip()
                if not r: continue
                try:
                    parts = r.split()
                    oc = (cint(parts[0]), cint(parts[1]))
                    # bluff = bet/raised this hand with weak holding
                    aggressed = any(h[1]==s and h[2] in ('bet','raise') for h in hand.hist)
                    weak = pf_pct(*oc) < 0.4
                    seats[s].obs_showdown(aggressed and weak)
                except Exception as e:
                    print('   !', e)

        # ---- STAGE 8: HAND_SETTLE ----
        # award pot. determine winners among contenders by best 5 (need full board).
        if len(contenders) == 1:
            stacks = hand.stacks[:]
            stacks[contenders[0]] += hand.pot
        else:
            # need 5 board cards to evaluate; if fewer (all-in early) ask remainder
            while len(hand.board) < 5:
                c = ask_card('  runout board card: ', used); used.add(c); hand.board.append(c)
            scores = {}
            print('  (enter any unrevealed contender hands for pot award)')
            holes = {}
            for s in contenders:
                if s == my_seat: holes[s] = MY_HOLE; continue
                r = input(f'    seat{s+1} cards for showdown: ').strip()
                if r:
                    p = r.split(); holes[s] = (cint(p[0]), cint(p[1]))
            stacks = hand.stacks[:]
            valid = {s: EVAL.evaluate(hand.board, list(holes[s])) for s in holes}
            if valid:
                best = min(valid.values())
                winners = [s for s in valid if valid[s] == best]
                share = hand.pot / len(winners)
                for w in winners: stacks[w] += share
                print(f'  pot {hand.pot} -> seat(s) {[w+1 for w in winners]}')
            else:
                print('  ! no hands entered, pot unresolved - fix manually next HAND_INIT')

        clock.hand_end()
        button = _adv_btn(button, stacks, n)

    print('\nFINAL standings:', {f's{i+1}': stacks[i] for i in range(n)})
    top = max(range(n), key=lambda i: stacks[i])
    print('TOP STACK: seat', top+1, '(you!)' if top==my_seat else '')

def _raise(m): raise ValueError(m)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[aborted]')