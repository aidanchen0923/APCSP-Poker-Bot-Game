"""
headless test medium for pokerbot. NO human input.
- archetype zoo (the diverse opponents we cant get the real ones, so we synthesize the space)
- match runner reusing pokerbot's exact engine (so we test the SAME code that runs live)
- chip-conservation check (engine correctness)
- prior measurement (sets PRIOR from population behavior -> kills placeholder guesses)
- win-rate harness (our 1st-place rate vs the zoo -> the only metric that matters)
run:  python sim.py
"""
import random, statistics, sys
import pokerbot as pb
from pokerbot import Hand, Seat, EVAL, pf_pct, _seat_strength, _adv_btn, decide

# ---------------- intent resolver ----------------
# archetypes emit intent + size; resolver maps to a LEGAL engine action. robust to context.
def resolve(hand, seat, intent, potfrac, rng):
    legal = hand.legal(seat); tc = hand.to_call(seat)
    if intent == 'fold':
        return ('fold', -1) if 'fold' in legal else ('check', 0)
    if intent == 'aggressive':
        key = 'raise' if 'raise' in legal else ('bet' if 'bet' in legal else None)
        if key:
            lo, hi = legal[key]
            base = hand.street_commit[seat]
            target = int(base + tc + max(potfrac, 0.1) * max(hand.pot, hand.bb))
            return (key, max(lo, min(hi, target)))
        return ('call', legal['call'][0]) if 'call' in legal else ('check', 0)
    # passive
    if tc > 0 and 'call' in legal: return ('call', legal['call'][0])
    return ('check', 0)

# ---------------- archetype policies ----------------
# each: (hand, seat, hole, board, rng) -> (intent, potfrac)
def A_nit(h, s, hole, board, rng):
    st = _seat_strength(hole, board); tc = h.to_call(s)
    if tc == 0: return ('aggressive', 0.75) if st > 0.85 else ('passive', 0)
    if st > 0.9: return ('aggressive', 1.0)
    if st > 0.72: return ('passive', 0)
    return ('fold', 0)
def A_tag(h, s, hole, board, rng):  # honest tight-aggressive
    st = _seat_strength(hole, board); tc = h.to_call(s)
    if tc == 0: return ('aggressive', 0.66) if st > 0.6 else ('passive', 0)
    if st > 0.78: return ('aggressive', 0.9)
    if st > 0.52: return ('passive', 0)
    return ('fold', 0)
def A_maniac(h, s, hole, board, rng):
    st = _seat_strength(hole, board); tc = h.to_call(s); r = rng.random()
    if tc == 0: return ('aggressive', 1.0) if r < 0.85 else ('passive', 0)
    if st > 0.45 or r < 0.55: return ('aggressive', 1.0) if r < 0.4 else ('passive', 0)
    return ('fold', 0)
def A_station(h, s, hole, board, rng):
    st = _seat_strength(hole, board); tc = h.to_call(s)
    if tc == 0: return ('aggressive', 0.5) if st > 0.92 else ('passive', 0)
    return ('passive', 0)   # calls everything, never folds/raises
def A_bluffer(h, s, hole, board, rng):  # fixed random bluff freq, else honest
    if rng.random() < 0.30: return ('aggressive', 1.0)
    return A_tag(h, s, hole, board, rng)
def A_flowchart(h, s, hole, board, rng):  # deterministic buckets, no rng
    st = _seat_strength(hole, board); tc = h.to_call(s)
    band = int(st * 5)   # 0..5
    if tc == 0:
        return ('aggressive', 0.5 + 0.1*band) if band >= 3 else ('passive', 0)
    if band >= 4: return ('aggressive', 0.75)
    if band >= 2: return ('passive', 0)
    return ('fold', 0)

ZOO = {'nit':A_nit, 'tag':A_tag, 'maniac':A_maniac, 'station':A_station,
       'bluffer':A_bluffer, 'flowchart':A_flowchart}

# ---------------- our bot wrapper ----------------
class SimClock:
    def __init__(self, hl): self.hl = hl
    def hands_left(self): return self.hl
    def is_last(self): return self.hl <= 1
class OurBot:
    def __init__(self, n): self.models = [Seat() for _ in range(n)]
    def act(self, hand, seat, hole, board, clock, blinds, rng):
        pb.MY_HOLE = hole
        best, eq, _ = decide(hand, seat, self.models, clock, blinds,
                             eq_iters=2500, base_sims=200, last_sims=350)  # fast knobs for sim
        return best[0], best[1]

# ---------------- headless match ----------------
def play_match(players, start_stack, blinds, total_hands, rng, our_idx=None, record=None):
    """players: list of dict(type,policy or bot). returns final stacks. chips conserved.
    record: optional dict to accumulate per-type action freqs for prior measurement."""
    n = len(players)
    sb0, bb0, ante0, esc = blinds
    stacks = [start_stack]*n
    button = rng.randrange(n)
    total_chips = sum(stacks)
    for hno in range(total_hands):
        live = [stacks[i] > 0 for i in range(n)]
        if sum(live) <= 1: break
        lvl = hno
        sb = int(sb0*(esc**lvl)); bb = int(bb0*(esc**lvl)); ante = int(ante0*(esc**lvl))
        hand = Hand(n, stacks, button, sb, bb, ante, live)
        # deal
        deck = pb.ALL[:]; rng.shuffle(deck); di = 0
        holes = {}
        for i in range(n):
            if live[i]: holes[i] = (deck[di], deck[di+1]); di += 2
        board_full = deck[di:di+5]
        hands_left = total_hands - hno
        # betting streets
        while not hand.over:
            while hand.cur is not None:
                seat = hand.cur
                tc = hand.to_call(seat)
                hole = holes[seat]; board = hand.board
                if our_idx is not None and seat == our_idx:
                    kind, amt = players[seat]['bot'].act(hand, seat, hole, board,
                                                         SimClock(hands_left), blinds, rng)
                else:
                    intent, pf = players[seat]['policy'](hand, seat, hole, board, rng)
                    kind, amt = resolve(hand, seat, intent, pf, rng)
                # record freqs for prior measurement
                if record is not None and not (our_idx is not None and seat == our_idx):
                    t = players[seat]['type']; d = record.setdefault(t, dict(aggr_k=0,aggr_n=0,fold_k=0,face_n=0,bluff_k=0,sd_n=0))
                    d['aggr_n'] += 1
                    if kind in ('bet','raise'): d['aggr_k'] += 1
                    if tc > 0:
                        d['face_n'] += 1
                        if kind == 'fold': d['fold_k'] += 1
                # feed our bot's opponent models (mirror live loop)
                if our_idx is not None and seat != our_idx:
                    players[our_idx]['bot'].models[seat].obs_action(kind, tc > 0)
                hand.apply(kind, amt if kind in ('bet','raise','call') else 0)
                if hand._count_live() <= 1: hand.over = True; break
            if hand.over or hand.street >= 3: break
            nc = {1:3,2:1,3:1}[hand.street+1]
            hand.next_street(board_full[len(hand.board)-0:][:nc] if False else _next_board(hand, board_full, nc))
            if hand.over: break
        # showdown / award
        contenders = [s for s in range(n) if hand.live[s] and not hand.folded[s]]
        while len(hand.board) < 5 and len(contenders) > 1:
            hand.board.append(board_full[len(hand.board)])
        stacks = hand.stacks[:]
        if len(contenders) == 1:
            stacks[contenders[0]] += hand.pot
        elif contenders:
            vals = {s: EVAL.evaluate(hand.board, list(holes[s])) for s in contenders}
            best = min(vals.values()); winners = [s for s in vals if vals[s]==best]
            for w in winners: stacks[w] += hand.pot/len(winners)
            # bluff record: aggressed with weak holding then showed down
            if record is not None:
                for s in contenders:
                    if our_idx is not None and s == our_idx: continue
                    aggressed = any(a[1]==s and a[2] in ('bet','raise') for a in hand.hist)
                    d = record.get(players[s]['type'])
                    if d is not None:
                        d['sd_n'] += 1
                        if aggressed and pf_pct(*holes[s]) < 0.4: d['bluff_k'] += 1
            if our_idx is not None:
                for s in contenders:
                    if s == our_idx: continue
                    aggressed = any(a[1]==s and a[2] in ('bet','raise') for a in hand.hist)
                    players[our_idx]['bot'].models[s].obs_showdown(aggressed and pf_pct(*holes[s])<0.4)
        # conservation check (rounding from split pots -> tolerate tiny float)
        assert abs(sum(stacks) - total_chips) < 1e-6, f'CHIP LEAK: {sum(stacks)} vs {total_chips}'
        button = _adv_btn(button, stacks, n)
    return stacks
def _next_board(hand, board_full, nc):
    start = len(hand.board); return board_full[start:start+nc]

# ---------------- harnesses ----------------
def measure_priors(matches=200, n=6, seed=0):
    """run ZOO-only matches, collect population behavior -> suggested PRIOR. no our-bot."""
    rng = random.Random(seed); rec = {}
    types = list(ZOO)
    for _ in range(matches):
        chosen = [rng.choice(types) for _ in range(n)]
        players = [dict(type=t, policy=ZOO[t]) for t in chosen]
        play_match(players, 1000, (10,20,0,1.0), 12, rng, our_idx=None, record=rec)
    aggr=[]; fold=[]; bluff=[]
    print('\n-- per-archetype observed frequencies --')
    for t, d in sorted(rec.items()):
        a = d['aggr_k']/max(1,d['aggr_n']); f = d['fold_k']/max(1,d['face_n']); b = d['bluff_k']/max(1,d['sd_n'])
        print(f'  {t:10s} aggr {a:.2f}  fold2bet {f:.2f}  bluff {b:.2f}  (n={d["aggr_n"]})')
        aggr.append(a); fold.append(f); bluff.append(b)
    print('\nSUGGESTED PRIOR (population mean across zoo):')
    print(f"  PRIOR = dict(aggr={statistics.mean(aggr):.3f}, fold={statistics.mean(fold):.3f}, "
          f"bluff={statistics.mean(bluff):.3f}, strength=20.0)")
    return statistics.mean(aggr), statistics.mean(fold), statistics.mean(bluff)

def winrate(matches=60, n=6, seed=1, total_hands=12, mix=None):
    """our 1st-place rate vs random zoo opponents. baseline random = 1/n."""
    rng = random.Random(seed); firsts = 0; placings = []
    types = list(ZOO)
    for m in range(matches):
        our = rng.randrange(n)
        players = []
        for i in range(n):
            if i == our: players.append(dict(type='OURS', bot=OurBot(n)))
            else:
                t = (mix[rng.randrange(len(mix))] if mix else rng.choice(types))
                players.append(dict(type=t, policy=ZOO[t]))
        final = play_match(players, 1000, (10,20,0,1.0), total_hands, rng, our_idx=our)
        rank = sorted(range(n), key=lambda i: -final[i]).index(our) + 1
        placings.append(rank)
        if rank == 1: firsts += 1
    print(f'\nour 1st-place rate: {firsts}/{matches} = {firsts/matches:.1%}  (random baseline {1/n:.1%})')
    print(f'avg placing: {statistics.mean(placings):.2f} / {n}   (lower better)')
    return firsts/matches

if __name__ == '__main__':
    print('=== ENGINE + STRATEGY TEST HARNESS ===')
    print('[1] measuring population priors from zoo (also exercises engine + chip conservation)...')
    measure_priors(matches=120)
    print('\n[2] our 1st-place rate vs full random zoo...')
    winrate(matches=40)
    print('\n[3] vs all-maniac table (exploit check: should crush)...')
    winrate(matches=30, mix=['maniac'])
    print('\n[4] vs all-nit table (steal check)...')
    winrate(matches=30, mix=['nit'])