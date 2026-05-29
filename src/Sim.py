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
    """wrapper. knobs are instance attrs so experiments can vary them per match.
    eq_iters: inner equity MC iterations (live=18000, sim default=2500).
    base_sims/last_sims: outer rollout sim count for regular/last hand.
    log: if True, append per-decision data to self.decisions:
       (street, n_opp, eq, chosen_kind, chosen_amt, predicted_p1st, all_results)"""
    def __init__(self, n, eq_iters=2500, base_sims=200, last_sims=350, log=False):
        self.models = [Seat() for _ in range(n)]
        self.eq_iters = eq_iters; self.base_sims = base_sims; self.last_sims = last_sims
        self.log = log; self.decisions = []
    def act(self, hand, seat, hole, board, clock, blinds, rng):
        pb.MY_HOLE = hole
        best, eq, results = decide(hand, seat, self.models, clock, blinds,
                                   eq_iters=self.eq_iters, base_sims=self.base_sims,
                                   last_sims=self.last_sims)
        if self.log:
            n_opp = sum(1 for s in range(hand.n)
                        if hand.live[s] and not hand.folded[s] and s != seat)
            self.decisions.append(dict(
                street=hand.street, n_opp=n_opp, eq=eq,
                chosen_kind=best[0], chosen_amt=best[1], pred_p1st=best[2],
                all_results=[(k, a, p) for (k, a, p, _) in results]))
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

def winrate(matches=60, n=6, seed=1, total_hands=12, mix=None,
            start_stack=1000, blinds=(10,20,0,1.0),
            eq_iters=2500, base_sims=200, last_sims=350,
            progress_every=0, label=None, return_full=False):
    """1st-place rate vs zoo. baseline random = 1/n.
    mix: list of archetypes to fill non-our seats (uniform pick), else random zoo.
    progress_every: print 1st-rate every N matches (0=silent).
    returns dict(wr, ci_lo, ci_hi, avg_place, n, placings, by_archetype, last_rate)."""
    import math
    rng = random.Random(seed); firsts = 0; placings = []
    types = list(ZOO)
    by_arch_seen = {}; by_arch_their_first = {}
    last_count = 0; last_first = 0
    for m in range(matches):
        our = rng.randrange(n)
        opp_types = []
        players = []
        for i in range(n):
            if i == our:
                players.append(dict(type='OURS', bot=OurBot(n, eq_iters, base_sims, last_sims)))
            else:
                t = (mix[rng.randrange(len(mix))] if mix else rng.choice(types))
                opp_types.append(t)
                players.append(dict(type=t, policy=ZOO[t]))
        final = play_match(players, start_stack, blinds, total_hands, rng, our_idx=our)
        rank = sorted(range(n), key=lambda i: -final[i]).index(our) + 1
        placings.append(rank)
        if rank == 1: firsts += 1
        for t in opp_types:
            by_arch_seen[t] = by_arch_seen.get(t,0) + 1
            if rank == 1: by_arch_their_first[t] = by_arch_their_first.get(t,0)
        if rank == 1:
            for t in set(opp_types):  # we beat at least one of this type
                by_arch_their_first[t] = by_arch_their_first.get(t,0) + opp_types.count(t)
        # last-quintile drift check
        if m >= matches - max(10, matches//5):
            last_count += 1; last_first += (rank == 1)
        if progress_every and (m+1) % progress_every == 0:
            print(f'  ...{m+1}/{matches} 1st-so-far {firsts/(m+1):.1%}', flush=True)
    N = matches; wr = firsts/N
    se = math.sqrt(wr*(1-wr)/N) if N>0 else 0
    ci_lo, ci_hi = max(0,wr-1.96*se), min(1,wr+1.96*se)
    last_rate = last_first/last_count if last_count else None
    tag = f'[{label}] ' if label else ''
    print(f'{tag}N={N} h={total_hands} seed={seed}: 1st {firsts}/{N} = {wr:.1%} '
          f'CI[{ci_lo*100:.1f}%,{ci_hi*100:.1f}%] base {1/n:.1%}  avgplace {statistics.mean(placings):.2f}/{n}')
    out = dict(wr=wr, ci_lo=ci_lo, ci_hi=ci_hi, avg_place=statistics.mean(placings),
               n=N, firsts=firsts, last_rate=last_rate,
               placings={k: placings.count(k) for k in range(1,n+1)})
    if return_full:
        out['raw_placings'] = placings
        out['by_archetype_seen'] = by_arch_seen
    return out

def paired_ab(matches=200, n=6, base_seed=2025, total_hands=10,
              eq_iters=2500, base_sims=200, last_sims=350, regime_K=2):
    """DECISION 0 pre-registered test. paired A/B: same seed -> same deals, same opp picks,
    same seat. only difference is USE_REGIME toggle. measure paired difference in 1st-rate.
    pass iff 95% CI on diff EXCLUDES ZERO."""
    import math, pokerbot as _pb
    _pb.REGIME_K = regime_K
    diffs = []       # +1 = new won and old lost, -1 = old won and new lost, 0 = same outcome
    n_new = 0; n_old = 0
    for m in range(matches):
        seed = base_seed + m
        # NEW bot: regime ON
        _pb.USE_REGIME = True
        rng = random.Random(seed); types = list(ZOO)
        our = rng.randrange(n)
        opp_types = []
        for i in range(n):
            if i != our:
                opp_types.append(rng.choice(types))
        # same opp draws will replay because we re-seed identically below
        rng = random.Random(seed)
        our_check = rng.randrange(n); assert our_check == our
        players = []
        oi = 0
        for i in range(n):
            if i == our:
                players.append(dict(type='OURS', bot=OurBot(n, eq_iters, base_sims, last_sims)))
            else:
                t = rng.choice(types); players.append(dict(type=t, policy=ZOO[t]))
        final_new = play_match(players, 1000, (10,20,0,1.0), total_hands,
                               random.Random(seed*7919), our_idx=our)
        rank_new = sorted(range(n), key=lambda i: -final_new[i]).index(our) + 1
        won_new = (rank_new == 1)
        # OLD bot: regime OFF, identical setup
        _pb.USE_REGIME = False
        rng = random.Random(seed)
        our2 = rng.randrange(n); assert our2 == our
        players = []
        for i in range(n):
            if i == our:
                players.append(dict(type='OURS', bot=OurBot(n, eq_iters, base_sims, last_sims)))
            else:
                t = rng.choice(types); players.append(dict(type=t, policy=ZOO[t]))
        final_old = play_match(players, 1000, (10,20,0,1.0), total_hands,
                               random.Random(seed*7919), our_idx=our)
        rank_old = sorted(range(n), key=lambda i: -final_old[i]).index(our) + 1
        won_old = (rank_old == 1)
        n_new += won_new; n_old += won_old
        diffs.append(int(won_new) - int(won_old))
        if (m+1) % 10 == 0:
            cur_diff = sum(diffs)/len(diffs)
            print(f'  ...{m+1}/{matches} new {n_new} vs old {n_old} (diff {cur_diff:+.3f})',
                  flush=True)
    # paired difference + 95% CI
    mean_d = sum(diffs)/len(diffs)
    var_d = sum((d - mean_d)**2 for d in diffs) / (len(diffs)-1) if len(diffs)>1 else 0
    se = math.sqrt(var_d / len(diffs))
    ci_lo, ci_hi = mean_d - 1.96*se, mean_d + 1.96*se
    print(f'\n=== DECISION 0 RESULT ===')
    print(f'N={matches} paired matches, regime_K={regime_K}')
    print(f'NEW (chip-EV mid + P1st endgame): {n_new}/{matches} = {n_new/matches:.1%}')
    print(f'OLD (P1st everywhere):            {n_old}/{matches} = {n_old/matches:.1%}')
    print(f'Paired diff: {mean_d:+.3f}   95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]')
    if ci_lo > 0:
        print('PASS: CI excludes zero on the positive side. Adopt chip-EV mid-game.')
    elif ci_hi < 0:
        print('FAIL+: CI excludes zero on the negative side. Chip-EV mid-game is WORSE.')
    else:
        print('FAIL: CI includes zero. No measurable effect.')
    return dict(n_new=n_new, n_old=n_old, mean_diff=mean_d, ci_lo=ci_lo, ci_hi=ci_hi)

def raise_audit(seed=55, n=6, trials=4000):
    """isolate WHY raises are overrated. sample the resolver directly across realistic
    raise contexts and record folded-out vs called-won vs called-lost.
    KEY: contexts are built so avg equity is ~0.5. if 'called & won' rate stays near 0.5
    (or higher) instead of dropping, the resolver is using UNCONDITIONED equity at showdown
    -- i.e. it ignores that callers hold stronger-than-random hands. that's the double-count."""
    import pokerbot as _pb
    rng = random.Random(seed)
    _pb._RESOLVE_TRACE = []
    eq_sum = 0.0
    for _ in range(trials):
        n_opp = rng.randint(1, 4)
        opps = []
        for _ in range(n_opp):
            s = Seat()
            for _ in range(rng.randint(0, 25)):
                s.obs_action(rng.choice(['fold','call','raise','check']), rng.random()<0.5)
            opps.append((rng.randrange(1, n), 0, s))
        eq = rng.random(); eq_sum += eq
        pot = rng.randint(40, 400)
        my_added = rng.choice([pot//2, pot, pot*2])
        st = [1000]*n
        ctx = dict(pot=pot, my_added=my_added, my_fold=False, eq=eq, opps=opps)
        _pb._resolve_current(st, 0, ctx, rng)
    tr = _pb._RESOLVE_TRACE; _pb._RESOLVE_TRACE = None
    fo = sum(1 for t,_ in tr if t=='folded_out')
    cw = sum(1 for t,_ in tr if t=='called_won')
    cl = sum(1 for t,_ in tr if t=='called_lost')
    tot = max(1, fo+cw+cl); called = max(1, cw+cl)
    print(f'\nRESOLVER raise outcomes over {tot} sampled raises (avg input eq {eq_sum/trials:.2f}):')
    print(f'  folded out:    {fo/tot:.1%}')
    print(f'  called & won:  {cw/tot:.1%}')
    print(f'  called & lost: {cl/tot:.1%}')
    print(f'  => when called, our win-rate = {cw/called:.1%}')
    print(f'     if this ~= avg input eq (0.50), resolver treats callers as RANDOM hands.')
    print(f'     real callers are stronger -> true called-win should be WELL BELOW 0.50.')
    print(f'     gap between {cw/called:.1%} and reality = the raise overconfidence source.')
    return dict(folded_out=fo/tot, called_winrate=cw/called)

def calibration_run(matches=30, n=6, seed=99, total_hands=10, mix=None,
                    eq_iters=2500, base_sims=200, last_sims=350):
    """log every decision, group by predicted-P(1st) bucket, compare to actual outcome.
    a well-calibrated rollout: bucket 'predicted 60%' should win ~60% of the time.
    persistent over-prediction in a bucket = the rollout lies about that action class."""
    rng = random.Random(seed); types = list(ZOO)
    all_decisions = []
    firsts = 0
    for m in range(matches):
        our = rng.randrange(n)
        players = []
        for i in range(n):
            if i == our:
                players.append(dict(type='OURS',
                    bot=OurBot(n, eq_iters, base_sims, last_sims, log=True)))
            else:
                t = (mix[rng.randrange(len(mix))] if mix else rng.choice(types))
                players.append(dict(type=t, policy=ZOO[t]))
        final = play_match(players, 1000, (10,20,0,1.0), total_hands, rng, our_idx=our)
        rank = sorted(range(n), key=lambda i: -final[i]).index(our) + 1
        won = (rank == 1)
        if won: firsts += 1
        for d in players[our]['bot'].decisions:
            all_decisions.append((d, won))
    print(f'\nN={matches} matches, {len(all_decisions)} decisions, 1st-rate {firsts/matches:.1%}')
    buckets = [(0,.15),(.15,.30),(.30,.45),(.45,.60),(.60,.80),(.80,1.01)]
    print('\n  predicted -> actual win-rate (bucket count). gap = pred - actual')
    for lo, hi in buckets:
        in_b = [(d, w) for d, w in all_decisions if lo <= d['pred_p1st'] < hi]
        if not in_b: continue
        avg_pred = sum(d['pred_p1st'] for d, _ in in_b)/len(in_b)
        actual = sum(1 for _, w in in_b if w)/len(in_b)
        gap = avg_pred - actual
        flag = ' OVERCONFIDENT' if gap > 0.10 else (' underconfident' if gap < -0.10 else '')
        print(f'  [{lo:.2f},{hi:.2f}) pred {avg_pred:.2f} actual {actual:.2f} gap {gap:+.2f} n={len(in_b)}{flag}')
    print('\n  by chosen action:')
    for kind in ['fold','check','call','bet','raise']:
        sub = [(d, w) for d, w in all_decisions if d['chosen_kind'] == kind]
        if not sub: continue
        avg_pred = sum(d['pred_p1st'] for d, _ in sub)/len(sub)
        actual = sum(1 for _, w in sub if w)/len(sub)
        print(f'  {kind:6s} pred {avg_pred:.2f} actual {actual:.2f} n={len(sub)} gap {avg_pred-actual:+.2f}')
    print('\n  by street:')
    for st_name, st in [('preflop',0),('flop',1),('turn',2),('river',3)]:
        sub = [(d, w) for d, w in all_decisions if d['street'] == st]
        if not sub: continue
        avg_pred = sum(d['pred_p1st'] for d, _ in sub)/len(sub)
        actual = sum(1 for _, w in sub if w)/len(sub)
        print(f'  {st_name:8s} pred {avg_pred:.2f} actual {actual:.2f} n={len(sub)} gap {avg_pred-actual:+.2f}')
    return all_decisions

def confirm(matches=120, n=6, seeds=(42,137), total_hands=10, **kw):
    """run N matches split across seeds, combine. the experiment from last round."""
    import math
    per = matches // len(seeds)
    all_pl = []; firsts_total = 0
    for s in seeds:
        r = winrate(matches=per, n=n, seed=s, total_hands=total_hands,
                    progress_every=10, label=f'seed{s}', return_full=True, **kw)
        firsts_total += r['firsts']; all_pl += r['raw_placings']
    N = len(all_pl); wr = firsts_total/N
    se = math.sqrt(wr*(1-wr)/N)
    print(f'\nCOMBINED N={N} h={total_hands}: 1st {firsts_total}/{N} = {wr:.1%} '
          f'CI[{(wr-1.96*se)*100:.1f}%,{(wr+1.96*se)*100:.1f}%]  base {1/n:.1%}')
    print('placings:', {k: all_pl.count(k) for k in range(1,n+1)})
    return dict(wr=wr, ci_lo=wr-1.96*se, ci_hi=wr+1.96*se, n=N)

if __name__ == '__main__':
    # light smoke test - confirms engine + priors + bot wire up. ~30-60s.
    # for real experiments, write a driver script that imports sim and calls
    # measure_priors / winrate / confirm with the knobs you want.
    print('=== sim smoke test ===')
    print('[1] priors (engine + chip conservation, small sample):')
    measure_priors(matches=40)
    print('\n[2] winrate quick sanity (N=15, horizon 10):')
    winrate(matches=15, total_hands=10, seed=0, label='smoke')
    print('\nfor real runs, see drivers/ or write your own:')
    print('  import sim; sim.confirm(matches=120, total_hands=10)')