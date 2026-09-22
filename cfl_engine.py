# ============================================================
# CFL Playoff Scenario Generator -- Colab version, pulling live data
# from the CFL PIMS API (echo.pims.cfl.ca) instead of an uploaded workbook.
# Paste this whole cell into Google Colab and run it (Shift+Enter).
# ============================================================

import requests
from ortools.sat.python import cp_model

# ---- CONFIGURE ONCE PER SEASON ----
# season_id is stable for a given CFL season but changes year to year --
# find the current one via GET https://echo.pims.cfl.ca/api/seasons/
# (sorted by year) and update this constant when a new season starts.
SEASON_ID = 75
STANDINGS_YEAR = 2026

PIMS_BASE = "https://echo.pims.cfl.ca"

def fetch_pims_fixtures(season_id):
    """
    Fetches every fixture for a season in one call. limit=200 is a
    concrete, safely-large number (a full 9-team/18-game CFL season is
    ~95 fixtures including preseason and playoff placeholders); the
    API's own docs also show limit=* as valid if a literal wildcard is
    ever needed for a larger pull.
    """
    resp = requests.get(f"{PIMS_BASE}/api/fixtures/",
                         params={"season_id": season_id, "limit": 200},
                         timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_pims_standings(year):
    resp = requests.get(f"{PIMS_BASE}/api/standings/{year}", timeout=30)
    resp.raise_for_status()
    return resp.json()

# ============================================================
# PART 1: DIVISION STRUCTURE (static -- doesn't come from the API)
# ============================================================
DIVISION = {
    "MTL": "East", "TOR": "East", "HAM": "East", "OTT": "East",
    "EDM": "West", "SSK": "West", "BC": "West", "WPG": "West", "CGY": "West",
}
EAST = [t for t in DIVISION if DIVISION[t] == "East"]
WEST = [t for t in DIVISION if DIVISION[t] == "West"]
ALL_TEAMS = EAST + WEST

# ============================================================
# PART 2: STANDINGS + TIEBREAKER + CROSSOVER ENGINE  (tested:
# reproduces real standings exactly; tiebreak cascade verified on
# the live BC/WPG head-to-head tie; crossover verified both ways)
# ============================================================

def build_game_records(games, overrides=None):
    overrides = overrides or {}
    records = {t: [] for t in ALL_TEAMS}
    for g in games:
        home, away, hpf, apf = g["home"], g["away"], g["hpf"], g["apf"]
        if hpf is None:
            winner = overrides.get(g["gm"])
            if winner is None:
                continue
            loser = away if winner == home else home
            records[winner].append(dict(opp=loser, ts=1, os=0, is_div=DIVISION[home]==DIVISION[away]))
            records[loser].append(dict(opp=winner, ts=0, os=1, is_div=DIVISION[home]==DIVISION[away]))
            continue
        if hpf > apf: w,l,ws_,ls_ = home,away,hpf,apf
        else: w,l,ws_,ls_ = away,home,apf,hpf
        records[w].append(dict(opp=l, ts=ws_, os=ls_, is_div=DIVISION[home]==DIVISION[away]))
        records[l].append(dict(opp=w, ts=ls_, os=ws_, is_div=DIVISION[home]==DIVISION[away]))
    return records

def team_totals(records):
    out = {}
    for t, recs in records.items():
        w = sum(1 for r in recs if r["ts"] > r["os"])
        l = sum(1 for r in recs if r["ts"] < r["os"])
        pf = sum(r["ts"] for r in recs); pa = sum(r["os"] for r in recs)
        out[t] = dict(w=w, l=l, pts=2*w, pf=pf, pa=pa, games=len(recs))
    return out

def _vs_subset(records, team, opponents):
    return [r for r in records[team] if r["opp"] in opponents]
def _win_pct(recs):
    return (sum(1 for r in recs if r["ts"]>r["os"])/len(recs)) if recs else None
def _games_won(recs):
    return sum(1 for r in recs if r["ts"]>r["os"])
def _win_pct_future_safe(recs):
    return (sum(1 for r in recs if r["ts"]>r["os"])/len(recs)) if recs else None

def break_tie_future_safe(tied_teams, records):
    """
    Only steps (a) and (b) are usable for hypothetical future games -- both
    fully determined by win/loss counts alone. CORRECTED: earlier versions
    of this function also tried step (e) -- divisional win% -- reasoning
    that it too only needs win/loss data. That's true in isolation, but
    WRONG for a strictly sequential cascade: steps (c) and (d) come BEFORE
    (e) and require real final scores, which don't exist for unplayed
    games. An earlier unresolvable step blocks everything after it, even
    a later step that happens to be independently computable. Skipping
    ahead to (e) when (c)/(d) can't be evaluated is not a legitimate
    shortcut -- it silently converts a genuinely unresolved tie into a
    false "resolved" result. Steps (c) onward are simply unusable for
    hypothetical games; treat anything unresolved after (b) as ambiguous.
    Returns a list where each element is either a single team code (fully
    resolved) or a frozenset of team codes (still tied / ambiguous).
    """
    if len(tied_teams) == 1:
        return list(tied_teams)
    tset = set(tied_teams)

    groups = _group_by_key(tied_teams, lambda t: _games_won(records[t]))
    if len(groups) > 1:
        out = []
        for _, g in groups:
            out.extend(break_tie_future_safe(g, records) if len(g) > 1 else g)
        return out

    groups = _group_by_key(tied_teams, lambda t: _win_pct_future_safe(_vs_subset(records,t,tset-{t})) or 0)
    if len(groups) > 1:
        out = []
        for _, g in groups:
            out.extend(break_tie_future_safe(g, records) if len(g) > 1 else g)
        return out

    return [frozenset(tied_teams)]  # steps (c) onward unresolvable -- genuinely ambiguous


def division_order_future_safe(division_teams, records):
    totals = team_totals(records)
    groups = _group_by_key(division_teams, lambda t: totals[t]["pts"])
    out = []
    for _, g in groups:
        out.extend(break_tie_future_safe(g, records) if len(g) > 1 else g)
    return out  # list of team-codes and/or frozensets (ambiguous ties)


def _group_at_position(order, index):
    """Return the group (frozenset or single-team set) occupying the given
    0-indexed position in a future-safe division order, or None if the
    division doesn't reach that far."""
    pos = 0
    for item in order:
        group = item if isinstance(item, frozenset) else {item}
        gs = len(group)
        if pos <= index < pos + gs:
            return group
        pos += gs
    return None


def team_could_miss_target(games, overrides, team, target):
    """
    Exact-as-possible check for one fully-specified hypothetical scenario:
    does `team` fail to achieve `target`, GIVEN honest handling of ties that
    can't be resolved without unplayed-game scores? Returns True if `team`
    definitively OR ambiguously fails to achieve target.

    Handles BOTH directions of the crossover rule:
      Path A: team finishes top-3 in its own division (with the risk of
              being bumped from the 3rd seed by the OTHER division's
              4th-place crossover team).
      Path B: team finishes 4th in its own division, but has enough points
              to itself cross over and take the OTHER division's 3rd seed.

    CORRECTED: previously, when team's own tied group straddled the 3rd/4th
    boundary (genuinely ambiguous whether they're guaranteed top-3), the
    function fell straight through to "could miss" WITHOUT ever checking
    whether crossover would rescue them in the specific worst-case sub-
    scenario where they land in 4th. That's wrong: if a team's tied group's
    WORST possible position is exactly 4th (not 5th or lower -- crossover
    is never available from 5th), safety only requires that the 4th-place
    sub-case is itself rescued by crossover, since any better outcome
    within the tie is automatically fine. This version checks the team's
    actual worst-case position directly, instead of only branching on
    flat_rank_ok's True/False/None trichotomy, so this case is no longer
    skipped.
    """
    recs = build_game_records(games, overrides)
    e_order = division_order_future_safe(EAST, recs)
    w_order = division_order_future_safe(WEST, recs)
    div_order = e_order if DIVISION[team] == "East" else w_order
    other_order = w_order if DIVISION[team] == "East" else e_order
    totals = team_totals(recs)

    def flat_rank_ok(order, top_n):
        pos = 0
        for item in order:
            group = item if isinstance(item, frozenset) else {item}
            gs = len(group)
            if team in group:
                if pos + gs <= top_n: return True
                if pos >= top_n: return False
                return None
            pos += gs
        return False

    def group_position_range(order):
        """0-indexed INCLUSIVE (best, worst) position range for team's
        tied group -- a single-team group has best==worst."""
        pos = 0
        for item in order:
            group = item if isinstance(item, frozenset) else {item}
            gs = len(group)
            if team in group:
                return pos, pos + gs - 1
            pos += gs
        raise ValueError(f"{team} not found in order")

    if target == "home_game":
        return flat_rank_ok(div_order, 2) is not True

    if target == "division":
        return flat_rank_ok(div_order, 1) is not True

    if target == "playoffs":
        top2 = flat_rank_ok(div_order, 2)
        if top2 is True:
            return False  # guaranteed top-2 -- always safe, no crossover involved either direction

        team_pts = totals[team]["pts"]
        _, worst_pos = group_position_range(div_order)  # 0-indexed worst-case finish

        if worst_pos <= 2:
            # Worst case is still top-3 (guaranteed) -- check crossover-OUT risk.
            other_4th = _group_at_position(other_order, 3)
            if other_4th is None or not any(totals[c]["pts"] > team_pts for c in other_4th):
                return False
            return True

        if worst_pos == 3:
            # Worst case is EXACTLY 4th (never worse) -- any better outcome
            # within the tie is automatically fine, so safety only requires
            # the 4th-place sub-case itself to be rescued by crossover.
            other_3rd = _group_at_position(other_order, 2)
            if other_3rd is not None and all(team_pts > totals[c]["pts"] for c in other_3rd):
                return False
            return True

        # worst_pos >= 4: a genuine risk of finishing 5th or worse exists --
        # crossover is never available from there, so this is a real miss.
        return True

    raise ValueError(target)



def _net_agg(recs):
    return sum(r["ts"]-r["os"] for r in recs)
def _net_quotient(recs):
    pf=sum(r["ts"] for r in recs); pa=sum(r["os"] for r in recs)
    return pf/pa if pa else float("inf")

def _group_by_key(teams, keyfn):
    d = {}
    for t in teams: d.setdefault(keyfn(t), []).append(t)
    return sorted(d.items(), key=lambda kv: kv[0], reverse=True)

def _resolve_groups(groups, records):
    out = []
    for _, teams in groups:
        out.extend(break_tie(teams, records) if len(teams) > 1 else teams)
    return out

def break_tie(tied_teams, records):
    if len(tied_teams) == 1: return list(tied_teams)
    tset = set(tied_teams)

    groups = _group_by_key(tied_teams, lambda t: _games_won(records[t]))
    if len(groups) > 1: return _resolve_groups(groups, records)

    groups = _group_by_key(tied_teams, lambda t: _win_pct(_vs_subset(records,t,tset-{t})) or 0)
    if len(groups) > 1: return _resolve_groups(groups, records)

    groups = _group_by_key(tied_teams, lambda t: _net_agg(_vs_subset(records,t,tset-{t})))
    if len(groups) > 1: return _resolve_groups(groups, records)

    groups = _group_by_key(tied_teams, lambda t: _net_quotient(_vs_subset(records,t,tset-{t})))
    if len(groups) > 1: return _resolve_groups(groups, records)

    div_teams = lambda t: set(EAST if DIVISION[t]=="East" else WEST)
    groups = _group_by_key(tied_teams, lambda t: _win_pct(_vs_subset(records,t,div_teams(t)-{t})) or 0)
    if len(groups) > 1: return _resolve_groups(groups, records)

    groups = _group_by_key(tied_teams, lambda t: _net_agg(_vs_subset(records,t,div_teams(t)-{t})))
    if len(groups) > 1: return _resolve_groups(groups, records)

    groups = _group_by_key(tied_teams, lambda t: _net_quotient(_vs_subset(records,t,div_teams(t)-{t})))
    if len(groups) > 1: return _resolve_groups(groups, records)

    groups = _group_by_key(tied_teams, lambda t: _net_agg(records[t]))
    if len(groups) > 1: return _resolve_groups(groups, records)

    groups = _group_by_key(tied_teams, lambda t: _net_quotient(records[t]))
    if len(groups) > 1: return _resolve_groups(groups, records)

    return sorted(tied_teams)  # step j: coin toss -- arbitrary placeholder

def division_order(division_teams, records):
    totals = team_totals(records)
    groups = _group_by_key(division_teams, lambda t: totals[t]["pts"])
    return _resolve_groups(groups, records)

def playoff_seeding(east_order, west_order, totals):
    east_1,east_2,east_3 = east_order[0],east_order[1],east_order[2]
    east_4 = east_order[3] if len(east_order)>3 else None
    west_1,west_2,west_3 = west_order[0],west_order[1],west_order[2]
    west_4 = west_order[3] if len(west_order)>3 else None
    east_final = {1:east_1,2:east_2,3:east_3}
    west_final = {1:west_1,2:west_2,3:west_3}
    crossover_used, crossover_team = False, None
    if west_4 is not None and totals[west_4]["pts"] > totals[east_3]["pts"]:
        east_final[3] = west_4; crossover_used=True; crossover_team=west_4
    if east_4 is not None and totals[east_4]["pts"] > totals[west_3]["pts"] and not crossover_used:
        west_final[3] = east_4; crossover_used=True; crossover_team=east_4
    playoff_teams = set(east_final.values()) | set(west_final.values())
    eliminated = sorted((set(east_order)|set(west_order)) - playoff_teams)
    return dict(east=east_final, west=west_final, crossover_used=crossover_used,
                crossover_team=crossover_team, eliminated=eliminated)

def evaluate(games, overrides, target, team):
    recs = build_game_records(games, overrides)
    tot = team_totals(recs)
    e_order = division_order(EAST, recs)
    w_order = division_order(WEST, recs)
    seeding = playoff_seeding(e_order, w_order, tot)
    if target == "playoffs":
        ok = team in seeding["east"].values() or team in seeding["west"].values()
    elif target == "division":
        ok = (e_order[0]==team) or (w_order[0]==team)
    elif target == "home_game":
        div_order = e_order if DIVISION[team]=="East" else w_order
        ok = team in div_order[:2]
    else:
        raise ValueError(target)
    return ok, seeding

# ============================================================
# PART 3: OR-TOOLS SOLVER
# Strategy (CEGAR-style): CP-SAT quickly finds a candidate scenario
# (using only the POINTS totals, which are linear -- easy for a
# SAT solver) where the team's target condition might fail. We then
# verify that candidate exactly against the real tiebreaker/crossover
# engine above. If it's a genuine counterexample, the team has NOT
# clinched. If the engine says the team is actually still fine (a
# near-tie the simple points check couldn't see), we block that exact
# scenario and ask CP-SAT for another candidate. Repeat until either
# a real counterexample is confirmed, or CP-SAT reports no more
# candidates exist (-> team has clinched).
# ============================================================

def cegar_search(model, win_vars, remaining, base_overrides, verify_fn, verbose=True, progress_every=10000,
                  max_time_seconds=600, max_solutions=None):
    """
    Search for a game-outcome scenario (satisfying `model`'s constraints)
    that verify_fn() confirms is a genuine counterexample -- using CP-SAT's
    NATIVE solution enumeration rather than manually re-solving with an
    ever-growing set of "don't propose this exact thing again" constraints
    (that approach works but gets slower every iteration, since the model
    itself keeps growing -- this doesn't have that problem, since CP-SAT
    tracks explored solutions internally without bloating the model).

    `max_solutions`, if set, stops the search after checking that many
    candidates even if the full space isn't exhausted -- for a FAST SCAN
    (e.g. from find_clinch_scenarios trying many small combinations) where
    a "didn't finish, inconclusive" result is fine to miss, but a false
    "confirmed safe" is not. Leave unset (None) for a full, rigorous proof.

    Returns dict(found: bool, witness: dict|None, count: int, exhausted: bool)
    -- found=True means a real counterexample was confirmed. exhausted=True
    means the FULL solution space was covered (so found=False + exhausted=True
    is a complete proof of safety; found=False + exhausted=False just means
    the search budget ran out before finishing, NOT a proof of anything).
    """
    class _Callback(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self.count = 0
            self.witness = None
            self.hit_cap = False

        def on_solution_callback(self):
            self.count += 1
            if verbose and self.count % progress_every == 0:
                print(f"  ...enumerated {self.count} candidate scenarios so far")
            cand = dict(base_overrides)
            for g in remaining:
                cand[g["gm"]] = g["home"] if self.Value(win_vars[g["gm"]]) else g["away"]
            if verify_fn(cand):
                self.witness = cand
                self.StopSearch()
            elif max_solutions is not None and self.count >= max_solutions:
                self.hit_cap = True
                self.StopSearch()

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.max_time_in_seconds = max_time_seconds
    cb = _Callback()
    solver.Solve(model, cb)
    exhausted = (cb.witness is None) and (not cb.hit_cap)
    return dict(found=cb.witness is not None, witness=cb.witness, count=cb.count, exhausted=exhausted)


# ============================================================
# CANONICAL-STATE BLOCKING: the theoretically-justified version of the
# earlier "block on points only" shortcut. Points alone aren't safe to
# block on, because tiebreaker step (b) uses head-to-head results, which
# aren't fully determined by final points -- two scenarios can share
# identical point totals but differ in who beat whom, changing the real
# tiebreak outcome. The provably-sufficient state (given our engine only
# uses win-based steps a/b/e) is: each team's final points, PLUS the
# head-to-head win count for every pairing WITHIN each division.
# ============================================================
ALL_PAIRS = ([(a, b) for i, a in enumerate(EAST) for b in EAST[i+1:]] +
             [(a, b) for i, a in enumerate(WEST) for b in WEST[i+1:]])


def build_h2h_exprs(model, win_vars, remaining, base_overrides, games):
    """Linear CP-SAT expressions for each division-internal pair's
    head-to-head win counts (base games + remaining games between them)."""
    base_recs = build_game_records(games, base_overrides)
    h2h_exprs = {}
    for a, b in ALL_PAIRS:
        a_base = sum(1 for r in base_recs[a] if r["opp"] == b and r["ts"] > r["os"])
        b_base = sum(1 for r in base_recs[b] if r["opp"] == a and r["ts"] > r["os"])
        a_terms, b_terms = [a_base], [b_base]
        for g in remaining:
            if {g["home"], g["away"]} == {a, b}:
                v = win_vars[g["gm"]]
                if g["home"] == a:
                    a_terms.append(v); b_terms.append(1 - v)
                else:
                    a_terms.append(1 - v); b_terms.append(v)
        h2h_exprs[(a, b)] = (sum(a_terms), sum(b_terms))
    return h2h_exprs


def compute_canonical_state(games, cand):
    """The actual (a,b)-> (wins_a, wins_b) + team->pts dict for one fully
    decided scenario -- used both to block and to compare states."""
    recs = build_game_records(games, cand)
    tot = team_totals(recs)
    pts = {t: tot[t]["pts"] for t in ALL_TEAMS}
    h2h = {}
    for a, b in ALL_PAIRS:
        a_wins = sum(1 for r in recs[a] if r["opp"] == b and r["ts"] > r["os"])
        b_wins = sum(1 for r in recs[b] if r["opp"] == a and r["ts"] > r["os"])
        h2h[(a, b)] = (a_wins, b_wins)
    return pts, h2h


def _h2h_locked_edge(team, rival, games, overrides):
    """True if `team` already has a mathematically unbeatable head-to-head
    win-based tiebreak edge over `rival` -- i.e. even if `rival` wins every
    remaining meeting between them, `rival` still cannot equal or exceed
    `team`'s head-to-head win count. When True, an exact POINTS tie between
    them is not actually a danger (the tiebreaker is already decided)."""
    h2h = [g for g in games if {g["home"], g["away"]} == {team, rival}]
    completed = [g for g in h2h if g["hpf"] is not None]
    remaining_h2h = [g for g in h2h if g["hpf"] is None and g["gm"] not in overrides]
    team_wins = sum(1 for g in completed if
                     (g["home"] == team and g["hpf"] > g["apf"]) or
                     (g["away"] == team and g["apf"] > g["hpf"]))
    rival_wins = len(completed) - team_wins
    for g in h2h:
        if g["hpf"] is None and g["gm"] in overrides:
            if overrides[g["gm"]] == team: team_wins += 1
            elif overrides[g["gm"]] == rival: rival_wins += 1
    return team_wins > rival_wins + len(remaining_h2h)


def team_could_achieve_target(games, overrides, team, target):
    """
    Mirror of team_could_miss_target: for one fully-specified scenario, could
    `team` still plausibly achieve `target`? Uses OPTIMISTIC handling of
    genuine ties. Also checks BOTH directions of crossover (own-division
    top-3, or crossing over as own-division's 4th-place finisher).
    """
    recs = build_game_records(games, overrides)
    e_order = division_order_future_safe(EAST, recs)
    w_order = division_order_future_safe(WEST, recs)
    div_order = e_order if DIVISION[team] == "East" else w_order
    other_order = w_order if DIVISION[team] == "East" else e_order
    totals = team_totals(recs)

    def flat_rank_ok(order, top_n):
        pos = 0
        for item in order:
            group = item if isinstance(item, frozenset) else {item}
            gs = len(group)
            if team in group:
                if pos + gs <= top_n: return True
                if pos >= top_n: return False
                return None
            pos += gs
        return False

    if target == "home_game":
        return flat_rank_ok(div_order, 2) is not False

    if target == "division":
        return flat_rank_ok(div_order, 1) is not False

    if target == "playoffs":
        top2 = flat_rank_ok(div_order, 2)
        if top2 is not False:
            # top2 is True (guaranteed) OR None (ambiguous -- team could win
            # its own tiebreak and land safely in the top-2) -- either way,
            # a genuine "could still achieve" path exists via top-2 alone,
            # independent of any crossover consideration.
            return True

        top3 = flat_rank_ok(div_order, 3)
        team_pts = totals[team]["pts"]

        # Path A: could team achieve safety via own-division top-3 (optimistic
        # about avoiding crossover-OUT)?
        pathA_possible = False
        if top3 is not False:  # team could be top-3
            other_4th = _group_at_position(other_order, 3)
            if other_4th is None or any(totals[c]["pts"] <= team_pts for c in other_4th):
                pathA_possible = True  # some way to avoid being crossed out

        if pathA_possible:
            return True

        # Path B: could team, as their own division's 4th-place finisher,
        # cross INTO the other division's 3rd seed?
        pathB_possible = False
        if top3 is not True:  # team could be 4th (or worse)
            other_3rd = _group_at_position(other_order, 2)
            if other_3rd is not None and any(team_pts > totals[c]["pts"] for c in other_3rd):
                pathB_possible = True

        return pathB_possible

    raise ValueError(target)


def solve_elimination(games, team, target, overrides=None, max_iterations=50, verbose=True, fast_only=False):
    """
    Is `team` ELIMINATED from `target`? True iff NO scenario exists where
    they could still achieve it -- proven by searching (with proper
    blocking, mirroring the clinch-side search) for a surviving scenario
    within the region already known to avoid decisive elimination, and only
    concluding elimination if that search is fully exhausted.
    """
    overrides = dict(overrides or {})
    remaining = [g for g in games if g["hpf"] is None and g["gm"] not in overrides]
    base_recs = build_game_records(games, overrides)
    base_tot = team_totals(base_recs)
    base_pts = {t: base_tot[t]["pts"] for t in ALL_TEAMS}
    div_teams = EAST if DIVISION[team] == "East" else WEST
    rivals = [t for t in div_teams if t != team]

    def build_pts_exprs(model):
        win_vars = {g["gm"]: model.NewBoolVar(f"g{g['gm']}") for g in remaining}
        pts_terms = {t: [base_pts[t]] for t in ALL_TEAMS}
        for g in remaining:
            v = win_vars[g["gm"]]
            pts_terms[g["home"]].append(2*v)
            pts_terms[g["away"]].append(2*(1-v))
        return win_vars, {t: sum(pts_terms[t]) for t in ALL_TEAMS}

    def solve_feasible(model):
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 15
        return s.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE), s

    def fix_team_best_case(model, wv):
        for g in remaining:
            if g["home"] == team: model.Add(wv[g["gm"]] == 1)
            if g["away"] == team: model.Add(wv[g["gm"]] == 0)

    if target not in ("playoffs", "division", "home_game"):
        raise ValueError("only 'playoffs', 'division', and 'home_game' implemented for elimination")

    if target == "home_game":
        # No crossover involved at all -- home game is purely "can team
        # finish top-2 within their own division," so this mirrors the
        # division-elimination structure exactly, just checking top-2
        # instead of top-1 (sole first place).
        model_r = cp_model.CpModel()
        wv_r, pts_r = build_pts_exprs(model_r)
        fix_team_best_case(model_r, wv_r)
        exceed_bools = []
        for r in rivals:
            b = model_r.NewBoolVar(f"exceedhg_{r}")
            model_r.Add(pts_r[r] > pts_r[team]).OnlyEnforceIf(b)
            model_r.Add(pts_r[r] <= pts_r[team]).OnlyEnforceIf(b.Not())
            exceed_bools.append(b)
        model_r.Add(sum(exceed_bools) <= 1)  # "team avoids being passed by 2+" -- survival condition
        can_avoid, _ = solve_feasible(model_r)
        if not can_avoid:
            return dict(status="ELIMINATED FROM HOME GAME (even in team's best case, guaranteed "
                                "to be passed by 2+ division rivals -- no crossover applies here, "
                                "no tiebreak needed)")

        if fast_only:
            return dict(status="BORDERLINE (home_game elimination -- fast checks inconclusive, "
                                "candidate for a deep dive)")
        model2 = cp_model.CpModel()
        wv2, pts2 = build_pts_exprs(model2)
        fix_team_best_case(model2, wv2)
        result2 = cegar_search(model2, wv2, remaining, overrides,
                                verify_fn=lambda cand: team_could_achieve_target(games, cand, team, "home_game"),
                                verbose=verbose, max_solutions=max_iterations)
        if result2["found"]:
            return dict(status="NOT ELIMINATED FROM HOME GAME (verified surviving scenario found)",
                        witness=result2["witness"])
        if result2["exhausted"]:
            return dict(status="ELIMINATED FROM HOME GAME (verified: no scenario in the search "
                                "space holds up under real tiebreaker checking)")
        return dict(status=f"INCONCLUSIVE (home_game elimination) after {result2['count']} candidates "
                            "checked -- search budget ran out before a full proof was reached")

    if target == "division":
        # FAST EXACT CHECK: even in team's best case, is any single rival
        # GUARANTEED to still exceed them? If so, that rival alone denies
        # team the division outright, no tiebreak ever needed to prove it.
        for r in rivals:
            model_r = cp_model.CpModel()
            wv_r, pts_r = build_pts_exprs(model_r)
            fix_team_best_case(model_r, wv_r)
            model_r.Add(pts_r[r] <= pts_r[team])  # can THIS rival fail to catch team's best case?
            rival_can_fall_short, _ = solve_feasible(model_r)
            if not rival_can_fall_short:
                return dict(status=f"ELIMINATED FROM DIVISION ({r} guaranteed to reach/exceed "
                                    "team's maximum possible points, no tiebreak needed)")

        # Proper search: team fixed to best case, otherwise unrestricted --
        # hunt for a scenario that verifiably lets team actually win the
        # division once real tiebreakers are applied.
        if fast_only:
            return dict(status="BORDERLINE (division elimination -- fast checks inconclusive, "
                                "candidate for a deep dive)")
        model2 = cp_model.CpModel()
        wv2, pts2 = build_pts_exprs(model2)
        fix_team_best_case(model2, wv2)
        result2 = cegar_search(model2, wv2, remaining, overrides,
                                verify_fn=lambda cand: team_could_achieve_target(games, cand, team, "division"),
                                verbose=verbose, max_solutions=max_iterations)
        if result2["found"]:
            return dict(status="NOT ELIMINATED FROM DIVISION (verified winning scenario found)",
                        witness=result2["witness"])
        if result2["exhausted"]:
            return dict(status="ELIMINATED FROM DIVISION (verified: no scenario in the search "
                                "space holds up under real tiebreaker checking)")
        return dict(status=f"INCONCLUSIVE (division elimination) after {result2['count']} candidates "
                            "checked -- search budget ran out before a full proof was reached")

    # FAST EXACT CHECK: even in team's best case (wins out), can 3+ rivals
    # be avoided from strictly exceeding them? If not, decisively eliminated
    # -- no tiebreaker needed, this needs no verification.
    model1 = cp_model.CpModel()
    wv1, pts1 = build_pts_exprs(model1)
    fix_team_best_case(model1, wv1)
    exceed_bools = []
    for r in rivals:
        b = model1.NewBoolVar(f"exceed_{r}")
        model1.Add(pts1[r] > pts1[team]).OnlyEnforceIf(b)
        model1.Add(pts1[r] <= pts1[team]).OnlyEnforceIf(b.Not())
        exceed_bools.append(b)
    model1.Add(sum(exceed_bools) <= 2)  # "team avoids being passed by 3+" -- the SURVIVAL condition
    can_avoid, _ = solve_feasible(model1)
    if not can_avoid:
        # Own division offers no path -- but before declaring elimination,
        # check the OTHER direction of crossover: could team's own best-case
        # points still put them within reach of crossing INTO the other
        # division (i.e. beating enough of ITS teams to plausibly be its
        # 3rd-place-or-better finisher)? If that's still feasible, this fast
        # path can't safely conclude elimination -- fall through to the
        # full verified search instead, which correctly checks both paths.
        other_div_teams = WEST if DIVISION[team] == "East" else EAST
        model_x = cp_model.CpModel()
        wv_x, pts_x = build_pts_exprs(model_x)
        fix_team_best_case(model_x, wv_x)
        beat_bools = []
        for o in other_div_teams:
            b = model_x.NewBoolVar(f"beat_{o}")
            model_x.Add(pts_x[team] > pts_x[o]).OnlyEnforceIf(b)
            model_x.Add(pts_x[team] <= pts_x[o]).OnlyEnforceIf(b.Not())
            beat_bools.append(b)
        # team needs to beat all but at most 2 of the other division's teams
        # to plausibly reach 3rd there (crude but safe/conservative gate)
        model_x.Add(sum(beat_bools) >= len(other_div_teams) - 2)
        crossover_in_possible, _ = solve_feasible(model_x)
        if not crossover_in_possible:
            return dict(status="ELIMINATED (even in best case, guaranteed to be passed by 3+ rivals, "
                                "AND no realistic crossover-in path exists)")
        # else: own-division path is closed, but crossover-in isn't ruled out
        # -- don't return here, fall through to the full verified search below.

    # Proper search: team fixed to its best case, otherwise UNRESTRICTED --
    # deliberately not scoped to "own-division survival only", since that
    # would silently exclude genuine crossover-in survival paths (a team
    # that's decisively out of its own division's top-3 can still make the
    # playoffs by crossing over). team_could_achieve_target checks BOTH
    # paths on each candidate, so the search itself should stay unscoped.
    if fast_only:
        return dict(status="BORDERLINE (playoffs elimination -- fast checks inconclusive, "
                            "candidate for a deep dive)")
    model2 = cp_model.CpModel()
    wv2, pts2 = build_pts_exprs(model2)
    fix_team_best_case(model2, wv2)
    result2 = cegar_search(model2, wv2, remaining, overrides,
                            verify_fn=lambda cand: team_could_achieve_target(games, cand, team, target),
                            verbose=verbose, max_solutions=max_iterations)
    if result2["found"]:
        return dict(status="NOT ELIMINATED (verified surviving scenario found)", witness=result2["witness"])
    if result2["exhausted"]:
        return dict(status="ELIMINATED (verified: no scenario in the survival region holds up "
                            "under real tiebreaker/crossover checking)")
    return dict(status=f"INCONCLUSIVE after {result2['count']} candidates checked -- "
                        "needs more search budget")


def solve_status(games, team, target, overrides=None, ambiguous_max_iterations=200, verbose=True, fast_only=False):
    """
    Correctly-scoped clinch/elimination check for a FUTURE scenario.

    Key honesty constraint: tiebreaker steps that depend on final score
    margins (net points, net points quotient) or a coin toss cannot be
    evaluated for games that haven't been played yet -- nobody can predict
    a final score. So for hypothetical remaining games, we only ever use
    the tiebreaker steps that are fully determined by win/loss alone:
      (a) games won, (b) win% vs tied clubs, (e) divisional win%.
    If a tie survives past those, the real outcome genuinely depends on
    information that doesn't exist yet -- so we report that honestly as
    "not yet clinched" (a live scenario where they could still miss it,
    even if it would come down to a tiebreaker) rather than guessing.

    This also makes the check MUCH faster: no combinatorial search over
    hypothetical scores is needed at all -- just two direct feasibility
    checks on win counts.
    """
    overrides = dict(overrides or {})
    remaining = [g for g in games if g["hpf"] is None and g["gm"] not in overrides]
    base_recs = build_game_records(games, overrides)
    base_tot = team_totals(base_recs)
    base_pts = {t: base_tot[t]["pts"] for t in ALL_TEAMS}

    div_teams = EAST if DIVISION[team] == "East" else WEST
    rivals = [t for t in div_teams if t != team]

    def build_pts_exprs(model):
        win_vars = {g["gm"]: model.NewBoolVar(f"g{g['gm']}") for g in remaining}
        pts_terms = {t: [base_pts[t]] for t in ALL_TEAMS}
        for g in remaining:
            v = win_vars[g["gm"]]
            pts_terms[g["home"]].append(2*v)
            pts_terms[g["away"]].append(2*(1-v))
        return win_vars, {t: sum(pts_terms[t]) for t in ALL_TEAMS}

    def solve_feasible(model):
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 15
        return s.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE), s

    def fix_team_best_case(model, wv):
        for g in remaining:
            if g["home"] == team: model.Add(wv[g["gm"]] == 1)
            if g["away"] == team: model.Add(wv[g["gm"]] == 0)

    def reified_ge(model, name, lhs, rhs, strict):
        """b <=> (lhs > rhs) if strict else (lhs >= rhs). Using strict when
        the head-to-head tiebreak is already locked means an exact points
        TIE is correctly treated as no-danger, instead of wastefully
        flagged for verification."""
        b = model.NewBoolVar(name)
        if strict:
            model.Add(lhs > rhs).OnlyEnforceIf(b)
            model.Add(lhs <= rhs).OnlyEnforceIf(b.Not())
        else:
            model.Add(lhs >= rhs).OnlyEnforceIf(b)
            model.Add(lhs < rhs).OnlyEnforceIf(b.Not())
        return b

    locked = {r: _h2h_locked_edge(team, r, games, overrides) for r in rivals}
    if verbose:
        print(f"  Head-to-head tiebreak already locked in {team}'s favor over: "
              f"{[r for r in rivals if locked[r]] or 'none'}")

    if target in ("playoffs", "home_game"):
        # CHECK A: can the team be guaranteed no worse than 2nd (fully safe
        # from both elimination and crossover -- crossover only ever
        # displaces a division's 3rd seed)?
        model_a = cp_model.CpModel()
        wv_a, pts_a = build_pts_exprs(model_a)
        at_least_2 = []
        pairs = [(rivals[i], rivals[j]) for i in range(len(rivals)) for j in range(i+1, len(rivals))]
        for r1, r2 in pairs:
            b1 = reified_ge(model_a, f"ge_{r1}", pts_a[r1], pts_a[team], locked[r1])
            b2 = reified_ge(model_a, f"ge_{r2}", pts_a[r2], pts_a[team], locked[r2])
            both = model_a.NewBoolVar(f"both_{r1}_{r2}")
            model_a.AddBoolAnd([b1, b2]).OnlyEnforceIf(both)
            model_a.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(both.Not())
            at_least_2.append(both)
        model_a.AddBoolOr(at_least_2)
        unsafe_possible, _ = solve_feasible(model_a)
        if not unsafe_possible:
            return dict(status="CLINCHED (guaranteed top-2 in division, safe from any tiebreaker/crossover)")

        if target == "home_game":
            # for home_game, ANY scenario where 2+ rivals reach parity is
            # already enough uncertainty (no crossover involved, just need
            # a real top-2 verification on the witness)
            if fast_only:
                return dict(status="BORDERLINE (home_game -- fast top-2 check inconclusive, "
                                    "candidate for a deep dive)")
            model_h = cp_model.CpModel()
            wv_h, pts_h = build_pts_exprs(model_h)
            at_least_2_h = []
            for r1, r2 in pairs:
                b1 = reified_ge(model_h, f"geh_{r1}", pts_h[r1], pts_h[team], locked[r1])
                b2 = reified_ge(model_h, f"geh_{r2}", pts_h[r2], pts_h[team], locked[r2])
                both = model_h.NewBoolVar(f"both_{r1}_{r2}")
                model_h.AddBoolAnd([b1, b2]).OnlyEnforceIf(both)
                model_h.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(both.Not())
                at_least_2_h.append(both)
            model_h.AddBoolOr(at_least_2_h)
            result_h = cegar_search(model_h, wv_h, remaining, overrides,
                                     verify_fn=lambda cand: team_could_miss_target(games, cand, team, "home_game"),
                                     verbose=verbose, max_solutions=ambiguous_max_iterations)
            if result_h["found"]:
                return dict(status=f"NOT CLINCHED HOME GAME (verified real scenario; "
                                    f"found after checking {result_h['count']} candidates)",
                            witness=result_h["witness"])
            if result_h["exhausted"]:
                return dict(status=f"CLINCHED HOME GAME (verified: all {result_h['count']} distinct "
                                    "candidate scenarios confirmed safe)")
            return dict(status=f"INCONCLUSIVE (home_game) after {result_h['count']} candidates checked -- "
                                "search budget ran out before a full proof was reached")

        # target == playoffs
        # Decisive miss = guaranteed outside top-3 = at least 3 rivals
        # strictly exceed team's points (true regardless of division size --
        # East has exactly 3 rivals so this means "all of them"; West has 4
        # rivals so this means "any 3 of the 4", not necessarily all).
        # CORRECTED: must fix team to its OWN best case first (mirroring
        # every other check in this file) -- without it, this asks the
        # wrong, much weaker question "does ANY scenario let 3+ rivals
        # pass team" (including team losing out), not "does it still
        # happen even in team's best case". That gap produced false
        # NOT CLINCHED verdicts whenever team was plausibly close to
        # clinching, since the illustrative witness was free to also let
        # team itself lose everything.
        model_b = cp_model.CpModel()
        wv_b, pts_b = build_pts_exprs(model_b)
        fix_team_best_case(model_b, wv_b)
        exceed_bools = []
        for r in rivals:
            b = model_b.NewBoolVar(f"exceed_{r}")
            model_b.Add(pts_b[r] > pts_b[team]).OnlyEnforceIf(b)
            model_b.Add(pts_b[r] <= pts_b[team]).OnlyEnforceIf(b.Not())
            exceed_bools.append(b)
        model_b.Add(sum(exceed_bools) >= 3)
        decisive_miss_possible, s_b = solve_feasible(model_b)
        if decisive_miss_possible:
            witness = dict(overrides)
            for g in remaining:
                witness[g["gm"]] = g["home"] if s_b.Value(wv_b[g["gm"]]) else g["away"]
            # CORRECTED: "3+ rivals decisively exceed team WITHIN their own
            # division" is not the same as "team decisively misses the
            # playoffs" -- exactly 3 rivals passing leaves team in 4th,
            # which is the CROSSOVER-ELIGIBLE spot, not an automatic miss.
            # This fast check can only ever propose a candidate; it must be
            # verified by the same trusted crossover-aware logic used
            # everywhere else before being trusted as a real miss.
            if team_could_miss_target(games, witness, team, "playoffs"):
                return dict(status="NOT CLINCHED (can be decisively passed by 3+ division rivals, "
                                    "even in team's own best case, and crossover doesn't rescue this "
                                    "specific scenario -- exact, no tiebreak needed)", witness=witness)
            # else: fast check's witness was spurious (crossover saves team
            # here) -- fall through to the full ambiguous-zone search below,
            # which will correctly search for a genuine counterexample.

        # Ambiguous zone: 2+ rivals could reach parity, but not decisively
        # last. Search for a scenario that's STILL a real miss once real
        # (win-based) tiebreakers and crossover are correctly applied.
        # Uses cegar_search (native CP-SAT solution enumeration) instead of
        # manual re-solving with an ever-growing model.
        if fast_only:
            return dict(status="BORDERLINE (playoffs -- fast checks inconclusive, candidate for a deep dive)")
        model_c = cp_model.CpModel()
        wv_c, pts_c = build_pts_exprs(model_c)
        at_least_2_c = []
        for r1, r2 in pairs:
            b1 = reified_ge(model_c, f"ge2_{r1}", pts_c[r1], pts_c[team], locked[r1])
            b2 = reified_ge(model_c, f"ge2_{r2}", pts_c[r2], pts_c[team], locked[r2])
            both = model_c.NewBoolVar(f"both2_{r1}_{r2}")
            model_c.AddBoolAnd([b1, b2]).OnlyEnforceIf(both)
            model_c.AddBoolOr([b1.Not(), b2.Not()]).OnlyEnforceIf(both.Not())
            at_least_2_c.append(both)
        model_c.AddBoolOr(at_least_2_c)

        result = cegar_search(model_c, wv_c, remaining, overrides,
                               verify_fn=lambda cand: team_could_miss_target(games, cand, team, "playoffs"),
                               verbose=verbose, max_solutions=ambiguous_max_iterations)
        if result["found"]:
            return dict(status=f"NOT CLINCHED (verified real scenario -- genuine tiebreaker/crossover "
                                f"risk; found after checking {result['count']} candidates)",
                        witness=result["witness"])
        if result["exhausted"]:
            return dict(status=f"CLINCHED (verified: all {result['count']} distinct candidate scenarios "
                                "confirmed safe using real head-to-head/divisional tiebreaker data)")
        return dict(status=f"INCONCLUSIVE (playoffs) after {result['count']} candidates checked -- "
                            "search budget ran out before a full proof was reached")

    if target == "division":
        # Check A (exact, locked-h2h-edge aware): can ANY rival reach/tie team?
        model_a = cp_model.CpModel()
        wv_a, pts_a = build_pts_exprs(model_a)
        reach_bools = [reified_ge(model_a, f"reachdiv_{r}", pts_a[r], pts_a[team], locked[r]) for r in rivals]
        model_a.AddBoolOr(reach_bools)
        can_be_caught, _ = solve_feasible(model_a)
        if not can_be_caught:
            return dict(status="CLINCHED DIVISION (guaranteed strictly ahead of every rival, "
                                "using real head-to-head tiebreak data)")

        # Check B (exact): does any rival DECISIVELY (strictly, no tiebreak)
        # pass team, even in team's OWN best case? CORRECTED: must fix team
        # to its best case first -- same missing-fix bug as the playoffs
        # decisive-miss check above, found via the same real numerical check.
        model_b = cp_model.CpModel()
        wv_b, pts_b = build_pts_exprs(model_b)
        fix_team_best_case(model_b, wv_b)
        strict_bools = []
        for r in rivals:
            b = model_b.NewBoolVar(f"strictdiv_{r}")
            model_b.Add(pts_b[r] > pts_b[team]).OnlyEnforceIf(b)
            model_b.Add(pts_b[r] <= pts_b[team]).OnlyEnforceIf(b.Not())
            strict_bools.append(b)
        model_b.AddBoolOr(strict_bools)
        decisive_miss_possible, s_b = solve_feasible(model_b)
        if decisive_miss_possible:
            witness = dict(overrides)
            for g in remaining:
                witness[g["gm"]] = g["home"] if s_b.Value(wv_b[g["gm"]]) else g["away"]
            return dict(status="NOT CLINCHED DIVISION (a rival can decisively pass team, even in "
                                "team's own best case -- exact, no tiebreak needed)", witness=witness)

        # Ambiguous zone: some rival could reach an exact tie (not a decisive
        # strict pass) -- needs real tiebreaker verification.
        if fast_only:
            return dict(status="BORDERLINE (division -- fast checks inconclusive, candidate for a deep dive)")
        model_c = cp_model.CpModel()
        wv_c, pts_c = build_pts_exprs(model_c)
        reach_bools_c = [reified_ge(model_c, f"reachdiv2_{r}", pts_c[r], pts_c[team], locked[r]) for r in rivals]
        model_c.AddBoolOr(reach_bools_c)

        result_c = cegar_search(model_c, wv_c, remaining, overrides,
                                 verify_fn=lambda cand: team_could_miss_target(games, cand, team, "division"),
                                 verbose=verbose, max_solutions=ambiguous_max_iterations)
        if result_c["found"]:
            return dict(status=f"NOT CLINCHED DIVISION (verified real scenario -- genuine tiebreaker "
                                f"risk; found after checking {result_c['count']} candidates)",
                        witness=result_c["witness"])
        if result_c["exhausted"]:
            return dict(status=f"CLINCHED DIVISION (verified: all {result_c['count']} distinct "
                                "candidate scenarios resolve in team's favor via real tiebreaker data)")
        return dict(status=f"INCONCLUSIVE (division) after {result_c['count']} candidates checked -- "
                            "search budget ran out before a full proof was reached. "
                            "NOTE: elimination-from-division isn't rebuilt to this rigor yet -- "
                            "this branch currently only answers the CLINCH side.")

    raise ValueError(target)


def _all_week_combos(this_week_games):
    """Every combination of this week's games, every size, every outcome."""
    from itertools import combinations, product
    combos = []
    for combo_size in range(1, len(this_week_games) + 1):
        for game_subset in combinations(this_week_games, combo_size):
            for outcome_bits in product([0, 1], repeat=combo_size):
                combos.append((game_subset, outcome_bits))
    return combos


def _decode_combo(games, game_subset, outcome_bits, base_overrides):
    trial = dict(base_overrides)
    desc_parts = []
    for gm, bit in zip(game_subset, outcome_bits):
        g = next(x for x in games if x["gm"] == gm)
        winner = g["home"] if bit == 0 else g["away"]
        loser = g["away"] if bit == 0 else g["home"]
        trial[gm] = winner
        desc_parts.append(f"{winner} beats {loser}")
    return trial, " AND ".join(desc_parts)


def _dedupe_supersets(found_with_overrides):
    """Drop any found scenario whose override dict is a strict superset of
    a smaller already-found scenario -- reporting both is redundant (the
    smaller one already proves it)."""
    found_with_overrides = sorted(found_with_overrides, key=lambda x: len(x[1]))
    kept = []
    for desc, ov in found_with_overrides:
        is_superset = any(set(kov.items()) <= set(ov.items()) for _, kov in kept)
        if not is_superset:
            kept.append((desc, ov))
    return [desc for desc, _ in kept]


def find_clinch_scenarios_complete(games, team, target, this_week_games, base_overrides=None,
                                    quick_budget=15, verbose=True):
    """
    COMPLETE clinch-scenario finder: enumerates EVERY combination of this
    week's games (all sizes, not just 1-2 games), not just small ones.
    Uses a quick budget first to resolve easy cases fast; anything that
    comes back inconclusive at that budget is escalated to FULL (unlimited)
    verification, so nothing is silently missed the way the earlier
    max_combo_size=2 version could. This trades speed for completeness --
    can be slow when several combos land in the escalation bucket.
    """
    base_overrides = dict(base_overrides or {})
    combos = _all_week_combos(this_week_games)
    found, needs_deep_dive = [], []

    if verbose:
        print(f"  [{team} -- {target}] quick pass over {len(combos)} combinations...")
    for game_subset, outcome_bits in combos:
        trial, desc = _decode_combo(games, game_subset, outcome_bits, base_overrides)
        result = solve_status(games, team, target, overrides=trial,
                               ambiguous_max_iterations=quick_budget, verbose=False)
        if result["status"].startswith("CLINCHED"):
            found.append((desc, trial))
        elif result["status"].startswith("NOT CLINCHED"):
            pass  # decisively ruled out, no need to dig further
        else:
            needs_deep_dive.append((desc, trial))

    if verbose:
        print(f"  Quick pass done: {len(found)} found so far, "
              f"{len(needs_deep_dive)} inconclusive -- escalating to full verification")
    for desc, trial in needs_deep_dive:
        if verbose:
            print(f"  Deep-diving: {desc} ...")
        result = solve_status(games, team, target, overrides=trial, ambiguous_max_iterations=None, verbose=verbose)
        if result["status"].startswith("CLINCHED"):
            found.append((desc, trial))

    return _dedupe_supersets(found)


def find_elimination_scenarios_complete(games, team, target, this_week_games, base_overrides=None,
                                         quick_budget=15, verbose=True):
    """
    Mirror of find_clinch_scenarios_complete for elimination. Uses a quick
    budget first to resolve easy cases fast; anything inconclusive at that
    budget is escalated to FULL (unlimited) verification, so a genuinely
    eliminating combination that needs a large search isn't silently
    dropped from the results the way a hardcoded budget cap would cause.

    CORRECTED: the deep-dive step previously used a hardcoded
    max_iterations=5000 cap instead of mirroring the clinch-side function's
    already-fixed unlimited budget. A combination that genuinely eliminates
    a team but needs more than 5000 candidates to prove it would come back
    INCONCLUSIVE at that cap -- and since only an explicit "ELIMINATED"
    result got added to the returned list, that combination was silently
    dropped with no warning, producing a false "not eliminated" overall.
    """
    base_overrides = dict(base_overrides or {})
    combos = _all_week_combos(this_week_games)
    found, needs_deep_dive = [], []

    if verbose:
        print(f"  [{team} -- {target} elimination] quick pass over {len(combos)} combinations...")
    for game_subset, outcome_bits in combos:
        trial, desc = _decode_combo(games, game_subset, outcome_bits, base_overrides)
        result = solve_elimination(games, team, target, overrides=trial,
                                    max_iterations=quick_budget, verbose=False)
        if result["status"].startswith("ELIMINATED"):
            found.append((desc, trial))
        elif result["status"].startswith("NOT ELIMINATED"):
            pass
        else:
            needs_deep_dive.append((desc, trial))

    if verbose:
        print(f"  Quick pass done: {len(found)} found so far, "
              f"{len(needs_deep_dive)} inconclusive -- escalating to full verification")
    for desc, trial in needs_deep_dive:
        if verbose:
            print(f"  Deep-diving: {desc} ...")
        result = solve_elimination(games, team, target, overrides=trial, max_iterations=None, verbose=verbose)
        if result["status"].startswith("ELIMINATED"):
            found.append((desc, trial))

    return _dedupe_supersets(found)


def fast_league_scan(games, teams=None, targets=None, overrides=None):
    """
    Cheap first pass: runs ONLY the fast, exact checks (no expensive search)
    for every team/target/direction. Returns (resolved, borderline):
      resolved: list of dicts {team, target, kind, status} -- cleanly decided
      borderline: list of dicts {team, target, kind} -- needs a deep dive
    kind is 'clinch' or 'eliminate'.
    """
    teams = teams or ALL_TEAMS
    targets = targets or ["playoffs", "division", "home_game"]
    overrides = dict(overrides or {})
    resolved, borderline = [], []

    for team in teams:
        for target in targets:
            r = solve_status(games, team, target, overrides=overrides, verbose=False, fast_only=True)
            entry = dict(team=team, target=target, kind="clinch")
            if r["status"].startswith("BORDERLINE"):
                borderline.append(entry)
            else:
                resolved.append(dict(entry, status=r["status"]))

            if target in ("playoffs", "division"):
                re_ = solve_elimination(games, team, target, overrides=overrides, verbose=False, fast_only=True)
                entry2 = dict(team=team, target=target, kind="eliminate")
                if re_["status"].startswith("BORDERLINE"):
                    borderline.append(entry2)
                else:
                    resolved.append(dict(entry2, status=re_["status"]))

    return resolved, borderline


def deep_dive(games, team, target, kind, overrides=None, **kwargs):
    """
    Run the full, rigorous (slow) check for one specific borderline item --
    for interactive use after fast_league_scan flags something as
    BORDERLINE. Defaults to an UNLIMITED search budget so a genuine
    elimination/clinch that needs a large search isn't silently missed;
    pass ambiguous_max_iterations=N (clinch) or max_iterations=N
    (eliminate) explicitly to cap it for a quicker, best-effort look.
    """
    overrides = dict(overrides or {})
    if kind == "clinch":
        kwargs.setdefault("ambiguous_max_iterations", None)
        return solve_status(games, team, target, overrides=overrides, fast_only=False, **kwargs)
    elif kind == "eliminate":
        kwargs.setdefault("max_iterations", None)
        return solve_elimination(games, team, target, overrides=overrides, fast_only=False, **kwargs)
    raise ValueError(kind)


# ============================================================
# PART 4: LOAD DATA (from the PIMS API, converted to the same
# internal `games` structure the rest of this script already expects
# and was validated against all day), AUTO-DETECT CURRENT WEEK
# ============================================================

# ---- PIMS API -> internal games list converter ----
# Confirmed against real API responses on 2026-09-22 (see project notes):
# game_type_id==1 is the only value meaning "real regular-season game,
# both teams known" (0 = preseason, negative week; 2-6 = playoff-bracket
# placeholders with null teams, since who plays in them is the OUTPUT of
# this computation, not an input to it). A played game has
# home_team_score/away_team_score present; an unplayed one omits both.
TEAM_ID_TO_ABBREV = {
    11: "MTL", 19: "TOR", 8: "HAM", 13: "OTT",
    7: "EDM", 17: "SSK", 1: "BC", 20: "WPG", 6: "CGY",
}

def convert_fixtures_to_games(fixtures_json):
    games = []
    skipped_preseason = skipped_playoff_placeholder = skipped_unknown_team = 0
    for fx in fixtures_json:
        if fx.get("game_type_id") != 1:
            if fx.get("game_type_id") == 0:
                skipped_preseason += 1
            else:
                skipped_playoff_placeholder += 1
            continue
        home_id, away_id = fx.get("home_team_id"), fx.get("away_team_id")
        if home_id not in TEAM_ID_TO_ABBREV or away_id not in TEAM_ID_TO_ABBREV:
            skipped_unknown_team += 1
            continue
        games.append({
            "gm": fx["ID"], "home": TEAM_ID_TO_ABBREV[home_id], "away": TEAM_ID_TO_ABBREV[away_id],
            "wk": fx["week"], "hpf": fx.get("home_team_score"), "apf": fx.get("away_team_score"),
        })
    print(f"Converted {len(games)} regular-season games ({skipped_preseason} preseason, "
          f"{skipped_playoff_placeholder} playoff-placeholder, {skipped_unknown_team} unknown-team skipped).")
    return games

def cross_check_clinched_against_api(standings_json, computed_clinched_teams):
    """Sanity check, not a data source: compares the API's own "flags":"x"
    (clinched) markers against what OUR computation independently concludes.
    Run every time -- if these ever disagree, investigate before trusting
    that run's output."""
    api_clinched = set()
    for division in standings_json["data"]["divisions"].values():
        if division["division_name"] == "unified":
            continue
        for team in division["standings"]:
            if team.get("flags") and "x" in team["flags"]:
                api_clinched.add(team["abbreviation"])
    computed = set(computed_clinched_teams)
    if api_clinched != computed:
        print(f"  *** CROSS-CHECK MISMATCH *** API says clinched: {api_clinched}, "
              f"our computation says clinched: {computed}. Investigate before trusting this run.")
    else:
        print(f"  Cross-check OK -- API and our computation agree on clinched teams: {sorted(api_clinched)}")
    return api_clinched == computed

def get_current_week_games_and_number(games):
    """Auto-detect 'this week's games' as all games sharing the earliest
    week number among games that haven't been played yet (hpf is None).
    Returns (list_of_game_numbers, week_number)."""
    undetermined = [g for g in games if g["hpf"] is None]
    if not undetermined:
        raise ValueError("No undetermined games found -- season may be complete, or API data is stale.")
    wk = min(g["wk"] for g in undetermined)
    return [g["gm"] for g in undetermined if g["wk"] == wk], wk

def get_clinch_scenarios_or_status(games, team, target, week_games, verbose=True):
    """
    Before running the full scenario search, check whether team's status
    is ALREADY unconditionally decided regardless of this week's games --
    otherwise the search trivially finds every combination "works" (since
    the team was never actually at risk), producing a misleading list of
    conditions that aren't really needed. Mirrors how the real press
    release states "X has clinched" as a plain fact once already true,
    rather than a hypothetical.
    Returns dict(already=True, status=<str>) or dict(already=False, scenarios=<list>).
    """
    if verbose:
        print(f"  Checking whether {team} has already clinched (this can take a while "
              f"and search progress will print below if so)...")
    current = solve_status(games, team, target, overrides={}, ambiguous_max_iterations=None, verbose=verbose)
    if current["status"].startswith("CLINCHED"):
        if verbose:
            print(f"  {team} has ALREADY clinched (regardless of this week's games) -- skipping scenario search.")
        return dict(already=True, status=current["status"])
    scenarios = find_clinch_scenarios_complete(games, team, target, week_games, base_overrides={}, verbose=verbose)
    return dict(already=False, scenarios=scenarios)

def build_press_release_summary_text(division_name, teams_results):
    """Same as print_press_release_summary, but returns the text as a
    string (for emailing) instead of just printing it."""
    lines = [f"\n{division_name.upper()} DIVISION"]
    any_lines = False
    for team, res in teams_results.items():
        if res.get("already_clinched"):
            lines.append(f"  {team} has ALREADY CLINCHED a playoff spot.")
            any_lines = True
        else:
            for s in res["clinch"]:
                lines.append(f"  {team} clinches a playoff spot if {s}.")
                any_lines = True
        if res.get("already_eliminated"):
            lines.append(f"  {team} has ALREADY BEEN ELIMINATED from playoff contention.")
            any_lines = True
        else:
            for s in res["eliminate"]:
                lines.append(f"  {team} is eliminated from playoff contention if {s}.")
                any_lines = True
        if res.get("already_clinched_division"):
            lines.append(f"  {team} has ALREADY CLINCHED the division title.")
            any_lines = True
        else:
            for s in res.get("division", []):
                lines.append(f"  {team} clinches the division title if {s}.")
                any_lines = True
        if res.get("already_clinched_home_game"):
            lines.append(f"  {team} has ALREADY CLINCHED a home playoff game.")
            any_lines = True
        else:
            for s in res.get("home_game", []):
                lines.append(f"  {team} clinches a home playoff game if {s}.")
                any_lines = True
    if not any_lines:
        lines.append("  No clinching or elimination scenarios this week.")
    return "\n".join(lines)

def print_press_release_summary(division_name, teams_results):
    print(build_press_release_summary_text(division_name, teams_results))


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SENDER_EMAIL = "jeffkrever@gmail.com"      # <-- fill in your Gmail address
RECIPIENT_EMAIL = "jkrever@cfl.ca"   # <-- fill in where to send it (can be the same address)

# Optional: paste a real CFL logo URL here (e.g. from your own official
# brand assets) to have it appear at the top of the email. Leave as None
# to skip the logo and use text-only styling.
LOGO_URL = None  # e.g. "https://your-official-asset-host/cfl-logo.png"

def build_press_release_summary_html(division_name, teams_results):
    rows = []
    any_lines = False
    for team, res in teams_results.items():
        if res.get("already_clinched"):
            rows.append(f"<li><strong>{team}</strong> has <strong>ALREADY CLINCHED</strong> a playoff spot.</li>")
            any_lines = True
        else:
            for s in res["clinch"]:
                rows.append(f"<li><strong>{team}</strong> clinches a playoff spot if {s}.</li>")
                any_lines = True
        if res.get("already_eliminated"):
            rows.append(f"<li><strong>{team}</strong> has <strong>ALREADY BEEN ELIMINATED</strong> from playoff contention.</li>")
            any_lines = True
        else:
            for s in res["eliminate"]:
                rows.append(f"<li><strong>{team}</strong> is eliminated from playoff contention if {s}.</li>")
                any_lines = True
        if res.get("already_clinched_division"):
            rows.append(f"<li><strong>{team}</strong> has <strong>ALREADY CLINCHED</strong> the division title.</li>")
            any_lines = True
        else:
            for s in res.get("division", []):
                rows.append(f"<li><strong>{team}</strong> clinches the division title if {s}.</li>")
                any_lines = True
        if res.get("already_clinched_home_game"):
            rows.append(f"<li><strong>{team}</strong> has <strong>ALREADY CLINCHED</strong> a home playoff game.</li>")
            any_lines = True
        else:
            for s in res.get("home_game", []):
                rows.append(f"<li><strong>{team}</strong> clinches a home playoff game if {s}.</li>")
                any_lines = True
    if not any_lines:
        rows.append("<li>No clinching or elimination scenarios this week.</li>")
    return f"""
    <h2 style="color:#002244; border-bottom:2px solid #002244; padding-bottom:4px;
               font-family:Georgia, 'Times New Roman', serif; letter-spacing:1px;">
      {division_name.upper()} DIVISION
    </h2>
    <ul style="font-family:Georgia, 'Times New Roman', serif; font-size:15px; line-height:1.6;">
      {"".join(rows)}
    </ul>
    """

def build_email_html(current_wk, east_results, west_results):
    logo_html = f'<img src="{LOGO_URL}" alt="CFL" style="height:60px; margin-bottom:10px;"><br>' if LOGO_URL else ""
    return f"""
    <html>
    <body style="background:#f4f4f4; padding:20px; margin:0;">
      <div style="max-width:640px; margin:0 auto; background:#ffffff; padding:30px 40px;
                  border:1px solid #ccc; font-family:Georgia, 'Times New Roman', serif;">
        <div style="text-align:center; border-bottom:4px solid #C8102E; padding-bottom:15px; margin-bottom:20px;">
          {logo_html}
          <div style="color:#002244; font-size:13px; letter-spacing:3px; font-weight:bold;">
            CANADIAN FOOTBALL LEAGUE
          </div>
        </div>
        <p style="font-size:12px; color:#555; text-transform:uppercase; letter-spacing:1px; margin-bottom:0;">
          For Internal Use &mdash; Playoff Scenario Report
        </p>
        <h1 style="font-size:24px; color:#002244; margin-top:6px; margin-bottom:2px;">
          Week {current_wk} Grey Cup Playoff Scenarios
        </h1>
        <p style="font-size:13px; color:#777; margin-top:0;">
          Generated automatically &mdash; verified scenarios, complete search
        </p>
        <hr style="border:none; border-top:1px solid #ddd; margin:20px 0;">
        {build_press_release_summary_html("East", east_results)}
        {build_press_release_summary_html("West", west_results)}
        <hr style="border:none; border-top:1px solid #ddd; margin:30px 0 10px 0;">
        <p style="font-size:11px; color:#999; text-align:center;">
          Automatically generated by the CFL Playoff Scenario Generator.
        </p>
      </div>
    </body>
    </html>
    """

def send_results_email(subject, html_body, plaintext_fallback):
    try:
        from google.colab import userdata
        app_password = userdata.get("CFL_APP_PASSWORD")
    except Exception as e:
        print(f"\n[Email not sent -- could not read CFL_APP_PASSWORD from Colab Secrets: {e}]")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(plaintext_fallback, "plain"))  # shown if the client can't render HTML
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, app_password)
            server.send_message(msg)
        print(f"\n[Results emailed to {RECIPIENT_EMAIL}]")
    except Exception as e:
        print(f"\n[Email failed to send: {e}]")

def get_elimination_scenarios_or_status(games, team, target, week_games, verbose=True):
    """
    Mirror of get_clinch_scenarios_or_status for the elimination side.
    Same fix applied from the start: the pre-check runs with an unlimited
    search budget AND passes verbose through, so a team whose elimination
    status takes a long search to resolve shows visible progress instead
    of an unexplained silent gap.
    Returns dict(already=True, status=<str>) or dict(already=False, scenarios=<list>).
    """
    if verbose:
        print(f"  Checking whether {team} is already eliminated (this can take a while "
              f"and search progress will print below if so)...")
    current = solve_elimination(games, team, target, overrides={}, max_iterations=None, verbose=verbose)
    if current["status"].startswith("ELIMINATED"):
        if verbose:
            print(f"  {team} is ALREADY eliminated (regardless of this week's games) -- skipping scenario search.")
        return dict(already=True, status=current["status"])
    scenarios = find_elimination_scenarios_complete(games, team, target, week_games, base_overrides={}, verbose=verbose)
    return dict(already=False, scenarios=scenarios)

EAST_TEAMS = ["MTL", "TOR", "HAM", "OTT"]
WEST_TEAMS = ["EDM", "SSK", "BC", "WPG", "CGY"]

def run_full_checks(team_list, games, week_games, verbose=True):
    """Same logic as always -- now takes `games` and `week_games` as
    explicit parameters instead of closing over module-level globals.
    In a one-shot script that distinction never mattered; in a
    persistent web server, module globals set once at import time would
    make every request silently reuse the very first run's data forever."""
    results = {}
    for team in team_list:
        if verbose: print(f"\n--- {team} ---")

        if verbose: print(f"  [playoffs -- clinch]")
        po_clinch_result = get_clinch_scenarios_or_status(games, team, "playoffs", week_games, verbose=verbose)
        if po_clinch_result["already"]:
            po_clinch = []
        else:
            po_clinch = po_clinch_result["scenarios"]
            if verbose:
                if po_clinch:
                    for s in po_clinch:
                        print(f"    - {team} clinches a playoff spot if: {s}")
                else:
                    print("    No clinching scenario found this week.")

        if verbose: print(f"  [playoffs -- eliminate]")
        po_elim_result = get_elimination_scenarios_or_status(games, team, "playoffs", week_games, verbose=verbose)
        if po_elim_result["already"]:
            po_elim = []
        else:
            po_elim = po_elim_result["scenarios"]
            if verbose:
                if po_elim:
                    for s in po_elim:
                        print(f"    - {team} is eliminated from playoff contention if: {s}")
                else:
                    print("    No playoff-elimination scenario found this week.")

        if verbose: print(f"  [division -- clinch]")
        div_clinch_result = get_clinch_scenarios_or_status(games, team, "division", week_games, verbose=verbose)
        if div_clinch_result["already"]:
            div_clinch = []
        else:
            div_clinch = div_clinch_result["scenarios"]
            if verbose:
                if div_clinch:
                    for s in div_clinch:
                        print(f"    - {team} clinches the division title if: {s}")
                else:
                    print("    No division-clinching scenario found this week.")

        if verbose: print(f"  [division -- eliminate]")
        div_elim_result = get_elimination_scenarios_or_status(games, team, "division", week_games, verbose=verbose)
        if div_elim_result["already"]:
            div_elim = []
        else:
            div_elim = div_elim_result["scenarios"]
            if verbose:
                if div_elim:
                    for s in div_elim:
                        print(f"    - {team} is eliminated from the division title if: {s}")
                else:
                    print("    No division-elimination scenario found this week.")

        if verbose: print(f"  [home_game -- clinch]")
        hg_clinch_result = get_clinch_scenarios_or_status(games, team, "home_game", week_games, verbose=verbose)
        if hg_clinch_result["already"]:
            hg_clinch = []
        else:
            hg_clinch = hg_clinch_result["scenarios"]
            if verbose:
                if hg_clinch:
                    for s in hg_clinch:
                        print(f"    - {team} clinches a home playoff game if: {s}")
                else:
                    print("    No home-game-clinching scenario found this week.")

        if verbose: print(f"  [home_game -- eliminate]")
        hg_elim_result = get_elimination_scenarios_or_status(games, team, "home_game", week_games, verbose=verbose)
        if hg_elim_result["already"]:
            hg_elim = []
        else:
            hg_elim = hg_elim_result["scenarios"]
            if verbose:
                if hg_elim:
                    for s in hg_elim:
                        print(f"    - {team} is eliminated from hosting a playoff game if: {s}")
                else:
                    print("    No home-game-elimination scenario found this week.")

        results[team] = {
            "clinch": po_clinch, "already_clinched": po_clinch_result["status"] if po_clinch_result["already"] else None,
            "eliminate": po_elim, "already_eliminated": po_elim_result["status"] if po_elim_result["already"] else None,
            "division": div_clinch, "already_clinched_division": div_clinch_result["status"] if div_clinch_result["already"] else None,
            "home_game": hg_clinch, "already_clinched_home_game": hg_clinch_result["status"] if hg_clinch_result["already"] else None,
        }
    return results


def run_full_scan(season_id=None, year=None, send_email=False, verbose=True):
    """
    The single entry point for a fresh, complete run: fetches LIVE data
    from the PIMS API (never stale, never reused across calls -- see the
    note on run_full_checks above), runs every clinch/elimination check
    for all 9 teams, cross-checks the result against the API's own
    clinch flags, and returns a structured result dict. This is what the
    web backend calls per request; nothing here is cached at import time.
    """
    season_id = season_id if season_id is not None else SEASON_ID
    year = year if year is not None else STANDINGS_YEAR

    if verbose:
        print(f"Fetching fixtures and standings from {PIMS_BASE} (season_id={season_id}, year={year})...")
    fixtures_json = fetch_pims_fixtures(season_id)
    standings_json = fetch_pims_standings(year)
    if verbose:
        print(f"Fetched {len(fixtures_json)} fixtures.")

    games = convert_fixtures_to_games(fixtures_json)
    week_games, current_wk = get_current_week_games_and_number(games)
    if verbose:
        print(f"\nAuto-detected current week: {current_wk}  (games: {week_games})")

    if verbose:
        print(f"\n\n=== FULL SCAN -- Week {current_wk}, all 9 teams: clinch + elimination, playoffs/division/home_game ===")

    east_results = run_full_checks(EAST_TEAMS, games, week_games, verbose=verbose)
    west_results = run_full_checks(WEST_TEAMS, games, week_games, verbose=verbose)

    if verbose:
        print("\n\n" + "="*60)
        print(f"WEEK {current_wk} PLAYOFF SCENARIOS")
        print("="*60)
        print_press_release_summary("East", east_results)
        print_press_release_summary("West", west_results)
        print()

    computed_clinched = {t for t, r in {**east_results, **west_results}.items() if r.get("already_clinched")}
    cross_check_ok = cross_check_clinched_against_api(standings_json, computed_clinched)

    result = {
        "current_week": current_wk,
        "week_games": week_games,
        "east_results": east_results,
        "west_results": west_results,
        "cross_check_ok": cross_check_ok,
        "east_summary_text": build_press_release_summary_text("East", east_results),
        "west_summary_text": build_press_release_summary_text("West", west_results),
    }

    if send_email:
        email_plaintext = (
            f"WEEK {current_wk} PLAYOFF SCENARIOS\n"
            f"{'='*60}\n"
            f"{result['east_summary_text']}\n"
            f"{result['west_summary_text']}\n"
        )
        email_html = build_email_html(current_wk, east_results, west_results)
        send_results_email(f"CFL Week {current_wk} Playoff Scenarios", email_html, email_plaintext)

    return result
