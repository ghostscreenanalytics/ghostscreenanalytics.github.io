"""Clutch defense + playmaking, per player-season, from NBA.com play-by-play (nbastats format)
and the 2025-26 CDN format. Reconstructs on-court lineups for 4th quarters and overtime from
substitutions, then counts, for clutch plays only:
  individual: STL, BLK, DREB, OREB, AST, charges drawn, personal fouls committed
  on court:   team chances/points and opponent chances/points while the player is on the floor
Clutch = period >= 4, <= 5:00 left, margin before the play within 5.
Output: def_seasons.csv  (pid, yr, playoffs, ...)
"""
import pandas as pd, numpy as np, tarfile, glob, os, sys
from collections import defaultdict
SRC = sys.argv[1] if len(sys.argv) > 1 else 'pipeline/inputs/nba_data/datasets'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'pipeline/out/clutch_def_seasons.csv'
YEARS = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
COLS = ['stl','blk','dreb','oreb','ast','chg','pf','on_ch','on_pts','on_och','on_opts','on_g']

def tsec(s):
    a = s.astype(str).str.split(':', expand=True)
    return pd.to_numeric(a[0], errors='coerce')*60 + pd.to_numeric(a[1], errors='coerce')

def run_nbastats(path, yr, po, acc, stats):
    with tarfile.open(path) as t:
        m = [x for x in t.getmembers() if x.name.endswith('.csv')][0]
        d = pd.read_csv(t.extractfile(m), low_memory=False, usecols=['GAME_ID','EVENTNUM','EVENTMSGTYPE',
            'EVENTMSGACTIONTYPE','PERIOD','PCTIMESTRING','HOMEDESCRIPTION','VISITORDESCRIPTION','SCOREMARGIN',
            'PERSON1TYPE','PLAYER1_ID','PLAYER1_TEAM_ID','PLAYER2_ID','PLAYER2_TEAM_ID','PLAYER3_ID','PLAYER3_TEAM_ID'])
    sm = pd.to_numeric(d.SCOREMARGIN.replace('TIE','0'), errors='coerce')
    d['m_before'] = sm.groupby(d.GAME_ID).ffill().fillna(0).groupby(d.GAME_ID).shift(1).fillna(0)
    d['sec'] = tsec(d.PCTIMESTRING)
    d = d[d.PERIOD >= 4]
    desc = (d.HOMEDESCRIPTION.fillna('') + ' ' + d.VISITORDESCRIPTION.fillna('')).values
    for (gid, per), g in d.groupby(['GAME_ID','PERIOD'], sort=False):
        teams = pd.unique(pd.concat([g.PLAYER1_TEAM_ID, g.PLAYER2_TEAM_ID]).dropna())
        teams = [int(x) for x in teams if x > 0][:2]
        if len(teams) != 2: stats['skip'] += 1; continue
        rows = list(g.itertuples(index=False))
        dsc = desc[d.index.get_indexer(g.index)]
        # starters: first appearance not as sub-in
        seen = {}; first_in = set()
        for r in rows:
            if r.EVENTMSGTYPE == 8:
                for pid, tm, inn in ((r.PLAYER1_ID, r.PLAYER1_TEAM_ID, False), (r.PLAYER2_ID, r.PLAYER1_TEAM_ID, True)):
                    if pid > 0 and pid not in seen:
                        seen[pid] = tm
                        if inn: first_in.add(pid)
            else:
                for pid, tm in ((r.PLAYER1_ID, r.PLAYER1_TEAM_ID), (r.PLAYER2_ID, r.PLAYER2_TEAM_ID), (r.PLAYER3_ID, r.PLAYER3_TEAM_ID)):
                    if pid > 0 and pid < 1610000000 and tm == tm and tm > 0 and pid not in seen:
                        # ignore technicals etc. by players not on court? keep simple
                        seen[pid] = tm
        on = {t: set(p for p, tm in seen.items() if tm == t and p not in first_in) for t in teams}
        valid = all(len(on[t]) == 5 for t in teams)
        stats['ok' if valid else 'bad'] += 1
        clutch_seen = set()
        last_shot_team = None
        for r, ds in zip(rows, dsc):
            et = r.EVENTMSGTYPE
            if et == 8:
                tm = int(r.PLAYER1_TEAM_ID) if r.PLAYER1_TEAM_ID == r.PLAYER1_TEAM_ID else None
                if tm in on:
                    on[tm].discard(r.PLAYER1_ID); on[tm].add(r.PLAYER2_ID)
                continue
            cl = (r.sec <= 300) and (abs(r.m_before) <= 5)
            if et in (1, 2): last_shot_team = r.PLAYER1_TEAM_ID
            if not cl: continue
            off = None; ch = 0.0; pts = 0
            if et == 1:
                off = r.PLAYER1_TEAM_ID; ch = 1.0; pts = 3 if '3PT' in ds else 2
                if r.PLAYER2_ID > 0 and r.PLAYER2_TEAM_ID == off: acc[r.PLAYER2_ID]['ast'] += 1
            elif et == 2:
                off = r.PLAYER1_TEAM_ID; ch = 1.0
                if r.PLAYER3_ID > 0 and 'BLOCK' in ds: acc[r.PLAYER3_ID]['blk'] += 1
            elif et == 3:
                off = r.PLAYER1_TEAM_ID; ch = 0.44; pts = 0 if 'MISS' in ds else 1
            elif et == 5:
                off = r.PLAYER1_TEAM_ID if r.PLAYER1_TEAM_ID == r.PLAYER1_TEAM_ID else (r.PLAYER1_ID if r.PLAYER1_ID > 1610000000 else None)
                ch = 1.0
                if r.PLAYER2_ID > 0 and 'STEAL' in ds.upper(): acc[r.PLAYER2_ID]['stl'] += 1
            elif et == 4:
                if 0 < r.PLAYER1_ID < 1610000000 and r.PLAYER1_TEAM_ID == r.PLAYER1_TEAM_ID:
                    k = 'oreb' if r.PLAYER1_TEAM_ID == last_shot_team else 'dreb'
                    acc[r.PLAYER1_ID][k] += 1
            elif et == 6:
                if 0 < r.PLAYER1_ID < 1610000000: acc[r.PLAYER1_ID]['pf'] += 1
                if r.EVENTMSGACTIONTYPE == 26 and r.PLAYER2_ID > 0: acc[r.PLAYER2_ID]['chg'] += 1
            if off is not None and off == off and ch > 0:
                off = int(off)
                if not valid or off not in on: continue
                de = teams[1] if off == teams[0] else teams[0]
                for p in on[off]:
                    a = acc[p]; a['on_ch'] += ch; a['on_pts'] += pts
                for p in on[de]:
                    a = acc[p]; a['on_och'] += ch; a['on_opts'] += pts
                for p in on[off] | on[de]:
                    if (gid, p) not in clutch_seen: clutch_seen.add((gid, p)); acc[p]['on_g'] += 1

def run_cdn(path, yr, po, acc, stats):
    with tarfile.open(path) as t:
        m = [x for x in t.getmembers() if x.name.endswith('.csv')][0]
        d = pd.read_csv(t.extractfile(m), low_memory=False, usecols=['gameId','orderNumber','clock','period',
            'actionType','subType','descriptor','shotResult','personId','teamId','scoreHome','scoreAway',
            'assistPersonId','foulDrawnPersonId'])
    d = d.sort_values(['gameId','orderNumber'])
    for c in ['scoreHome','scoreAway']:
        d[c] = pd.to_numeric(d[c], errors='coerce'); d[c] = d.groupby('gameId')[c].ffill().fillna(0)
    d['m_before'] = (d.scoreHome - d.scoreAway).groupby(d.gameId).shift(1).fillna(0)
    t2 = d['clock'].astype(str).str.extract(r'PT(\d+)M([\d.]+)S').astype(float)
    d['sec'] = t2[0]*60 + t2[1]
    d = d[d.period >= 4]
    for (gid, per), g in d.groupby(['gameId','period'], sort=False):
        teams = [int(x) for x in pd.unique(g.teamId.dropna()) if x > 0][:2]
        if len(teams) != 2: stats['skip'] += 1; continue
        rows = list(g.itertuples(index=False))
        seen = {}; first_in = set()
        for r in rows:
            if r.personId and r.personId > 0 and r.teamId == r.teamId and r.personId not in seen:
                seen[r.personId] = int(r.teamId)
                if r.actionType == 'substitution' and r.subType == 'in': first_in.add(r.personId)
        on = {t: set(p for p, tm in seen.items() if tm == t and p not in first_in) for t in teams}
        valid = all(len(on[t]) == 5 for t in teams)
        stats['ok' if valid else 'bad'] += 1
        cs = set()
        for r in rows:
            at = r.actionType; tm = int(r.teamId) if r.teamId == r.teamId else None
            if at == 'substitution':
                if tm in on:
                    if r.subType == 'out': on[tm].discard(r.personId)
                    else: on[tm].add(r.personId)
                continue
            if not ((r.sec <= 300) and (abs(r.m_before) <= 5)): continue
            off = None; ch = 0.0; pts = 0; made = r.shotResult == 'Made'
            if at in ('2pt', '3pt'):
                off = tm; ch = 1.0; pts = (3 if at == '3pt' else 2) if made else 0
                if made and r.assistPersonId == r.assistPersonId and r.assistPersonId > 0: acc[int(r.assistPersonId)]['ast'] += 1
            elif at == 'freethrow':
                off = tm; ch = 0.44; pts = 1 if made else 0
            elif at == 'turnover':
                off = tm; ch = 1.0
            elif at == 'steal' and r.personId > 0: acc[r.personId]['stl'] += 1
            elif at == 'block' and r.personId > 0: acc[r.personId]['blk'] += 1
            elif at == 'rebound' and r.personId > 0 and tm is not None:
                acc[r.personId]['oreb' if r.subType == 'offensive' else 'dreb'] += 1
            elif at == 'foul':
                if r.personId > 0: acc[r.personId]['pf'] += 1
                if str(r.subType) == 'offensive' and str(r.descriptor) == 'charge' and r.foulDrawnPersonId == r.foulDrawnPersonId and r.foulDrawnPersonId > 0:
                    acc[int(r.foulDrawnPersonId)]['chg'] += 1
            if valid and off in on and ch > 0:
                de = teams[1] if off == teams[0] else teams[0]
                for p in on[off]: acc[p]['on_ch'] += ch; acc[p]['on_pts'] += pts
                for p in on[de]: acc[p]['on_och'] += ch; acc[p]['on_opts'] += pts
                for p in on[off] | on[de]:
                    if (gid, p) not in cs: cs.add((gid, p)); acc[p]['on_g'] += 1

jobs = []
for path in sorted(glob.glob(f'{SRC}/nbastats_*.tar.xz')):
    nm = os.path.basename(path)[:-7]; yr = int(nm.split('_')[-1]); po = '_po_' in nm
    jobs.append((run_nbastats, path, yr, po))
for nm, po in [('cdnnba_2025', False), ('cdnnba_po_2025', True)]:
    if os.path.exists(f'{SRC}/{nm}.tar.xz'): jobs.append((run_cdn, f'{SRC}/{nm}.tar.xz', 2025, po))
if YEARS: jobs = [j for j in jobs if j[2] in YEARS]
out = []
for fn, path, yr, po in jobs:
    acc = defaultdict(lambda: defaultdict(float)); stats = defaultdict(int)
    fn(path, yr, po, acc, stats)
    for p, a in acc.items():
        out.append(dict(pid=int(p), yr=yr, playoffs=po, **{c: a.get(c, 0.0) for c in COLS}))
    print(os.path.basename(path), dict(stats), flush=True)
pd.DataFrame(out).to_csv(OUT, index=False)
print('saved', len(out))
