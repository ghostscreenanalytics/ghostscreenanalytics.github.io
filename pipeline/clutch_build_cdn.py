"""Add 2025-26 (regular season + playoffs) from the NBA CDN play-by-play format,
in the same per-player-season format as build.py. Appends to player_seasons.csv."""
import pandas as pd, numpy as np, tarfile, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else 'pipeline/inputs/nba_data/datasets'
PS = sys.argv[2] if len(sys.argv) > 2 else 'pipeline/out/clutch_player_seasons.csv'

def load(path):
    with tarfile.open(path) as t:
        m = [x for x in t.getmembers() if x.name.endswith('.csv')][0]
        return pd.read_csv(t.extractfile(m), low_memory=False,
            usecols=['gameId','orderNumber','clock','period','actionType','shotResult','personId',
                     'playerNameI','scoreHome','scoreAway','assistPersonId','subType'])
rows = []
for name, po in [('cdnnba_2025', False), ('cdnnba_po_2025', True)]:
    d = load(f'{SRC}/{name}.tar.xz').sort_values(['gameId','orderNumber']).reset_index(drop=True)
    for c in ['scoreHome','scoreAway']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
        d[c] = d.groupby('gameId')[c].ffill().fillna(0)
    m_after = d.scoreHome - d.scoreAway
    d['m_before'] = m_after.groupby(d.gameId).shift(1).fillna(0)
    t = d['clock'].astype(str).str.extract(r'PT(\d+)M([\d.]+)S').astype(float)
    d['sec'] = t[0]*60 + t[1]
    d['clutch'] = (d.period >= 4) & (d.sec <= 300) & (d.m_before.abs() <= 5)
    at = d.actionType.astype(str); made = d.shotResult.astype(str).eq('Made')
    fga = at.isin(['2pt','3pt']); fgm = fga & made
    fta = at.eq('freethrow'); ftm = fta & made
    tov = at.eq('turnover')
    is3 = at.eq('3pt')
    pts = fgm*np.where(is3,3,2) + ftm
    b = pd.DataFrame({'pid': d.personId, 'name': d.playerNameI, 'clutch': d.clutch, 'game': d.gameId,
        'pts': pts, 'fgm': fgm.astype(int), 'fga': fga.astype(int), 'tpm': (fgm&is3).astype(int),
        'tpa': (fga&is3).astype(int), 'ftm': ftm.astype(int), 'fta': fta.astype(int), 'tov': tov.astype(int)})
    b = b[(b.pid > 0) & (b[['fga','fta','tov']].sum(axis=1) > 0)]
    g = b.groupby(['pid','clutch']).agg(name=('name','last'), pts=('pts','sum'), fgm=('fgm','sum'),
        fga=('fga','sum'), tpm=('tpm','sum'), tpa=('tpa','sum'), ftm=('ftm','sum'), fta=('fta','sum'),
        tov=('tov','sum'), games=('game','nunique')).reset_index()
    a = d[fgm & (pd.to_numeric(d.assistPersonId, errors='coerce') > 0)]
    ag = a.groupby([pd.to_numeric(a.assistPersonId), 'clutch']).size().reset_index(name='ast')
    ag.columns = ['pid','clutch','ast']
    g = g.merge(ag, on=['pid','clutch'], how='outer')
    g['ast'] = g.ast.fillna(0); g['name'] = g.name.fillna('')
    g = g.fillna(0)
    g['season'] = '2025-26'; g['yr'] = 2025; g['playoffs'] = po
    rows.append(g)
    print(name, len(d), int(d.clutch.sum()), g.pid.nunique())
new = pd.concat(rows, ignore_index=True)
old = pd.read_csv(PS)
old = old[old.yr != 2025]
# names: prefer full names from older seasons
full = old.sort_values('yr').groupby('pid').name.last()
new['name'] = new.pid.map(full).fillna(new.name)
out = pd.concat([old, new[old.columns]], ignore_index=True)
out.to_csv(PS, index=False)
c = new[(new.clutch==True)&(~new.playoffs)].nlargest(6,'pts')[['name','pts','fgm','fga','games']]
print(c.to_string(index=False))
