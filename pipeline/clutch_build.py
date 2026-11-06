"""Build per-player-season clutch and non-clutch stats from NBA.com play-by-play.
Clutch (NBA.com definition): 4th quarter or OT, <=5:00 left, score within 5.
Margin is the score margin before the play.
"""
import pandas as pd, numpy as np, tarfile, glob, os, io, sys
from collections import defaultdict

SRC = 'pipeline/inputs/nba_data/datasets'
OUT = 'pipeline/out/clutch_player_seasons.csv'
LATE = 'pipeline/out/clutch_late_shots.csv'

def load(path):
    with tarfile.open(path) as t:
        m = [x for x in t.getmembers() if x.name.endswith('.csv')][0]
        return pd.read_csv(t.extractfile(m), low_memory=False,
            usecols=['GAME_ID','EVENTMSGTYPE','PERIOD','PCTIMESTRING','HOMEDESCRIPTION',
                     'VISITORDESCRIPTION','SCOREMARGIN','PLAYER1_ID','PLAYER1_NAME',
                     'PLAYER2_ID','PLAYER2_NAME'])

rows = []; late = []
for path in sorted(glob.glob(f'{SRC}/nbastats_*.tar.xz')):
    name = os.path.basename(path).replace('.tar.xz','')
    po = '_po_' in name
    yr = int(name.split('_')[-1])
    season = f'{yr}-{str(yr+1)[2:]}'
    d = load(path)
    # margin (home - visitor), forward filled within game, margin BEFORE each event
    sm = d['SCOREMARGIN'].replace('TIE', '0')
    sm = pd.to_numeric(sm, errors='coerce')
    d['m_after'] = sm.groupby(d['GAME_ID']).ffill().fillna(0)
    d['m_before'] = d.groupby('GAME_ID')['m_after'].shift(1).fillna(0)
    t = d['PCTIMESTRING'].astype(str).str.split(':', expand=True)
    d['sec'] = pd.to_numeric(t[0], errors='coerce')*60 + pd.to_numeric(t[1], errors='coerce')
    d['clutch'] = (d['PERIOD'] >= 4) & (d['sec'] <= 300) & (d['m_before'].abs() <= 5)
    hd = d['HOMEDESCRIPTION'].fillna(''); vd = d['VISITORDESCRIPTION'].fillna('')
    d['home'] = hd.str.len() > 0
    desc = np.where(d['home'], hd, vd)
    desc = pd.Series(desc, index=d.index)
    et = d['EVENTMSGTYPE']
    is3 = desc.str.contains('3PT')
    fgm = et == 1; fga = et.isin([1, 2])
    ftm = (et == 3) & ~desc.str.contains('MISS')
    fta = (et == 3)
    # exclude technical FTs? keep all FTs (they count toward points)
    tov = (et == 5)
    pts = fgm*np.where(is3, 3, 2) + ftm*1
    base = pd.DataFrame({
        'pid': d['PLAYER1_ID'], 'name': d['PLAYER1_NAME'], 'clutch': d['clutch'],
        'game': d['GAME_ID'],
        'pts': pts, 'fgm': fgm.astype(int), 'fga': fga.astype(int),
        'tpm': (fgm & is3).astype(int), 'tpa': (fga & is3).astype(int),
        'ftm': ftm.astype(int), 'fta': fta.astype(int), 'tov': tov.astype(int)})
    base = base[(base['pid'] > 0) & base['name'].notna() &
                (base[['fga','fta','tov']].sum(axis=1) > 0)]
    g = base.groupby(['pid','name','clutch']).agg(
        pts=('pts','sum'), fgm=('fgm','sum'), fga=('fga','sum'), tpm=('tpm','sum'),
        tpa=('tpa','sum'), ftm=('ftm','sum'), fta=('fta','sum'), tov=('tov','sum'),
        games=('game','nunique')).reset_index()
    # assists: made FG with player2
    a = d[fgm & (d['PLAYER2_ID'] > 0) & d['PLAYER2_NAME'].notna()]
    ag = a.groupby(['PLAYER2_ID','PLAYER2_NAME','clutch']).size().reset_index(name='ast')
    ag.columns = ['pid','name','clutch','ast']
    g = g.merge(ag, on=['pid','name','clutch'], how='outer').fillna(0)
    g['season'] = season; g['yr'] = yr; g['playoffs'] = po
    rows.append(g)
    # final-minute shots to tie or take the lead
    own_before = np.where(d['home'], d['m_before'], -d['m_before'])
    val = np.where(is3, 3, 2)
    cond = (d['PERIOD'] >= 4) & (d['sec'] <= 60) & fga & (own_before <= 0) & (own_before + val >= 0)
    L = pd.DataFrame({'pid': d['PLAYER1_ID'], 'name': d['PLAYER1_NAME'], 'made': fgm.astype(int),
                      'yr': yr, 'playoffs': po})[cond]
    late.append(L)
    print(name, len(d), int(d['clutch'].sum()), flush=True)

ps = pd.concat(rows, ignore_index=True)
ps.to_csv(OUT, index=False)
pd.concat(late, ignore_index=True).to_csv(LATE, index=False)
print('saved', ps.shape)
