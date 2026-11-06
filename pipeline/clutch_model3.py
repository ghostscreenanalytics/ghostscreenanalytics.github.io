"""GSA Clutch Score v2: offense + playmaking + defense, current players.
All in points over a replacement-level / average baseline. Regular season counts once; every playoff piece counts twice
and is shown as Big games.
  Scoring volume      = chances x DELTA
  Scoring efficiency  = chances x blended efficiency vs league (clutch shrunk, 50/50 with normal)
  Playmaking          = 0.7 x AST + 0.7 x OREB                       (Game Score weights)
  Defense plays       = 1.0 x STL + 0.7 x BLK + 0.3 x DREB + 1.0 x charges drawn
  Team defense on floor = (league eff - opp pts per chance with him on) x opp chances / 5, shrunk
  Big games           = 2 x (playoff scoring value over replacement + playoff playmaking + defense + team defense)
Backtest target: next regular season on-court clutch net points (team pts - opponent pts while he plays, vs league).
"""
import pandas as pd, numpy as np, json, sys
from scipy.stats import spearmanr
D = 'pipeline/out/clutch_'
PS = D+'player_seasons.csv'; DF = D+'def_seasons.csv'
ROSTER = 'data/forecast.json'
if len(sys.argv) > 1: D, PS, DF, ROSTER = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
PO_W, K, K2 = 2.0, 300, 1000
W_AST, W_OREB, W_STL, W_BLK, W_DREB, W_CHG = 0.7, 0.7, 1.0, 0.7, 0.3, 1.0

d = pd.read_csv(PS).groupby(['pid','yr','playoffs','clutch'], as_index=False).agg(
    name=('name','last'), pts=('pts','sum'), fgm=('fgm','sum'), fga=('fga','sum'), tpm=('tpm','sum'),
    tpa=('tpa','sum'), ftm=('ftm','sum'), fta=('fta','sum'), tov=('tov','sum'), games=('games','sum'))
full = d.sort_values('yr').groupby('pid').name.last(); d['name'] = d.pid.map(full)
d['tsa'] = d.fga + 0.44*d.fta; d['ch'] = d.tsa + d.tov
lg = d.groupby(['yr','playoffs','clutch']).agg(P=('pts','sum'), CH=('ch','sum'), TS=('tsa','sum')).reset_index()
lg['lg_eff'] = lg.P/lg.CH; lg['lg_ts'] = lg.P/(2*lg.TS)
d = d.merge(lg[['yr','playoffs','clutch','lg_eff','lg_ts']], on=['yr','playoffs','clutch'])
C = d[d.clutch].copy(); N = d[~d.clutch].copy()
C['rel'] = C.pts/C.ch.replace(0, np.nan) - C.lg_eff
q = C[(~C.playoffs) & (C.ch >= 30)].sort_values('rel'); cw = q.ch.cumsum()/q.ch.sum()
DELTA = float(-q.rel[cw >= 0.20].iloc[0])
C['val'] = C.pts - (C.lg_eff - DELTA)*C.ch

df = pd.read_csv(DF).groupby(['pid','yr','playoffs'], as_index=False).sum()
# league clutch efficiency for on-court defense baseline (use C's league eff by yr/playoffs)
le = lg[lg.clutch][['yr','playoffs','lg_eff']]
df = df.merge(le, on=['yr','playoffs'], how='left')
df['plays'] = W_STL*df.stl + W_BLK*df.blk + W_DREB*df.dreb + W_CHG*df.chg
df['make'] = W_AST*df.ast + W_OREB*df.oreb
df['dsave'] = df.lg_eff*df.on_och - df.on_opts          # opponent points below league on his floor time
df['net'] = (df.on_pts - df.lg_eff*df.on_ch) - (df.on_opts - df.lg_eff*df.on_och)

def window(y0, y1, pids=None):
    c = C[(C.yr >= y0) & (C.yr <= y1)]; n = N[(N.yr >= y0) & (N.yr <= y1)]; f = df[(df.yr >= y0) & (df.yr <= y1)]
    if pids is not None: c = c[c.pid.isin(pids)]; n = n[n.pid.isin(pids)]; f = f[f.pid.isin(pids)]
    cr, cp = c[~c.playoffs], c[c.playoffs]; fr, fp = f[~f.playoffs], f[f.playoffs]
    g = pd.DataFrame(index=pd.Index(sorted(set(c.pid) | set(f.pid)), name='pid'))
    S = lambda x, col: x.groupby('pid')[col].sum().reindex(g.index).fillna(0)
    ch_rs = S(cr, 'ch')
    relc = (S(cr,'pts') - (cr.lg_eff*cr.ch).groupby(cr.pid).sum().reindex(g.index).fillna(0))/ch_rs.replace(0, np.nan)
    nch = S(n, 'ch'); reln = (S(n,'pts') - (n.lg_eff*n.ch).groupby(n.pid).sum().reindex(g.index).fillna(0))/nch.replace(0, np.nan)
    shr = relc.fillna(0)*ch_rs/(ch_rs+K)
    g['eff_est'] = 0.5*shr + 0.5*reln.fillna(0)
    g['vol'] = ch_rs*DELTA
    g['eff'] = ch_rs*g.eff_est
    g['make'] = S(fr, 'make')
    g['plays'] = S(fr, 'plays')
    och_rs = S(fr, 'on_och')
    g['team_d'] = S(fr, 'dsave')/5*och_rs/(och_rs+K2)
    och_po = S(fp, 'on_och')
    po_teamd = S(fp, 'dsave')/5*och_po/(och_po+K2)
    g['big'] = PO_W*(S(cp, 'val') + S(fp, 'make') + S(fp, 'plays') + po_teamd)
    g['big_score'] = PO_W*S(cp, 'val')
    g['off_score'] = g.vol + g.eff + g.big_score                      # v1 score (offense only)
    g['score'] = g.vol + g.eff + g.make + g.plays + g.team_d + g.big
    g['def_total'] = g.plays + g.team_d
    # display stats
    for col in ['pts','ch','tsa','tov','fgm','fga','tpm','tpa','ftm','fta','games']: g[col] = S(c, col)
    for col in ['stl','blk','dreb','oreb','ast','chg','pf','on_och','on_opts','on_ch','on_pts']: g[col] = S(f, col)
    g['po_pts'] = S(cp, 'pts')
    g['pe'] = g.pts/g.ch.replace(0, np.nan)
    g['lpe'] = (c.lg_eff*c.ch).groupby(c.pid).sum().reindex(g.index)/g.ch.replace(0, np.nan)
    g['ts'] = g.pts/(2*g.tsa.replace(0, np.nan)); g['lts'] = (c.lg_ts*c.tsa).groupby(c.pid).sum().reindex(g.index)/g.tsa.replace(0, np.nan)
    g['opp_eff'] = g.on_opts/g.on_och.replace(0, np.nan)
    g['lg_def'] = (f.lg_eff*f.on_och).groupby(f.pid).sum().reindex(g.index)/g.on_och.replace(0, np.nan)
    g['tsn'] = (S(n,'pts')/(2*S(n,'tsa').replace(0, np.nan)))
    return g

# ---------- backtest ----------
nxt = df[~df.playoffs][['pid','yr','net','on_och','on_ch']]
recs = []
for t in range(2000, 2026):
    w = window(t-3, t-1)
    y = nxt[nxt.yr == t].set_index('pid')
    y = y[(y.on_ch + y.on_och) >= 60]
    j = w.join(y[['net']], how='inner')
    j = j[j.ch >= 30]
    j['yr'] = t
    recs.append(j[['yr','net','pts','off_score','score','def_total']])
B = pd.concat(recs)
bt = {}
for f, nm in [('pts', 'Clutch points only'), ('off_score', 'Clutch Score v1 (offense only)'),
              ('score', 'Clutch Score v2 (offense + playmaking + defense)'), ('def_total', 'Defense pieces alone')]:
    te = B[B.yr >= 2013]; tr = B[B.yr < 2013]
    bt[nm] = dict(test_r=round(float(spearmanr(te[f], te.net).correlation), 3),
                  train_r=round(float(spearmanr(tr[f], tr.net).correlation), 3))
print(json.dumps(bt, indent=1), (B.yr >= 2013).sum(), (B.yr < 2013).sum())

# ---------- current players ----------
F = json.load(open(ROSTER))
roster = {p['pid']: (p['name'], t) for t, L in F['rosters'].items() for p in L}
LAST = int(C.yr.max())
b3 = window(LAST-2, LAST, set(roster)); bc = window(1996, LAST, set(roster))
for b in (b3, bc):
    b['name'] = [roster[i][0] for i in b.index]; b['team'] = [roster[i][1] for i in b.index]
    b.sort_values('score', ascending=False, inplace=True); b['rank'] = np.arange(1, len(b)+1)
pd.set_option('display.width', 260)
cols = ['name','team','score','vol','eff','make','plays','team_d','big','off_score','pts','stl','blk','dreb','chg','opp_eff','lg_def']
print(b3[b3.ch >= 25].head(25)[cols].round(2).to_string())
print(bc[bc.ch >= 25].head(10)[cols].round(1).to_string())
print('best defense (3y):'); print(b3[b3.ch >= 25].sort_values('def_total', ascending=False).head(10)[cols].round(2).to_string())
meta = dict(delta=round(DELTA, 4), po_weight=PO_W, K=K, K2=K2, weights=dict(ast=W_AST, oreb=W_OREB, stl=W_STL, blk=W_BLK, dreb=W_DREB, chg=W_CHG),
            last_season=f"{LAST}-{str(LAST+1)[2:]}", backtest=bt, n_test=int((B.yr >= 2013).sum()), n_train=int((B.yr < 2013).sum()))
json.dump(meta, open(D+'summary3.json', 'w'), indent=1)
b3.to_csv(D+'board3v2.csv'); bc.to_csv(D+'boardcv2.csv')
