"""GSA Clutch Score, current players.
Value over replacement on clutch chances (shots + turnovers), playoff clutch counted double.
  chances = FGA + 0.44*FTA + TOV
  eff     = points per chance
  value   = points - replacement_eff * chances      (replacement = league eff - DELTA, per season)
  score   = regular-season value + 2 x playoff value, over the last three seasons
Backtest: which version of the score best predicts a player's next-season clutch value.
"""
import pandas as pd, numpy as np, json, sys
PS = sys.argv[1] if len(sys.argv) > 1 else 'pipeline/out/clutch_player_seasons.csv'
OUTD = sys.argv[2] if len(sys.argv) > 2 else 'pipeline/out/clutch_'
ROSTER = sys.argv[3] if len(sys.argv) > 3 else 'data/forecast.json'
PO_W = 2.0
d = pd.read_csv(PS)
d = d.groupby(['pid','yr','playoffs','clutch'], as_index=False).agg(
    name=('name','last'), pts=('pts','sum'), fgm=('fgm','sum'), fga=('fga','sum'), tpm=('tpm','sum'),
    tpa=('tpa','sum'), ftm=('ftm','sum'), fta=('fta','sum'), tov=('tov','sum'), ast=('ast','sum'), games=('games','sum'))
full = d.sort_values('yr').groupby('pid').name.last()
d['name'] = d.pid.map(full)
d['tsa'] = d.fga + 0.44*d.fta
d['ch'] = d.tsa + d.tov
lg = d.groupby(['yr','playoffs','clutch']).agg(P=('pts','sum'), CH=('ch','sum'), TS=('tsa','sum')).reset_index()
lg['lg_eff'] = lg.P/lg.CH; lg['lg_ts'] = lg.P/(2*lg.TS)
d = d.merge(lg[['yr','playoffs','clutch','lg_eff','lg_ts']], on=['yr','playoffs','clutch'])
d['rel'] = d.pts/d.ch.replace(0, np.nan) - d.lg_eff

C = d[d.clutch].copy(); N = d[~d.clutch].copy()
# replacement level: 20th percentile of relative efficiency, regular-season player-seasons with 30+ chances,
# weighted by chances
q = C[(~C.playoffs) & (C.ch >= 30)].sort_values('rel')
cw = q.ch.cumsum()/q.ch.sum()
DELTA = float(-q.rel[cw >= 0.20].iloc[0])
C['repl'] = C.lg_eff - DELTA
C['val'] = C.pts - C.repl*C.ch
C['w'] = np.where(C.playoffs, PO_W, 1.0)
C['wval'] = C.val*C.w
C['avg_add'] = C.pts - C.lg_eff*C.ch          # vs. average, for display
print('replacement = league efficiency minus', round(DELTA, 3))

# ---------- backtest: predict next regular season clutch value ----------
# features from the three seasons before season t (regular season + playoffs)
rs = C[~C.playoffs].groupby(['pid','yr']).agg(val=('val','sum'), pts=('pts','sum'), ch=('ch','sum')).reset_index()
recs = []
byp = {p: g for p, g in C.groupby('pid')}
nbyp = {p: g for p, g in N.groupby('pid')}
K = 300
for (p, y), row in rs.set_index(['pid','yr']).iterrows():
    if y < 2000: continue
    g = byp[p]; prior = g[(g.yr >= y-3) & (g.yr < y)]
    if prior.ch.sum() < 30: continue
    n = nbyp.get(p); np_ = n[(n.yr >= y-3) & (n.yr < y)] if n is not None else None
    pr = prior[~prior.playoffs]; pp = prior[prior.playoffs]
    ch = pr.ch.sum()
    rel_c = (pr.pts.sum() - (pr.lg_eff*pr.ch).sum())/max(ch, 1)
    rel_n = ((np_.pts.sum() - (np_.lg_eff*np_.ch).sum())/np_.ch.sum()) if np_ is not None and np_.ch.sum() > 0 else 0
    shr = rel_c*ch/(ch+K)
    recs.append(dict(pid=p, yr=y, y_val=row.val,
        f_pts=pr.pts.sum(),                                # raw clutch points
        f_val=pr.val.sum(),                                # value over replacement, regular season
        f_score=pr.val.sum() + PO_W*pp.val.sum(),          # + playoffs double
        f_model=ch*(0.5*shr + 0.5*rel_n + DELTA) + PO_W*pp.val.sum()))   # efficiency shrunk toward normal shooting
B = pd.DataFrame(recs)
tr = B[B.yr <= 2012]; te = B[B.yr >= 2013]
from scipy.stats import spearmanr
bt = {}
names = {'f_pts': 'Clutch points only', 'f_val': 'Value over replacement (regular season)',
         'f_score': 'GSA Clutch Score (adds playoffs x2)', 'f_model': 'Clutch Score, efficiency blended with normal shooting'}
for f, nm in names.items():
    r = spearmanr(te[f], te.y_val).correlation
    rtr = spearmanr(tr[f], tr.y_val).correlation
    bt[nm] = dict(test_r=round(float(r), 3), train_r=round(float(rtr), 3))
print(json.dumps(bt, indent=1), len(tr), len(te))

# ---------- current players ----------
F = json.load(open(ROSTER))
roster = {p['pid']: (p['name'], t) for t, L in F['rosters'].items() for p in L}
LAST = int(C.yr.max())
def board(win):
    X = C[(C.yr >= LAST - win + 1)] if win else C
    Nn = N[(N.yr >= LAST - win + 1)] if win else N
    g = X.groupby('pid').agg(pts=('pts','sum'), ch=('ch','sum'), tsa=('tsa','sum'), tov=('tov','sum'),
        fgm=('fgm','sum'), fga=('fga','sum'), tpm=('tpm','sum'), tpa=('tpa','sum'), ftm=('ftm','sum'),
        fta=('fta','sum'), ast=('ast','sum'), games=('games','sum'), score=('wval','sum'),
        add=('avg_add','sum'), lgx=('lg_eff', lambda s: 0)).drop(columns='lgx')
    rsv = X[~X.playoffs].groupby('pid').val.sum(); pov = X[X.playoffs].groupby('pid').val.sum()
    pop = X[X.playoffs].groupby('pid').pts.sum(); poch = X[X.playoffs].groupby('pid').ch.sum()
    poadd = X[X.playoffs].groupby('pid').avg_add.sum()
    lgch = (X.lg_eff*X.ch).groupby(X.pid).sum()
    lgts = (X.lg_ts*X.tsa).groupby(X.pid).sum()
    g['rs_val'] = rsv; g['po_val'] = pov.reindex(g.index).fillna(0)
    g['po_pts'] = pop.reindex(g.index).fillna(0); g['po_ch'] = poch.reindex(g.index).fillna(0)
    g['po_add'] = poadd.reindex(g.index).fillna(0)
    g['eff'] = g.pts/g.ch; g['lg_eff'] = lgch/g.ch
    g['ts'] = g.pts/(2*g.tsa); g['lg_ts'] = lgts/g.tsa
    nn = Nn.groupby('pid').agg(p=('pts','sum'), t=('tsa','sum'))
    g['ts_non'] = (nn.p/(2*nn.t)).reindex(g.index)
    # model version (backtested best): clutch efficiency shrunk and blended 50/50 with normal efficiency
    Xr = X[~X.playoffs]
    chr_ = Xr.groupby('pid').ch.sum()
    relc = (Xr.pts - Xr.lg_eff*Xr.ch).groupby(Xr.pid).sum()/chr_
    reln = (Nn.pts - Nn.lg_eff*Nn.ch).groupby(Nn.pid).sum()/Nn.groupby('pid').ch.sum()
    shr = relc*chr_/(chr_+K)
    g['rel_clutch'] = relc; g['rel_normal'] = reln.reindex(g.index)
    g['eff_est'] = 0.5*shr.reindex(g.index) + 0.5*g.rel_normal.fillna(0)
    g['model'] = chr_.reindex(g.index).fillna(0)*(g.eff_est.fillna(0) + DELTA) + PO_W*g.po_val
    g = g[g.index.isin(roster.keys())].copy()
    g['name'] = [roster[i][0] for i in g.index]; g['team'] = [roster[i][1] for i in g.index]
    g = g.sort_values('model', ascending=False)
    g['rank'] = np.arange(1, len(g)+1)
    return g
b3 = board(3); bc = board(0)
pd.set_option('display.width', 250)
cols = ['name','team','pts','ch','eff','lg_eff','ts','rel_clutch','rel_normal','po_val','score','model']
print(b3.head(25)[cols].round(3).to_string())
print(bc.head(12)[cols].round(3).to_string())
meta = dict(delta=round(DELTA, 4), po_weight=PO_W, last_season=f"{LAST}-{str(LAST+1)[2:]}",
            backtest=bt, n_train=int(len(tr)), n_test=int(len(te)), roster_players=len(roster))
json.dump(meta, open(OUTD+'summary2.json', 'w'), indent=1)
b3.to_csv(OUTD+'board3.csv'); bc.to_csv(OUTD+'boardc.csv')
# season-by-season clutch for current players (for the card)
S = C[C.pid.isin(roster.keys())].groupby(['pid','yr']).agg(pts=('pts','sum'), ch=('ch','sum'), val=('wval','sum')).reset_index()
S.to_csv(OUTD+'cur_seasons.csv', index=False)
