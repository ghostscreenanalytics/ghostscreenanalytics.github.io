"""GSA Clutch Model.
Clutch Points Added (CPA): points scored in the clutch minus what a league-average
clutch shooter scores on the same shot volume, minus turnovers valued at a
league-average possession. Per-season league baselines (regular season and playoffs separate).
Backtest: does clutch efficiency repeat? Train <=2012-13, test 2013-14..2024-25.
"""
import pandas as pd, numpy as np, json

d = pd.read_csv('pipeline/out/clutch_player_seasons.csv')
d['tsa'] = d.fga + 0.44*d.fta
# canonical name per pid = most recent
name = d.sort_values('yr').groupby('pid')['name'].last()
d['name'] = d.pid.map(name)

# league baselines per season x playoffs x clutch
lg = d.groupby(['yr','playoffs','clutch']).agg(pts=('pts','sum'), tsa=('tsa','sum'),
                                               tov=('tov','sum'), fga=('fga','sum'), fta=('fta','sum')).reset_index()
lg['ppt'] = lg.pts/lg.tsa                       # points per true shot attempt
lg['ppp'] = lg.pts/(lg.tsa + lg.tov)            # points per scoring possession (shot or TO)
d = d.merge(lg[['yr','playoffs','clutch','ppt','ppp']], on=['yr','playoffs','clutch'])
d['exp'] = d.ppt*d.tsa
d['cpa'] = (d.pts - d.exp) - d.tov*d.ppp
d['rel'] = d.pts/d.tsa.replace(0,np.nan) - d.ppt   # efficiency vs league, per TSA

C = d[d.clutch]; N = d[~d.clutch]

# ---------------- career leaderboard ----------------
def career(df):
    g = df.groupby('pid').agg(name=('name','first'), pts=('pts','sum'), tsa=('tsa','sum'),
        fgm=('fgm','sum'), fga=('fga','sum'), tpm=('tpm','sum'), tpa=('tpa','sum'),
        ftm=('ftm','sum'), fta=('fta','sum'), tov=('tov','sum'), ast=('ast','sum'),
        exp=('exp','sum'), cpa=('cpa','sum'), games=('games','sum'),
        first=('yr','min'), last=('yr','max'))
    g['ts'] = g.pts/(2*g.tsa)
    g['lg_ts'] = g.exp/(2*g.tsa)
    return g
car_all = career(C)
car_rs = career(C[~C.playoffs]); car_po = career(C[C.playoffs])
noncl = career(N)
car_all['ts_nonclutch'] = noncl.ts.reindex(car_all.index)
car_all['rs_cpa'] = car_rs.cpa.reindex(car_all.index).fillna(0)
car_all['po_cpa'] = car_po.cpa.reindex(car_all.index).fillna(0)
car_all['rs_pts'] = car_rs.pts.reindex(car_all.index).fillna(0)
car_all['po_pts'] = car_po.pts.reindex(car_all.index).fillna(0)
# rise: clutch efficiency vs his own non-clutch efficiency (each vs its own league baseline)
nrel = (N.groupby('pid').pts.sum() - N.groupby('pid').exp.sum())/N.groupby('pid').tsa.sum()
crel = (car_all.pts - car_all.exp)/car_all.tsa
car_all['rise'] = (crel - nrel.reindex(car_all.index))*100  # pts per 100 TSA

# ---------------- backtest ----------------
# For each player-season t+1 (regular season, >=40 clutch TSA), predict clutch rel efficiency
# from info through season t only.
rs = d[~d.playoffs]
cs = rs[rs.clutch].groupby(['pid','yr'])[['pts','exp','tsa']].sum()
cs['rel'] = (cs.pts - cs.exp)/cs.tsa.replace(0, np.nan)
ns = rs[~rs.clutch].groupby(['pid','yr'])[['pts','exp','tsa']].sum()
recs = []
pids = cs.index.get_level_values(0).unique()
cs_by = {p: g.droplevel(0) for p, g in cs.groupby(level=0)}
ns_by = {p: g.droplevel(0) for p, g in ns.groupby(level=0)}
for p, g in cs_by.items():
    n = ns_by.get(p)
    for y in g.index:
        if g.loc[y,'tsa'] < 40: continue
        prior_c = g[g.index < y];
        if len(prior_c) == 0 or prior_c.tsa.sum() < 20: continue
        prior_n = n[n.index < y] if n is not None else None
        if prior_n is None or prior_n.tsa.sum() < 200: continue
        recs.append(dict(pid=p, yr=y, tsa=g.loc[y,'tsa'], y_rel=g.loc[y,'rel'],
            c_tsa=prior_c.tsa.sum(), c_rel=(prior_c.pts.sum()-prior_c.exp.sum())/prior_c.tsa.sum(),
            n_tsa=prior_n.tsa.sum(), n_rel=(prior_n.pts.sum()-prior_n.exp.sum())/prior_n.tsa.sum()))
B = pd.DataFrame(recs)
tr = B[B.yr <= 2012]; te = B[B.yr >= 2013]

def wrmse(pred, df): return float(np.sqrt(np.average((df.y_rel - pred)**2, weights=df.tsa)))
def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    return float(np.average(a*b, weights=w)/np.sqrt(np.average(a*a, weights=w)*np.average(b*b, weights=w)))

# fit shrinkage k for clutch history: est = c_rel * c_tsa/(c_tsa+k)
best = None
for k in [0,25,50,100,150,200,300,400,600,800,1200,2000,4000,1e9]:
    pr = tr.c_rel*tr.c_tsa/(tr.c_tsa+k)
    e = wrmse(pr, tr)
    if best is None or e < best[1]: best = (k, e)
K = best[0]
# blend: y = a*shrunk_clutch + b*nonclutch, weighted least squares on train
X = np.c_[tr.c_rel*tr.c_tsa/(tr.c_tsa+K), tr.n_rel]
W = np.sqrt(tr.tsa.values)
coef = np.linalg.lstsq(X*W[:,None], tr.y_rel.values*W, rcond=None)[0]
Xn = np.c_[tr.n_rel]; coef_n = np.linalg.lstsq(Xn*W[:,None], tr.y_rel.values*W, rcond=None)[0]

def preds(df):
    sc = df.c_rel*df.c_tsa/(df.c_tsa+K)
    return {'League average (benchmark)': np.zeros(len(df)),
            'Past clutch shooting, raw': df.c_rel.values,
            'Past clutch shooting, shrunk': sc.values,
            'Past non-clutch shooting': coef_n[0]*df.n_rel.values,
            'GSA blend': coef[0]*sc.values + coef[1]*df.n_rel.values}
bt = {}
for nm, pr in preds(te).items():
    bt[nm] = dict(rmse=round(wrmse(pr, te)*100, 2))
base = bt['League average (benchmark)']['rmse']
for nm in bt: bt[nm]['vs_benchmark_pct'] = round((1 - bt[nm]['rmse']/base)*100, 1)

# split-half reliability: odd vs even seasons, players with >=150 clutch TSA in each half
cs2 = rs[rs.clutch]; ns2 = rs[~rs.clutch]
def halves(df):
    df = df.assign(h=df.yr % 2)
    g = df.groupby(['pid','h']).agg(pts=('pts','sum'), exp=('exp','sum'), tsa=('tsa','sum')).reset_index()
    g['rel'] = (g.pts-g.exp)/g.tsa
    w = g.pivot(index='pid', columns='h', values=['rel','tsa']).dropna()
    return w
hc = halves(cs2); hc = hc[(hc['tsa'][0] >= 150) & (hc['tsa'][1] >= 150)]
hn = halves(ns2); hn = hn.loc[hc.index]
rel_clutch = float(np.corrcoef(hc['rel'][0], hc['rel'][1])[0,1])
rel_non = float(np.corrcoef(hn['rel'][0], hn['rel'][1])[0,1])
cross = float(np.corrcoef(hn['rel'][0], hc['rel'][1])[0,1])

# ---------------- modeled career rating ----------------
# GSA Clutch Rating: career clutch volume x skill estimate (blend), i.e. CPA you'd expect again
car_all['nrel'] = nrel.reindex(car_all.index)
sc = crel*car_all.tsa/(car_all.tsa+K)
car_all['skill'] = (coef[0]*sc + coef[1]*car_all.nrel)*100     # pts per 100 clutch TSA vs avg
car_all['skill_cpa'] = car_all.skill/100*car_all.tsa

out = dict(
    definition='4th quarter or OT, 5:00 or less left, score within 5 (NBA.com clutch definition)',
    coverage='NBA.com play-by-play, 1996-97 through 2024-25, regular season and playoffs',
    shrink_k=K, blend=dict(clutch=round(float(coef[0]),3), nonclutch=round(float(coef[1]),3)),
    backtest=bt, n_train=int(len(tr)), n_test=int(len(te)),
    split_half=dict(players=int(len(hc)), clutch_r=round(rel_clutch,3), nonclutch_r=round(rel_non,3),
                    nonclutch_predicts_clutch_r=round(cross,3)))
json.dump(out, open('pipeline/out/clutch_summary.json','w'), indent=1)
car_all.to_csv('pipeline/out/clutch_career.csv')
d.to_csv('pipeline/out/clutch_seasons_scored.csv', index=False)
print(json.dumps(out, indent=1))
cols = ['name','first','last','pts','tsa','ts','lg_ts','cpa','rs_cpa','po_cpa','rise','skill','skill_cpa']
pd.set_option('display.width', 250)
print('\nTOP CAREER CLUTCH POINTS ADDED'); print(car_all.nlargest(20,'cpa')[cols].round(3).to_string())
print('\nTOP CLUTCH POINTS'); print(car_all.nlargest(10,'pts')[cols].round(3).to_string())
print('\nWORST CPA'); print(car_all.nsmallest(8,'cpa')[cols].round(3).to_string())
q = car_all[car_all.tsa >= 800]
print('\nBIGGEST RISE (>=800 clutch TSA)'); print(q.nlargest(10,'rise')[cols].round(3).to_string())
print('\nBIGGEST DROP'); print(q.nsmallest(8,'rise')[cols].round(3).to_string())
print('\nMODELED SKILL CPA'); print(car_all.nlargest(12,'skill_cpa')[cols].round(3).to_string())
