"""Write data/clutch.json for the site from model outputs."""
import pandas as pd, numpy as np, json, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else 'data/clutch.json'
S = pd.read_csv('pipeline/out/clutch_seasons_scored.csv')
car = pd.read_csv('pipeline/out/clutch_career.csv', index_col=0)
summ = json.load(open('pipeline/out/clutch_summary.json'))
C = S[S.clutch].copy(); C['sa'] = C.pts - C.exp
lab = lambda y: f"{y}-{str(y+1)[2:]}"

car['sa'] = car.pts - car.exp
rs = C[~C.playoffs].groupby('pid').sa.sum(); po = C[C.playoffs].groupby('pid').sa.sum()
car['rs_sa'] = rs.reindex(car.index).fillna(0); car['po_sa'] = po.reindex(car.index).fillna(0)
# seasons with positive points added
pos = C.groupby(['pid','yr']).sa.sum().reset_index()
car['seasons'] = pos.groupby('pid').size().reindex(car.index)
car['pos_seasons'] = pos[pos.sa > 0].groupby('pid').size().reindex(car.index).fillna(0)
car['rank_sa'] = car.sa.rank(ascending=False, method='min').astype(int)
car['rank_cpa'] = car.cpa.rank(ascending=False, method='min').astype(int)
n_all = len(car)

P = car[car.tsa >= 100].sort_values('sa', ascending=False)
r1 = lambda x: round(float(x), 1); r3 = lambda x: round(float(x), 3)
players = [dict(id=int(i), n=r['name'], y=f"{lab(int(r['first']))[:4]}-{str(int(r['last'])+1)}",
    g=int(r.games), pts=int(r.pts), fgm=int(r.fgm), fga=int(r.fga), tpm=int(r.tpm), tpa=int(r.tpa),
    ftm=int(r.ftm), fta=int(r.fta), tov=int(r.tov), ast=int(r.ast), tsa=r1(r.tsa),
    ts=r3(r.ts), lg=r3(r.lg_ts), tsn=r3(r.ts_nonclutch) if pd.notna(r.ts_nonclutch) else None,
    sa=r1(r.sa), cpa=r1(r.cpa), rs=r1(r.rs_sa), po=r1(r.po_sa),
    rise=r1(r.rise) if pd.notna(r.rise) else None, sk=r1(r.skill) if pd.notna(r.skill) else None,
    ns=int(r.seasons), np_=int(r.pos_seasons), rk=int(r.rank_sa), rkc=int(r.rank_cpa))
    for i, r in P.iterrows()]

sea = C[~C.playoffs].groupby(['pid','yr']).agg(pts=('pts','sum'), tsa=('tsa','sum'), exp=('exp','sum'),
                                               g=('games','sum')).reset_index()
sea['sa'] = sea.pts - sea.exp; sea['ts'] = sea.pts/(2*sea.tsa); sea['lg'] = sea.exp/(2*sea.tsa)
sea['n'] = sea.pid.map(car.name)
top_seasons = [dict(n=r.n, s=lab(int(r.yr)), pts=int(r.pts), ts=r3(r.ts), lg=r3(r.lg), sa=r1(r.sa), g=int(r.g))
               for r in sea.nlargest(25, 'sa').itertuples()]
most_pts = [dict(n=r.n, s=lab(int(r.yr)), pts=int(r.pts), ts=r3(r.ts), lg=r3(r.lg), sa=r1(r.sa), g=int(r.g))
            for r in sea.nlargest(15, 'pts').itertuples()]

meta = dict(summ)
meta.update(dict(players_all=n_all, players_listed=len(players),
    clutch_fga=int(C.fga.sum()), clutch_fta=int(C.fta.sum()), clutch_pts=int(C.pts.sum()),
    seasons='1996-97 to 2024-25', updated='2026-11-06'))
json.dump(dict(meta=meta, players=players, top_seasons=top_seasons, most_pts=most_pts),
          open(OUT, 'w'), separators=(',', ':'))
print(OUT, len(players), 'players')
print(top_seasons[:5]); print(most_pts[:3])
