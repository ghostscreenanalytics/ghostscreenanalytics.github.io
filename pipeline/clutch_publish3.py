"""data/clutch.json for Clutch Score v2 (offense + playmaking + defense)."""
import pandas as pd, numpy as np, json, sys
D = sys.argv[1] if len(sys.argv) > 1 else 'pipeline/out/clutch_'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data/clutch.json'
meta = json.load(open(D+'summary3.json')); old = json.load(open(D+'summary.json'))
S = pd.read_csv(D+'cur_seasons.csv')
lab = lambda y: f"{int(y)}-{str(int(y)+1)[2:]}"
r1 = lambda x: None if pd.isna(x) else round(float(x), 1)
r3 = lambda x: None if pd.isna(x) else round(float(x), 3)
def pack(path):
    b = pd.read_csv(path, index_col=0); b = b[b.ch >= 25].sort_values('score', ascending=False)
    b['rank'] = np.arange(1, len(b)+1)
    b['orank'] = b.off_score.rank(ascending=False, method='min'); b['drank'] = b.def_total.rank(ascending=False, method='min')
    return [dict(id=int(i), n=r['name'], t=r.team, rk=int(r['rank']), sc=r1(r.score),
        vol=r1(r.vol), eff=r1(r.eff), mk=r1(r.make), pl=r1(r.plays), td=r1(r.team_d), big=r1(r.big),
        off=r1(r.off_score), dfn=r1(r.def_total), ork=int(r.orank), drk=int(r.drank),
        pts=int(r.pts), ch=r1(r.ch), g=int(r.games), pe=r3(r.pe), lpe=r3(r.lpe), ts=r3(r.ts), lts=r3(r.lts), tsn=r3(r.tsn),
        fg=f"{int(r.fgm)}-{int(r.fga)}", tp=f"{int(r.tpm)}-{int(r.tpa)}", ft=f"{int(r.ftm)}-{int(r.fta)}",
        tov=int(r.tov), ast=int(r.ast), orb=int(r.oreb), stl=int(r.stl), blk=int(r.blk), drb=int(r.dreb), chg=int(r.chg), pf=int(r.pf),
        oe=r3(r.opp_eff), ld=r3(r.lg_def), och=r1(r.on_och), ppts=int(r.po_pts))
        for i, r in b.iterrows()]
now = pack(D+'board3v2.csv'); car = pack(D+'boardcv2.csv')
ser = {int(p): [[lab(y), int(pt), r1(v)] for y, pt, v in zip(g.yr, g.pts, g.val)] for p, g in S.groupby('pid')}
meta.update(dict(window='2023-24 to 2025-26', definition=old['definition'],
    is_clutch_real=dict(backtest=old['backtest'], split_half=old['split_half']), updated='2026-11-06'))
json.dump(dict(meta=meta, now=now, career=car, series=ser), open(OUT, 'w'), separators=(',', ':'))
print(len(now), len(car)); [print(p['rk'], p['n'], p['sc'], p['off'], p['dfn']) for p in now[:6]]
