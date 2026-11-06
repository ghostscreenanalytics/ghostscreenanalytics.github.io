"""Write data/clutch.json for the current-player Clutch Score page."""
import pandas as pd, numpy as np, json, sys
D = sys.argv[1] if len(sys.argv) > 1 else 'pipeline/out/clutch_'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data/clutch.json'
meta = json.load(open(D+'summary2.json'))
old = json.load(open(D+'summary.json'))
DELTA = meta['delta']
S = pd.read_csv(D+'cur_seasons.csv')
lab = lambda y: f"{int(y)}-{str(int(y)+1)[2:]}"
r1 = lambda x: None if pd.isna(x) else round(float(x), 1)
r3 = lambda x: None if pd.isna(x) else round(float(x), 3)

def pack(path, minch):
    b = pd.read_csv(path, index_col=0)
    b = b[b.ch >= minch].copy()
    ch_rs = b.ch - b.po_ch
    b['c_vol'] = ch_rs*DELTA; b['c_eff'] = ch_rs*b.eff_est.fillna(0); b['c_big'] = meta['po_weight']*b.po_val
    b = b.sort_values('model', ascending=False); b['rank'] = np.arange(1, len(b)+1)
    out = []
    for pid, r in b.iterrows():
        out.append(dict(id=int(pid), n=r['name'], t=r.team, rk=int(r['rank']), sc=r1(r.model),
            vol=r1(r.c_vol), eff=r1(r.c_eff), big=r1(r.c_big),
            pts=int(r.pts), ch=r1(r.ch), g=int(r.games), pe=r3(r.eff), lpe=r3(r.lg_eff),
            ts=r3(r.ts), lts=r3(r.lg_ts), tsn=r3(r.ts_non), rc=r1(r.rel_clutch*100) if pd.notna(r.rel_clutch) else None,
            rn=r1(r.rel_normal*100) if pd.notna(r.rel_normal) else None,
            fg=f"{int(r.fgm)}-{int(r.fga)}", tp=f"{int(r.tpm)}-{int(r.tpa)}", ft=f"{int(r.ftm)}-{int(r.fta)}",
            tov=int(r.tov), ast=int(r.ast), ppts=int(r.po_pts), pch=r1(r.po_ch), padd=r1(r.po_add), raw=r1(r.score)))
    return out
now = pack(D+'board3.csv', 25)
car = pack(D+'boardc.csv', 25)
ser = {}
for pid, g in S.groupby('pid'):
    ser[int(pid)] = [[lab(y), int(p), r1(v)] for y, p, v in zip(g.yr, g.pts, g.val)]
meta.update(dict(window='2023-24 to 2025-26', definition=old['definition'],
    is_clutch_real=dict(backtest=old['backtest'], split_half=old['split_half'], n_test=old['n_test']),
    updated='2026-11-06'))
json.dump(dict(meta=meta, now=now, career=car, series=ser), open(OUT, 'w'), separators=(',', ':'))
print(OUT, len(now), len(car))
for p in now[:10]: print(p['rk'], p['n'], p['sc'], p['vol'], p['eff'], p['big'], p['pts'])
