"""Backup live source for the forecast: ESPN's public scoreboard and box scores.

cdn.nba.com refuses requests from GitHub's servers, so the daily run falls back to this. Games get ids "E<espn id>";
players are matched to NBA.com ids by name (and team) using inputs/rosters_2026.json and any names files in out/.
Unmatched players get a negative id so they still count for their team.
"""
import os, re, json, glob, unicodedata, urllib.request, datetime
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}"
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN"}
START, END = datetime.date(2026, 10, 20), datetime.date(2027, 4, 12)
TEAM_IDS = {1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA", 1610612741: "CHI", 1610612739: "CLE",
            1610612742: "DAL", 1610612743: "DEN", 1610612765: "DET", 1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
            1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612748: "MIA", 1610612749: "MIL", 1610612750: "MIN",
            1610612740: "NOP", 1610612752: "NYK", 1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
            1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR", 1610612762: "UTA", 1610612764: "WAS"}
TEAMS = set(TEAM_IDS.values())


def get(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())


def tri(a):
    return FIX.get(a, a)


def schedule():
    rows, d = [], START
    while d <= END:
        for e in get(SB.format(d.strftime("%Y%m%d"))).get("events", []):
            if e.get("season", {}).get("type") != 2: continue
            c = {x["homeAway"]: x for x in e["competitions"][0]["competitors"]}
            if tri(c["home"]["team"]["abbreviation"]) not in TEAMS or tri(c["away"]["team"]["abbreviation"]) not in TEAMS: continue   # TBD, All-Star
            st = e["status"]["type"]["state"]
            dt = pd.Timestamp(e["date"]).tz_localize(None) - pd.Timedelta(hours=5)
            rows.append(dict(gid="E" + e["id"], date=dt.normalize(), home=tri(c["home"]["team"]["abbreviation"]), away=tri(c["away"]["team"]["abbreviation"]),
                             status=3 if st == "post" else 1, hp=int(c["home"].get("score") or 0) if st == "post" else 0,
                             ap=int(c["away"].get("score") or 0) if st == "post" else 0))
        d += datetime.timedelta(days=1)
    S = pd.DataFrame(rows).drop_duplicates("gid"); S["source"] = "live"
    return S


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[.'\-]", " ", n); n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", n)
    return " ".join(n.split())


_IDX = None
def index():
    """(name, team) -> pid and name -> pid (only when the name is unique)."""
    global _IDX
    if _IDX is None:
        by_team, by_name = {}, {}
        try:
            for r in json.load(open(os.path.join(HERE, "inputs", "rosters_2026.json"))):
                k = norm(r[2]); by_team[(k, TEAM_IDS.get(int(r[0]), "?"))] = int(r[1]); by_name.setdefault(k, set()).add(int(r[1]))
        except Exception as e:
            print("rosters file unreadable:", e)
        for f in sorted(glob.glob(os.path.join(HERE, "out", "names_*.csv"))):
            try:
                n = pd.read_csv(f)
                for p, nm in zip(n.pid, n.name): by_name.setdefault(norm(nm), set()).add(int(p))
            except Exception: pass
        _IDX = (by_team, {k: next(iter(v)) for k, v in by_name.items() if len(v) == 1})
    return _IDX


def pid_for(name, team, espn_id):
    by_team, by_name = index(); k = norm(name)
    return by_team.get((k, team)) or by_name.get(k) or -int(espn_id)


def mins(x):
    try: return float(x)
    except Exception: return 0.0


def split(x):
    try: a, b = str(x).split("-"); return int(a), int(b)
    except Exception: return 0, 0


def box(gid, date):
    """rows in fc_live.COLS format for one ESPN game."""
    j = get(SUM.format(gid[1:])); rows = []
    for t in j["boxscore"]["players"]:
        team = tri(t["team"]["abbreviation"]); st = t["statistics"][0]; keys = st["keys"]; seen = set()
        for a in st["athletes"]:
            ath = a["athlete"]; pid = pid_for(ath["displayName"], team, ath["id"]); seen.add(norm(ath["displayName"]))
            v = dict(zip(keys, a.get("stats") or []))
            played = int(bool(a.get("stats")) and not a.get("didNotPlay"))
            reason = (a.get("reason") or "") if not played else ""
            if reason and "COACH" not in reason.upper(): reason = "INJURY/ILLNESS - " + reason
            fgm, fga = split(v.get("fieldGoalsMade-fieldGoalsAttempted")); f3m, f3a = split(v.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
            ftm, fta = split(v.get("freeThrowsMade-freeThrowsAttempted"))
            num = lambda k: int(float(str(v.get(k, 0)).replace("+", "") or 0)) if str(v.get(k, "0")).lstrip("+-").replace(".", "").isdigit() else 0
            rows.append(dict(gid=gid, date=date, pid=pid, name=ath["displayName"], team=team, starter=int(bool(a.get("starter"))), played=played,
                             reason=reason, min=mins(v.get("minutes")), pts=num("points"), oreb=num("offensiveRebounds"), dreb=num("defensiveRebounds"),
                             ast=num("assists"), stl=num("steals"), blk=num("blocks"), tov=num("turnovers"), fga=fga, fta=fta, f3a=f3a, f3m=f3m,
                             pm=num("plusMinus")))
        # players listed out on the injury report who are not on the box score at all
        for grp in j.get("injuries", []):
            if tri(grp.get("team", {}).get("abbreviation", "")) != team: continue
            for i in grp.get("injuries", []):
                ath = i.get("athlete", {}); nm = ath.get("displayName", "")
                if not nm or norm(nm) in seen or str(i.get("status", "")).lower() != "out": continue
                rows.append(dict(gid=gid, date=date, pid=pid_for(nm, team, ath.get("id", 0) or 0), name=nm, team=team, starter=0, played=0,
                                 reason="INJURY/ILLNESS - out", min=0.0, pts=0, oreb=0, dreb=0, ast=0, stl=0, blk=0, tov=0, fga=0, fta=0, f3a=0, f3m=0, pm=0))
    return rows
