"""Daily Joy Rankings update. Reads finished games from ESPN's public scoreboard and box scores (NBA.com refuses
GitHub's servers), scores every player, and keeps data/joy.json holding the games the sliders can reach.

Joy = Game Score + threes + steals + blocks + made dunks. Rows are:
[name index, team, date, opponent, playoffs, won, pts, reb, ast, stl, blk, 3pm, dunks, game score, game id]
Old rows use NBA.com game ids (numbers). New rows use "E<espn id>".
"""
import os, re, json, datetime, unicodedata, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "..", "data", "joy.json")
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={}"
SUM = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
FIX = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BRK": "BKN"}
KEEP = 105          # games kept for each on/off mix of the four bonuses
TYPES = {2, 3, 5}   # regular season, playoffs, play-in


def get(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read())


def norm(n):
    n = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"[.'\-]", " ", n); n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", n)
    return " ".join(n.split())


def score(r, m):
    return r[13] + (m & 1) * r[11] + (m >> 1 & 1) * r[9] + (m >> 2 & 1) * r[10] + (m >> 3 & 1) * r[12]


def game_rows(e, D, names):
    """Rows for every player in one finished game."""
    comp = e["competitions"][0]
    side = {FIX.get(c["team"]["abbreviation"], c["team"]["abbreviation"]): c for c in comp["competitors"]}
    s = get(SUM.format(e["id"]))
    dunks = {}
    for p in s.get("plays", []):
        txt = (p.get("text") or "") + " " + ((p.get("type") or {}).get("text") or "")
        if p.get("scoringPlay") and re.search("dunk", txt, re.I) and p.get("participants"):
            pid = str(p["participants"][0]["athlete"]["id"])
            dunks[pid] = dunks.get(pid, 0) + 1
    date = (datetime.datetime.strptime(e["date"][:16], "%Y-%m-%dT%H:%M") - datetime.timedelta(hours=5)).strftime("%Y%m%d")
    playoff = 1 if e.get("season", {}).get("type") == 3 else 0
    rows = []
    for t in s["boxscore"]["players"]:
        team = FIX.get(t["team"]["abbreviation"], t["team"]["abbreviation"])
        opp = [k for k in side if k != team][0]
        home = side[team].get("homeAway") == "home"
        won = 1 if int(side[team].get("score") or 0) > int(side[opp].get("score") or 0) else 0
        st = t["statistics"][0]
        keys = st["keys"]
        for a in st["athletes"]:
            if a.get("didNotPlay") or not a.get("stats"):
                continue
            v = dict(zip(keys, a["stats"]))
            try:
                pts, reb, ast, tov, stl, blk = [int(v[k]) for k in ("points", "rebounds", "assists", "turnovers", "steals", "blocks")]
                oreb, dreb, pf = [int(v[k]) for k in ("offensiveRebounds", "defensiveRebounds", "fouls")]
                fgm, fga = map(int, v["fieldGoalsMade-fieldGoalsAttempted"].split("-"))
                tpm, _ = map(int, v["threePointFieldGoalsMade-threePointFieldGoalsAttempted"].split("-"))
                ftm, fta = map(int, v["freeThrowsMade-freeThrowsAttempted"].split("-"))
            except (KeyError, ValueError):
                continue
            gs = pts + .4 * fgm - .7 * fga - .4 * (fta - ftm) + .7 * oreb + .3 * dreb + stl + .7 * ast + .7 * blk - .4 * pf - tov
            name = a["athlete"]["displayName"]
            k = names.get(norm(name))
            if k is None:
                k = len(D["n"]); D["n"].append(name); names[norm(name)] = k
            rows.append([k, team, date, ("v" if home else "@") + opp, playoff, won, pts, reb, ast, stl, blk, tpm,
                         dunks.get(str(a["athlete"]["id"]), 0), round(gs, 1), "E" + e["id"]])
    return rows


def main():
    D = json.load(open(PATH, encoding="utf-8"))
    names = {norm(n): i for i, n in enumerate(D["n"])}
    seen = set(D.get("seen", []))
    day = datetime.datetime.strptime(D["through"], "%Y-%m-%d").date()
    today = (datetime.datetime.utcnow() - datetime.timedelta(hours=5)).date()
    new, last = [], day
    d = day + datetime.timedelta(days=1)
    while d < today:
        events = get(SB.format(d.strftime("%Y%m%d"))).get("events", [])
        done = [e for e in events if e.get("season", {}).get("type") in TYPES]
        if all(e["status"]["type"]["state"] == "post" for e in done):
            for e in done:
                if e["id"] not in seen:
                    new += game_rows(e, D, names); seen.add(e["id"])
            last = d
        else:
            break                       # a game is unfinished: try this day again tomorrow
        d += datetime.timedelta(days=1)
    pool = D["r"] + new
    keep = set()
    for m in range(16):
        for i in sorted(range(len(pool)), key=lambda i: -score(pool[i], m))[:KEEP]:
            keep.add(i)
    D["r"] = [r for i, r in enumerate(pool) if i in keep]
    D["through"] = last.isoformat()
    D["seen"] = sorted(seen)[-200:]
    used = sorted({r[0] for r in D["r"]})
    remap = {o: n for n, o in enumerate(used)}
    D["n"] = [D["n"][o] for o in used]
    for r in D["r"]:
        r[0] = remap[r[0]]
    json.dump(D, open(PATH, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    print(f"Joy: {len(new)} player games read, through {D['through']}, {len(D['r'])} kept")


if __name__ == "__main__":
    main()
