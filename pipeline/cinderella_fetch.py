"""Download for the Cinderella model: every Division I men's game score from ESPN's public scoreboard, 2012-13 onward.

    python3 pipeline/cinderella_fetch.py     # writes cinderella-model/raw/games_<season>.csv
"""
import os, csv, json, time, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "cinderella-model", "raw")
os.makedirs(OUT, exist_ok=True)
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={}&groups=50&limit=400"
HEAD = {"User-Agent": "GhostScreenAnalytics-data-refresh"}


def day(d):
    j = json.loads(urllib.request.urlopen(urllib.request.Request(SB.format(d.strftime("%Y%m%d")), headers=HEAD), timeout=40).read())
    rows = []
    for e in j.get("events", []):
        c = e["competitions"][0]
        if not c.get("status", {}).get("type", {}).get("completed"):
            continue
        t = {x["homeAway"]: x for x in c["competitors"]}
        if len(t) != 2:
            continue
        h, a = t["home"], t["away"]
        note = " ".join(n.get("headline", "") for n in c.get("notes", []))
        rank = lambda x: (x.get("curatedRank") or {}).get("current", "")
        rows.append([d.isoformat(), e["id"], c.get("tournamentId", ""), int(bool(c.get("neutralSite"))), e.get("season", {}).get("type", ""), note,
                     h["team"]["id"], h["team"].get("location", ""), h["team"].get("conferenceId", ""), rank(h), h.get("score", ""),
                     a["team"]["id"], a["team"].get("location", ""), a["team"].get("conferenceId", ""), rank(a), a.get("score", "")])
    return rows


def main():
    today = datetime.date.today()
    for season in range(2013, 2028):
        if season == 2020:
            pass
        path = os.path.join(OUT, "games_{}.csv".format(season))
        if os.path.exists(path) and season < 2027:
            continue
        rows, bad = [], 0
        d, end = datetime.date(season - 1, 11, 1), min(datetime.date(season, 4, 12), today)
        while d <= end:
            for attempt in range(3):
                try:
                    rows += day(d); break
                except Exception as e:
                    if attempt == 2:
                        bad += 1; print(d, "failed", str(e)[:80])
                    time.sleep(1)
            d += datetime.timedelta(days=1); time.sleep(.05)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow("date id tourney neutral stype note hid hteam hconf hrank hpts aid ateam aconf arank apts".split()); w.writerows(rows)
        print(season, len(rows), "games", bad, "bad days")


if __name__ == "__main__":
    main()
