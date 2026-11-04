"""One-off download for the Cinderella model: team ratings by season and NCAA tournament games with seeds.

    python3 pipeline/cinderella_fetch.py     # writes cinderella-model/raw/
"""
import os, csv, json, time, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "cinderella-model", "raw")
os.makedirs(OUT, exist_ok=True)
HEAD = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
SB = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={}&groups=100&limit=300"


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=40).read()


def games(year):
    rows, d = [], datetime.date(year, 3, 10)
    while d <= datetime.date(year, 4, 12):
        try:
            j = json.loads(get(SB.format(d.strftime("%Y%m%d"))))
        except Exception as e:
            print(d, "failed", str(e)[:80]); d += datetime.timedelta(days=1); continue
        for e in j.get("events", []):
            c = e["competitions"][0]
            note = " ".join(n.get("headline", "") for n in c.get("notes", []))
            if "NCAA" not in note and "Men's Basketball Championship" not in note:
                continue
            if "NIT" in note or "CBI" in note:
                continue
            t = c["competitors"]
            if len(t) != 2:
                continue
            seed = lambda x: (x.get("curatedRank") or {}).get("current", "")
            rows.append([year, d.isoformat(), note, t[0]["team"].get("location", ""), seed(t[0]), t[0].get("score", ""),
                         t[1]["team"].get("location", ""), seed(t[1]), t[1].get("score", "")])
        d += datetime.timedelta(days=1); time.sleep(.1)
    return rows


def main():
    allg = []
    for y in list(range(2008, 2028)):
        try:
            b = get("https://barttorvik.com/{}_team_results.csv".format(y))
            open(os.path.join(OUT, "torvik_{}.csv".format(y)), "wb").write(b)
            print("torvik", y, len(b))
        except Exception as e:
            print("torvik", y, "failed", str(e)[:100])
        if 2008 <= y <= 2026 and y != 2020:
            r = games(y); allg += r; print("tournament", y, len(r))
    with open(os.path.join(OUT, "tournament_games.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow("year date note team1 seed1 score1 team2 seed2 score2".split()); w.writerows(allg)


if __name__ == "__main__":
    main()
