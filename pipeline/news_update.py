"""Latest Ghost Screen Analytics posts from Substack for the site's home and News pages.

    python3 pipeline/news_update.py      # writes data/news.json (keeps the old file if Substack can't be reached)

Substack sometimes refuses GitHub's servers, so this tries the post list first, then the public RSS feed,
on both the current and the old address.
"""
import os, json, urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "news.json")
SITE = "https://ghostscreenanalytics.substack.com"
HOSTS = [SITE, "https://thelowpost.substack.com"]
HEAD = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "application/json, application/rss+xml, application/xml, text/xml, */*", "Accept-Language": "en-US,en;q=0.9"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=30).read()


def link(u):
    return (u or "").replace("thelowpost.substack.com", "ghostscreenanalytics.substack.com")


def from_api(host):
    return rows_of(json.loads(get(host + "/api/v1/archive?sort=new&limit=30")))


def rows_of(posts):
    return [dict(title=p.get("title", ""), subtitle=p.get("subtitle") or p.get("description") or "", date=p.get("post_date"),
                 url=link(p.get("canonical_url")), image=p.get("cover_image")) for p in posts
            if p.get("audience") in ("everyone", "only_free", None) and p.get("type") in ("newsletter", "podcast", None) and p.get("canonical_url")]


def from_reader(host):
    # Substack blocks GitHub's servers, so this reads the same post list through a public reader service
    url = "https://r.jina.ai/" + host + "/api/v1/archive?sort=new&limit=30"
    req = urllib.request.Request(url, headers=dict(HEAD, **{"X-Return-Format": "text", "X-No-Cache": "true"}))
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    return rows_of(json.loads(raw[raw.index("["):raw.rindex("]") + 1]))


def from_feed(host):
    rows = []
    for it in ET.fromstring(get(host + "/feed")).iter("item"):
        when = parsedate_to_datetime(it.findtext("pubDate")).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        pic = it.find("enclosure")
        rows.append(dict(title=(it.findtext("title") or "").strip(), subtitle=(it.findtext("description") or "").strip(), date=when,
                         url=link(it.findtext("link")), image=pic.get("url") if pic is not None else None))
    return rows


def main():
    rows = None
    for read, host in [(from_api, h) for h in HOSTS] + [(from_feed, h) for h in HOSTS] + [(from_reader, SITE)]:
        try:
            rows = read(host)
            if rows:
                print("read", len(rows), "posts with", read.__name__, "from", host); break
        except Exception as e:
            print(read.__name__, host, "failed:", str(e)[:100])
    if not rows:
        print("Substack unavailable, keeping the old list"); return
    new = dict(source=SITE, posts=rows)
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if old.get("posts") == rows:
        print("newsletter list unchanged"); return
    json.dump(new, open(OUT, "w"), indent=1)
    print("newsletter list updated:", len(rows), "posts")


if __name__ == "__main__":
    main()
