"""Latest Ghost Screen Analytics posts from Substack for the site's home and News pages.

    python3 pipeline/news_update.py      # writes data/news.json (keeps the old file if Substack can't be reached)
"""
import os, json, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "news.json")
URL = "https://ghostscreenanalytics.substack.com/api/v1/archive?sort=new&limit=30"


def main():
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Ghost Screen Analytics site refresh)"})
        posts = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as e:
        print("Substack unavailable, keeping the old list:", str(e)[:120]); return
    rows = [dict(title=p.get("title", ""), subtitle=p.get("subtitle") or p.get("description") or "", date=p.get("post_date"),
                 url=p.get("canonical_url").replace("thelowpost.substack.com", "ghostscreenanalytics.substack.com"), image=p.get("cover_image")) for p in posts
            if p.get("audience") in ("everyone", "only_free", None) and p.get("type") in ("newsletter", "podcast", None) and p.get("canonical_url")]
    if not rows:
        print("no posts returned, keeping the old list"); return
    new = dict(source="https://ghostscreenanalytics.substack.com", posts=rows)
    old = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if old.get("posts") == rows:
        print("newsletter list unchanged"); return
    json.dump(new, open(OUT, "w"), indent=1)
    print("newsletter list updated:", len(rows), "posts")


if __name__ == "__main__":
    main()
