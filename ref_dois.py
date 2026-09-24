r"""Find the DOI of every reference, from Crossref, and record the match.

The MSSP Guide for Authors asks for DOIs "where available", and every published
MSSP paper read for this check prints them.  The reference list carried none.

Each entry is matched by a bibliographic query and accepted only when the
returned title agrees with the entry's own title to a normalised similarity of
0.90 and the year agrees; anything weaker is reported, not used.  Books and
proceedings without a DOI are listed as such.

    python ref_dois.py   ->  ref_dois.json
"""
import difflib
import io
import json
import os
import re
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
B = chr(92)
UA = {"User-Agent": "mssp-reference-check/1.0"}


def norm(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = re.sub(r"[{}$~\\'`\"^]", "", s)
    s = s.replace("---", " ").replace("--", " ")
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def entries():
    s = io.open(TEX, encoding="utf-8").read()
    bib = s[s.index(B + "begin{thebibliography}"):s.index(B + "end{thebibliography}")]
    out = []
    for chunk in bib.split(B + "bibitem")[1:]:
        key = re.match(r"\{([^}]+)\}", chunk).group(1)
        body = re.sub(r"\s+", " ", chunk[chunk.index("}") + 1:]).strip()
        parts = [p.strip() for p in body.split(",")]
        # a four-digit page number is not a year: take a parenthesised 19xx/20xx
        # first, a bare one only if there is none
        par = re.findall(r"\(((?:19|20)\d{2})\)", body)
        bare = re.findall(r"\b((?:19|20)\d{2})\b", body)
        yr = par[-1] if par else (bare[-1] if bare else "")
        out.append(dict(key=key, text=body, year=yr))
    return out


def title_of(text):
    """The title is the first comma-separated field after the author block."""
    # author block ends at the first field containing a lowercase word of 4+ letters
    fields = [f.strip() for f in text.split(",")]
    for i, f in enumerate(fields):
        if re.search(r"\b[a-z]{4,}\b", f) and not re.match(r"^(and|in:)\b", f):
            return ", ".join(fields[i:i + 2]) if len(f.split()) < 4 else f
    return text


def query(text):
    q = urllib.parse.urlencode({"query.bibliographic": text[:300], "rows": 3})
    url = "https://api.crossref.org/works?" + q
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(
                urllib.request.Request(url, headers=UA), timeout=60))["message"]["items"]
        except Exception:
            time.sleep(3 + 3 * attempt)
    return []


def main():
    res = []
    for e in entries():
        items = query(e["text"])
        best, score = None, 0.0
        mine = " ".join(norm(e["text"]))
        for it in items:
            t = " ".join(norm((it.get("title") or [""])[0]))
            if not t:
                continue
            # the entry text contains the title somewhere; compare against the best window
            sc = difflib.SequenceMatcher(None, t, mine).find_longest_match(
                0, len(t), 0, len(mine)).size / max(len(t), 1)
            y = str((it.get("issued", {}).get("date-parts") or [[None]])[0][0])
            if sc > score and (not e["year"] or y == e["year"]):
                best, score = it, sc
        ok = best is not None and score >= 0.90
        rec = dict(key=e["key"], year=e["year"], score=round(score, 3),
                   doi=best.get("DOI") if ok else None,
                   container=(best.get("container-title") or [""])[0] if ok else None,
                   short=(best.get("short-container-title") or [""])[0] if ok else None)
        res.append(rec)
        print("%-16s %s  %.2f  %s" % (e["key"], "ok " if ok else "-- ",
                                      score, rec["doi"] or ""))
        time.sleep(1.0)
    json.dump(res, open(os.path.join(HERE, "ref_dois.json"), "w"), indent=1)
    print("%d of %d matched" % (sum(1 for r in res if r["doi"]), len(res)))


if __name__ == "__main__":
    main()
