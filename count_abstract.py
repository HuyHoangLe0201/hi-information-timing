"""Word count of the abstract, against the LSENS length budget."""
import re
import os

TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "lsens_letter_v2", "lsens_letter.tex")
t = open(TEX, encoding="utf-8").read()

m = re.search(r"\\begin\{abstract\}\[gabs\](.*?)\\end\{abstract\}", t, re.S)
if not m:
    raise SystemExit("abstract block not found")
a = re.sub(r"\\[a-zA-Z]+", " ", m.group(1))
a = re.sub(r"[{}$\\]", " ", a)
words = [x for x in a.split() if any(c.isalnum() for c in x)]
# LSENS points at a separate abstract-requirements document that could not be
# retrieved; 150 is the strictest limit in use across IEEE letters-format
# journals, so it is the bound enforced here.
LIMIT = 150
status = "ok" if len(words) <= LIMIT else "OVER LIMIT"
print(f"abstract: {len(words)} words (limit {LIMIT})  {status}")

k = re.search(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", t, re.S)
if k:
    terms = [s.strip() for s in k.group(1).split(",") if s.strip()]
    print(f"index terms: {len(terms)} -> {'; '.join(terms)}")

refs = len(re.findall(r"\\bibitem\{", t))
print(f"references: {refs}")
