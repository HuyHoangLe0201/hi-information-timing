"""
Two prose checks on the manuscript.

(1) Abbreviations. Every acronym should be expanded at its FIRST occurrence in
    the body, and not re-expanded afterwards. Both failures are reported:
    used-before-defined, and defined-more-than-once.

(2) Repetition. Sentences that repeat an earlier one, either verbatim or as a
    close paraphrase, are flagged by content-word overlap (Jaccard) plus shared
    3-grams, so "says the same thing in different words" is caught too.
"""
import os
import re
from itertools import combinations

TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "mssp.tex")
raw = open(TEX, encoding="utf-8").read()

# ---- keep only the body: abstract through conclusion, drop the bibliography
body = raw.split(r"\begin{abstract}")[1].split(r"\begin{thebibliography}")[0]
body = re.sub(r"(?m)^\s*%.*$", "", body)                 # comment lines
body = re.sub(r"(?<!\\)%.*", "", body)                   # trailing comments
body = re.sub(r"\\begin\{table\}.*?\\end\{table\}", " ", body, flags=re.S)
body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", body, flags=re.S)
body = re.sub(r"\$[^$]*\$", " MATH ", body)              # inline math
body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " MATH ", body, flags=re.S)
body = re.sub(r"\\(?:eq)?ref\{[^}]*\}", " REF ", body)
body = re.sub(r"\\cite(?:\[[^\]]*\])?\{[^}]*\}", " CITE ", body)
body = re.sub(r"\\(?:emph|textit|textbf)\{([^}]*)\}", r"\1", body)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
body = body.replace("{", " ").replace("}", " ").replace("--", "-")
body = re.sub(r"\s+", " ", body).strip()

# ---------------------------------------------------------------- (1)
CANDIDATES = ["RUL", "CNN", "LSTM", "RMS", "IQR", "MSE", "SNR", "ML", "CRB",
              "C-MAPSS", "PRONOSTIA", "FIFO", "OOF", "MAD", "DOI", "EDICS"]
print("=== abbreviations ===")
print(f"{'acronym':<12}{'uses':>6}{'first at char':>15}   status")
print("-" * 74)
for a in CANDIDATES:
    hits = [m.start() for m in re.finditer(rf"\b{re.escape(a)}\b", body)]
    if not hits:
        continue
    # a definition looks like "expansion (ACRONYM)" or "ACRONYM (expansion)"
    defs = [m.start() for m in
            re.finditer(rf"\)\s*\(?\s*{re.escape(a)}\s*\)|\({re.escape(a)}\)", body)]
    defs += [m.start() for m in re.finditer(rf"\b{re.escape(a)}\b\s*\(", body)]
    first_use, first_def = hits[0], (min(defs) if defs else None)
    if first_def is None:
        status = "NEVER DEFINED"
    elif first_def > first_use + len(a) + 2:
        status = f"used before definition (def at {first_def})"
    elif len(defs) > 1:
        status = f"defined {len(defs)}x -- redundant"
    else:
        status = "ok"
    print(f"{a:<12}{len(hits):>6}{first_use:>15}   {status}")

# ---------------------------------------------------------------- (2)
STOP = set("""a an the of to in on for and or but is are was were be been being it its
this that these those with as by from at we our us they their there which who whom
whose than then so such not no nor can could may might must shall should will would
one two three both each every any all more most less least only just also very much
same other another how what when where why if while into over under about
math ref cite""".split())


def words(s):
    return [w for w in re.findall(r"[a-z][a-z-]+", s.lower()) if w not in STOP]


sents = [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", body) if len(s.strip()) > 45]
print(f"\n=== repetition ({len(sents)} sentences compared pairwise) ===")
flagged = []
for i, j in combinations(range(len(sents)), 2):
    a, b = set(words(sents[i])), set(words(sents[j]))
    if len(a) < 5 or len(b) < 5:
        continue
    jac = len(a & b) / len(a | b)
    wa, wb = words(sents[i]), words(sents[j])
    tri_a = {tuple(wa[k:k + 3]) for k in range(len(wa) - 2)}
    tri_b = {tuple(wb[k:k + 3]) for k in range(len(wb) - 2)}
    shared = tri_a & tri_b
    if jac >= 0.26 or len(shared) >= 1:
        flagged.append((jac, len(shared), i, j, sorted(a & b)))

flagged.sort(key=lambda t: (-t[1], -t[0]))
if not flagged:
    print("  no pair above threshold")
for jac, ntri, i, j, common in flagged[:12]:
    print(f"\n  overlap {jac:.2f}, {ntri} shared 3-gram(s)")
    print(f"    [{i}] {sents[i][:165]}")
    print(f"    [{j}] {sents[j][:165]}")
    print(f"    shared: {', '.join(common[:14])}")

# ---- distinctive phrases occurring more than once anywhere in the body ----
print("\n=== repeated 4-grams (content words only) ===")
allw = words(body)
grams = {}
for k in range(len(allw) - 3):
    g = tuple(allw[k:k + 4])
    grams.setdefault(g, []).append(k)
rep = {g: v for g, v in grams.items() if len(v) > 1}
if not rep:
    print("  none")
for g, v in sorted(rep.items(), key=lambda t: -len(t[1])):
    print(f"  x{len(v)}  \"{' '.join(g)}\"")
