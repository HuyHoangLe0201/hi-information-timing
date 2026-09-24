r"""Can a reader follow the notation, or must they already know it?

The readability audit measures sentences.  This measures the other half of
comprehension: whether every symbol is introduced before it is leaned on, and
whether a figure or table can be read without hunting through the body for what
its columns mean.

Three checks.

  symbols     every math symbol, with the line it first appears on and whether
              a definition marker ("write", "let", "denote", "is the", "=")
              occurs in the same sentence.  A symbol whose first appearance
              carries no such marker is being assumed.

  captions    a float caption that names a symbol the caption does not itself
              explain, so the reader has to leave the figure to read it.

  jargon      terms this paper uses as if standard which a reader outside
              condition monitoring or estimation theory would not have.  The
              list is hand-built, because no tool knows what is jargon.
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "paper", "mssp.tex")
raw = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")
body = raw.split("\\begin{document}")[-1].split("\\begin{thebibliography}")[0]

# --- 1. symbols ---------------------------------------------------------------
# Collect single-letter and short macro symbols out of every math span.
SPANS = []
for m in re.finditer(r"\$([^$]+)\$", body):
    SPANS.append((m.start(), m.group(1)))
for env in ("equation", "align"):
    for m in re.finditer(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (env, env),
                         body, re.S):
        SPANS.append((m.start(), m.group(1)))
SPANS.sort()

GREEK = (r"alpha|beta|gamma|delta|epsilon|varepsilon|zeta|eta|theta|iota|kappa|"
         r"lambda|mu|nu|xi|rho|sigma|tau|phi|varphi|chi|psi|omega|Gamma|Delta|"
         r"Theta|Lambda|Xi|Pi|Sigma|Phi|Psi|Omega")
SYM = re.compile(r"\\(%s)\b|(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])" % GREEK)

first = {}
for pos, tex in SPANS:
    for m in SYM.finditer(tex):
        s = m.group(1) or m.group(2)
        if s in ("d", "e", "i", "j", "k", "l", "m", "n", "p", "q", "r", "t",
                 "u", "v", "w", "x", "y", "z"):
            continue                      # indices and dummies, not notation
        first.setdefault(s, pos)

DEFN = re.compile(r"\b(write|writing|let|denote|denotes|denoting|is the|are the|"
                  r"be the|with|where|define|defined|call|is a|stands for)\b",
                  re.I)


def sentence_around(pos):
    a = body.rfind(".", max(0, pos - 400), pos)
    b = body.find(".", pos)
    return body[(a + 1 if a >= 0 else max(0, pos - 400)):
                (b + 1 if b >= 0 else pos + 200)]


print("=" * 88)
print("SYMBOLS: is each one introduced where it first appears?")
print("=" * 88)
undef = []
for s, pos in sorted(first.items(), key=lambda kv: kv[1]):
    sent = sentence_around(pos)
    ln = body[:pos].count("\n") + raw[:raw.index(body)].count("\n") + 1
    if not DEFN.search(sent) and "=" not in sent:
        undef.append((s, ln, " ".join(sent.split())[:96]))
print("%d distinct symbols; %d first appear with no definition marker\n"
      % (len(first), len(undef)))
for s, ln, sent in undef:
    print("  %-12s line %-6d %s" % (s, ln, sent))

# --- 2. captions --------------------------------------------------------------
print()
print("=" * 88)
print("CAPTIONS: can each float be read without leaving it?")
print("=" * 88)
caps = re.findall(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.S)
print("%d captions\n" % len(caps))
for c in caps:
    flat = " ".join(c.split())
    syms = set()
    for m in re.finditer(r"\$([^$]+)\$", flat):
        for mm in SYM.finditer(m.group(1)):
            syms.add(mm.group(1) or mm.group(2))
    bare = [s for s in syms
            if not re.search(r"\b(is|are|the|of)\b[^.]{0,40}\$?\\?%s" % re.escape(s), flat)]
    words = len(flat.split())
    mark = ""
    if words < 12:
        mark = "  <-- very short"
    elif bare:
        mark = "  <-- uses %s without saying what it is" % ", ".join(sorted(bare)[:4])
    print("  [%3d words]%s" % (words, mark))
    print("      " + (flat if len(flat) < 150 else flat[:147] + "..."))

# --- 3. jargon ----------------------------------------------------------------
print()
print("=" * 88)
print("JARGON: terms used as if standard")
print("=" * 88)
TERMS = [
    "Rayleigh quotient", "Schur complement", "generalised eigenvector",
    "leverage", "Wishart", "van Trees", "Cram\\'er--Rao", "delta method",
    "Savitzky--Golay", "out-of-fold", "surrogate", "spectral kurtosis",
    "kurtogram", "cyclostationary", "envelope analysis", "newsvendor",
    "participation ratio", "compositional", "nuisance parameter",
    "profiling", "integrated autocorrelation", "information inequality",
    "location Fisher information", "budget fraction", "surviving fraction",
    "tail factor", "censoring efficiency", "trendability", "prognosability",
]
EXPLAIN = re.compile(r"\b(that is|which is|i\.e\.|namely|in other words|the "
                     r"\w+ of|defined|meaning|the fraction|the ratio)\b", re.I)
for t in TERMS:
    pat = re.escape(t)
    hits = [m.start() for m in re.finditer(pat, body)]
    if not hits:
        continue
    sent = sentence_around(hits[0])
    ok = bool(EXPLAIN.search(sent))
    ln = body[:hits[0]].count("\n") + raw[:raw.index(body)].count("\n") + 1
    print("  %-32s %2d uses, first at line %-6d %s"
          % (t.replace("\\'", ""), len(hits), ln,
             "explained there" if ok else "NOT explained there"))
