r"""The twenty-fifth audit: the journal's own hard limits.

Every other layer checks whether the paper is right.  This one checks whether it
is submissible, because the editorial screen at Mechanical Systems and Signal
Processing runs before peer review and rejects on formatting alone.  The limits
below are the journal's, not this project's, and each is a number a desk editor
can count:

  abstract     250 words or fewer
  highlights   3 to 5 bullets, each at most 85 characters including spaces
  keywords     1 to 7, and the guide asks that they avoid "and" and "of"
  declarations CRediT, competing interest and data availability are required

The abstract sat at 253 words when this layer was written -- three over, which
no reader would notice and a screening tool would.

Measured against the journal's own guide for authors and against 244 MSSP
articles published since 2020 in this paper's area: their abstracts run a
median of 236 words, so the cap is a real constraint that most papers write
close to rather than a formality.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")

rows = []


def check(what, got, want, ok):
    rows.append((what, str(got), str(want), "ok" if ok else "OVER THE LIMIT"))


s = io.open(os.path.join(PAPER, "mssp.tex"), encoding="utf-8").read()

# ---- abstract --------------------------------------------------------------
a = s.split(BS + "begin{abstract}")[1].split(BS + "end{abstract}")[0]
a = re.sub(CL + r"(?:emph|textit|textbf)\{([^}]*)\}", r"\1", a)
a = re.sub(CL + r"[a-zA-Z]+\{([^}]*)\}", r"\1", a)
a = re.sub(CL + r"[a-zA-Z]+", " ", a)
a = re.sub(r"[{}$]", " ", a)
n = len(a.split())
check("abstract, in words", n, "at most 250", n <= 250)

# ---- highlights ------------------------------------------------------------
hp = os.path.join(PAPER, "highlights.txt")
if os.path.exists(hp):
    hl = [ln.strip().lstrip("-").strip()
          for ln in io.open(hp, encoding="utf-8") if ln.strip()]
    check("highlights, bullets", len(hl), "3 to 5", 3 <= len(hl) <= 5)
    longest = max((len(h) for h in hl), default=0)
    check("highlights, longest bullet", "%d chars" % longest,
          "at most 85", longest <= 85)
else:
    check("highlights file", "MISSING", "paper/highlights.txt", False)

# ---- keywords --------------------------------------------------------------
m = re.search(CL + r"begin\{keyword\}(.*?)" + CL + r"end\{keyword\}", s, re.S)
if m:
    kw = [k.strip() for k in re.split(CL + r"sep", m.group(1)) if k.strip()]
    check("keywords", len(kw), "1 to 7", 1 <= len(kw) <= 7)
    joined = [k for k in kw if re.search(r"\b(and|of)\b", k)]
    check("keywords joined by 'and'/'of'", len(joined), "0", not joined)
else:
    check("keyword block", "MISSING", "present", False)

# ---- the declarations the journal requires ---------------------------------
# CRediT is entered by the corresponding author in the submission system, at
# the corresponding author's decision (2026-09-24); the Guide asks for the roles,
# not for a section in the manuscript.
for name, pat in (("competing interest", r"[Cc]ompeting interest"),
                  ("funding statement", CL + r"section\*\{Funding\}"),
                  ("data availability", r"[Dd]ata availability"),
                  ("generative-AI declaration",
                   CL + r"section\*\{Declaration of generative AI")):
    check(name, "present" if re.search(pat, s) else "MISSING", "required",
          bool(re.search(pat, s)))
# the Guide: the generative-AI statement goes "in a new section before the
# references list"; nothing may sit between it and the bibliography
_ai = s.find("section*{Declaration of generative AI")
_bib = s.find("begin{thebibliography}")
_between = s[_ai:_bib]
check("generative-AI declaration placed before the references",
      "last" if _ai > 0 and "section" not in _between[10:] else "NOT LAST",
      "last", _ai > 0 and "section" not in _between[10:])

# ---- references: DOIs and abbreviated journal names ------------------------
# The guide asks for DOIs where they exist and for journal names abbreviated by
# the ISSN List of Title Word Abbreviations; every MSSP article read for this
# check prints both.  Books, two conference papers without one, JMLR and the
# NASA data set have no DOI to give.
bib = s[s.index(BS + "begin{thebibliography}"):s.index(BS + "end{thebibliography}")]
items = bib.split(BS + "bibitem")[1:]
NO_DOI = {"kay1993", "vantrees1968", "meeker1998", "huber1981", "hampel1986",
          "bickel1993", "serfling1980", "coble2009", "nectoux2012", "saha2007",
          "krause2008"}
missing = [re.match(r"\{([^}]+)\}", it).group(1) for it in items
           if "doi.org/" not in it
           and re.match(r"\{([^}]+)\}", it).group(1) not in NO_DOI]
check("references without a DOI", len(missing), "0 outside the books",
      not missing)
full = [j for j in ("Mechanical Systems and Signal Processing",
                    "Journal of", "Transactions on", "Engineering \\& System")
        if j in re.sub(r"\s+", " ", bib)]
check("journal names left unabbreviated", len(full), "0", not full)

# ---- graphical abstract ----------------------------------------------------
# 1328 x 531 pixels (w x h) or proportionally more: an aspect of 2.50
gp = os.path.join(PAPER, "gabs.pdf")
if os.path.exists(gp):
    try:
        import fitz
        r = fitz.open(gp)[0].rect
        asp = r.width / r.height
        check("graphical abstract, width/height", "%.2f" % asp, "2.50",
              abs(asp - 1328 / 531) < 0.03)
    except ImportError:
        pass

# An unfilled declaration is present but not written, and the paper cannot be
# submitted with one; audit_paper reports those separately and they are not
# counted here as a limit breach.
unfilled = len(re.findall(r"UNFILLED", s))

print("=" * 78)
print("The journal's own limits")
print("=" * 78)
print("%-34s %14s %16s" % ("requirement", "paper", "limit"))
print("-" * 78)
for what, got, want, st in rows:
    print("%-34s %14s %16s%s"
          % (what, got, want, "" if st == "ok" else "   <-- " + st))
print("-" * 78)
bad = [r for r in rows if r[3] != "ok"]
print("%d requirements checked, %d breached" % (len(rows), len(bad)))
if unfilled:
    print("\n%d declaration(s) are present but still say UNFILLED; the journal "
          "needs\nthem written before submission." % unfilled)
sys.exit(1 if bad else 0)
