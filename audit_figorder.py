r"""The twenty-second audit: does a float ever arrive before its citation?

Every figure in this manuscript once sat in the source ahead of the sentence
that first cites it -- ten of eleven -- because a figure written at the head of
the section it belongs to reads naturally in the source and typesets backwards.
With `[t]` the float goes to the top of the page, which is frequently above the
paragraph that introduces it, so the reader meets a picture with no idea yet
what it is for.

That pass fixed the figures and this layer then locked them, but it read only
\begin{figure} -- and so for a further round of revision nobody was checking the
tables.  Seven of the nine in the manuscript still sat ahead of their citation,
tab:crb printed a page before the sentence that sends the reader to it, and four
of the five tables in the supplement were never cited at all: each sat directly
under a subsection heading, which looks like an introduction and is not one.
The layer now reads both kinds of float in both documents.

WHAT IS CHECKED.  For every figure and table, the environment must begin AFTER
the first reference to its label.  That is the condition the author controls;
where the float finally lands is LaTeX's decision, but with the environment
placed after the citation and `h` among the allowed positions, the normal
outcome is the right one.  _placement.py checks the outcome itself, reading the
typeset PDF.

A float nobody cites fails too: it is the same defect at its limit.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")

bad = []
for stem in ("mssp", "supplement"):
    s = io.open(os.path.join(PAPER, stem + ".tex"), encoding="utf-8").read()
    lineno = lambda i: s[:i].count("\n") + 1
    rows = []
    for kind in ("fig", "tab"):
        env = "figure" if kind == "fig" else "table"
        for m in re.finditer(CL + r"label\{(" + kind + r":[^}]*)\}", s):
            lab = m.group(1)
            a = s.rfind(BS + "begin{" + env + "}", 0, m.start())
            b = s.find(BS + "end{" + env + "}", m.start())
            if a < 0 or b < 0:
                continue        # a label of that prefix outside a float
            cite = None
            for r in re.finditer(CL + r"ref\{" + re.escape(lab) + r"\}", s):
                cite = r.start()
                break
            rows.append((a, lab, lineno(a), lineno(b),
                         lineno(cite) if cite is not None else None,
                         cite is not None and cite < a))

    print("=" * 92)
    print("Does a float ever arrive before the text that cites it?   %s.tex"
          % stem)
    print("=" * 92)
    print("\n  %-20s %8s %8s   %s" % ("float", "begins", "cited", "order"))
    print("  " + "-" * 62)
    for a, lab, ab, bb, c, ok in sorted(rows):
        if c is None:
            v = "NEVER CITED"
            bad.append(stem + ":" + lab)
        elif ok:
            v = "cited %d lines earlier" % (ab - c)
        else:
            v = "ARRIVES %d LINES EARLY" % (c - bb)
            bad.append(stem + ":" + lab)
        print("  %-20s %8d %8s   %s" % (lab, ab, c if c else "-", v))
    print()

print("=" * 92)
print("every float follows the text that cites it" if not bad
      else "%d float(s) arrive before their citation: %s"
           % (len(bad), ", ".join(bad)))
print("=" * 92)
sys.exit(0 if not bad else 1)
