r"""The twentieth audit: does a section land, or does it just stop?

A reader feels a paper as a sequence of arrivals.  A section that ends on its
last measurement, or worse on a pointer to where the rest of that measurement
lives, gives the reader nothing to carry into the next one, and enough of those
in a row is what makes a paper feel like information thrown rather than argued.
Six sections ended that way at once here: three closed on "Supplementary
Section~S$n$ gives \dots", one on a cross-reference, and two on a parameter.

WHAT IS CHECKED.  For every numbered section, the last sentence of its prose
must be a CLAIM.  Two things disqualify it:

  (POINTER)  its main assertion is that some other document, section, table or
             figure contains something --- it ends by delegating;
  (STUB)     it is too short to carry a claim at all.

Floats are removed first: a caption is not the section's last word.  Proof
environments are removed too, because a section that ends on a proof ends on the
proof's last line by construction and that is a convention, not a failure.

This layer cannot tell a good landing from a dull one; no layer can.  What it
can tell is that the author did not simply stop, and that is the failure that
actually occurred.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"
HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")

MINWORDS = 8
POINTER = re.compile(
    r"^(supplementary section|section|table|figure|appendix|proposition)\b"
    r"[^.]*\b(gives?|holds?|reports?|lists?|collects?|contains?|carries|"
    r"sets? out|records?|identifies)\b", re.I)

s = re.sub("(?m)^%.*$", "", io.open(TEX, encoding="utf-8").read())
cut = s.find(BS + "begin{thebibliography}")
if cut > 0:
    s = s[:cut]
for env in ("figure", "table", "tikzpicture", "proof"):
    s = re.sub(CL + r"begin\{" + env + r"\*?\}.*?" + CL + r"end\{" + env
               + r"\*?\}", " ", s, flags=re.S)

heads = [(m.start(), bool(m.group(1)), m.group(2))
         for m in re.finditer(CL + "(sub)?section" + r"\{([^}]*)\}", s)]


def last_sentence(t):
    t = re.sub(CL + r"(label|ref|eqref|cite)\{?[^}]*\}?", " ", t)
    t = re.sub(CL + "[a-zA-Z]+", " ", t)
    t = re.sub(r"[{}$~]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t)
             if len(p.strip()) > 12]
    return parts[-1] if parts else ""


bad = []
print("=" * 92)
print("Does each section land, or does it just stop?")
print("=" * 92)
print()
for i, (a, sub, title) in enumerate(heads):
    if sub:
        continue
    end = len(s)
    for b, sb, _ in heads[i + 1:]:
        if not sb:
            end = b
            break
    ls = last_sentence(s[a:end])
    why = ""
    if len(ls.split()) < MINWORDS:
        why = "STUB"
    elif POINTER.match(ls):
        why = "POINTER"
    if why:
        bad.append((title, why))
    print("  %-44s %s" % (title[:44], why or "lands"))
    if why:
        print("      %s" % ls[:150])

print()
print("=" * 92)
print("every section ends on a claim" if not bad
      else "%d section(s) end by delegating or trailing off" % len(bad))
print("=" * 92)
sys.exit(0 if not bad else 1)
