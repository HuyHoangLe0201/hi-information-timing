r"""The tenth audit: does the road map describe the paper that was written?

A long paper promises its structure once, in a paragraph of the introduction,
and then nobody checks the promise again.  This one had drifted.  The road map
asked its six questions in the order

    sensitivity, precision, selection, identifiability, additive, bayes

while the sections stood in the order

    sensitivity, precision, additive, selection, identifiability, bayes,

because two of them had been exchanged to repair a forward reference and the
paragraph that announces them was not exchanged with them.  Nothing could
report that.  Every reference resolved, every number was right, the arrangement
audit saw an admissible order, and the order audit saw no forward dependency.
The defect is not in the paper's content or in its order but in the agreement
between the two, and agreement is a property of a pair.

WHAT IS CHECKED.  Let the numbered sections be s_1 .. s_n in document order, and
let the road map be the closing paragraphs of the introduction.  Two conditions:

  (COVER)  every numbered section after the introduction is named there;
  (ORDER)  the sequence of first mentions is non-decreasing in document
           position.

A span "Sections A--B" names both endpoints without claiming to discuss B next,
so a span is recorded as coverage and excluded from the order sequence; B must
still be mentioned in its own place for ORDER to see it.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
s = io.open(TEX, encoding="utf-8").read()
s = re.sub("(?m)^%.*$", "", s)
# The appendix is not a stage of the argument and the road map does not
# announce it, so the body ends where "\appendix begins.
_stop = [i for i in (s.find(BS + "appendix"),
                     s.find(BS + "begin{thebibliography}")) if i > 0]
body = s[:min(_stop)]

# --- the numbered sections, in document order --------------------------------
secs = []
for m in re.finditer(CL + "section{([^}]*)}", body):
    lab = re.search(CL + "label{(sec:[^}]*)}", body[m.end():m.end() + 200])
    secs.append((lab.group(1) if lab else None, m.group(1), m.start()))
order = [x[0] for x in secs]
title = {a: b for a, b, _ in secs}
pos = {k: i for i, k in enumerate(order) if k}

# --- the road map: the closing paragraphs of the introduction ----------------
# the paragraph was renamed when the introduction was rewritten; it is found
# by either name so an old draft still audits
# The paragraph has been renamed twice: once when the introduction was
# rewritten, and once when the road map was merged with the statement of
# contributions, so that the sections are named where their evidence is
# promised rather than in a second walk through the same list.
# The trailing period was removed from every paragraph title when elsarticle
# was found to add one of its own, so both spellings are accepted.
for _name in ("Contributions and organisation", "Organisation",
              "Notation and organisation"):
    start = body.find(BS + "paragraph{" + _name)
    if start >= 0:
        break
end = body.find(BS + "section{", body.find(BS + "section{", 1) - 1)
end = secs[1][2]                      # the second numbered section
assert start > 0, "the road map paragraph was not found"
road = body[start:end]

# A span names everything it spans, not only its two ends.  "Sections 5 to 8
# estimate the curve, correct it, test it and fix its scope" does tell a reader
# what 6 and 7 are for, and an audit that demanded each be named separately
# would push the road map back towards the section-by-section walk it was
# shortened out of.  Both the en-dash and the word "to" are recognised.
_SPAN = CL + "ref{(sec:[^}]*)}(?:--|\\s*to~?\\s*)" + CL + "ref{(sec:[^}]*)}"
spans = re.findall(_SPAN, road)
named = set()
for a, b in spans:
    if a in pos and b in pos:
        lo, hi = sorted((pos[a], pos[b]))
        named.update(k for k in order[lo:hi + 1] if k)
    else:
        named.add(a)
        named.add(b)
seq = []
for m in re.finditer(CL + "ref{(sec:[^}]*)}", re.sub(_SPAN, " ", road)):
    k = m.group(1)
    named.add(k)
    if k in pos and k not in seq:
        seq.append(k)

# --- COVER -------------------------------------------------------------------
# Three sections are exempt, and the reason is stated rather than assumed.  The
# introduction does not announce itself.  The discussion and the conclusion are
# not stages of the argument -- the road map says the paper has four stages, and
# they are what follows the fourth -- so a reader who is not told a paper ends
# with them has lost nothing.  Every section that carries a stage must be named.
EXEMPT = {order[0], "sec:discussion", "sec:conclusion"}
missing = [k for k in order if k and k not in named and k not in EXEMPT]

# --- ORDER -------------------------------------------------------------------
inversions = [(seq[i], seq[j])
              for i in range(len(seq)) for j in range(i + 1, len(seq))
              if pos[seq[i]] > pos[seq[j]]]

# --- report ------------------------------------------------------------------
print("=" * 92)
print("Does the road map describe the paper that was written?")
print("=" * 92)
print("\n  %d numbered sections; the road map names %d of them\n"
      % (len(order), len([k for k in order if k in named])))
print("  %-4s %-34s %s" % ("", "document order", "road map"))
print("  " + "-" * 80)
for i, k in enumerate(order):
    where = ("%d" % (seq.index(k) + 1)) if k in seq else (
        "named in a span" if k in named
        else ("exempt" if k in EXEMPT else "NOT NAMED"))
    print("  %-4d %-34s %s" % (i + 1, title[k][:34], where))

print("\n  COVER: ", end="")
print("every section is named" if not missing
      else "NOT NAMED: " + ", ".join(missing))
print("  ORDER: ", end="")
if not inversions:
    print("the road map's %d mentions follow document order" % len(seq))
else:
    print("%d inversion(s)" % len(inversions))
    for a, b in inversions:
        print("     the map names %s (section %d) before %s (section %d)"
              % (a, pos[a] + 1, b, pos[b] + 1))

print()
print("=" * 92)
ok = not missing and not inversions
print("the road map agrees with the paper" if ok
      else "the road map disagrees with the paper")
print("=" * 92)
sys.exit(0 if ok else 1)
