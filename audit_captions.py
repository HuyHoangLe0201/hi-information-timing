r"""The twenty-third audit: is a caption still a caption?

Captions drift.  Each round of revision answers a reader's objection where the
objection was raised, and the place a reader raises an objection about a figure
is the figure, so method, caveat and cross-reference accumulate under the
picture until the caption is a paragraph.  This set reached a mean of 77 words
and a longest of 140 -- about two pages of the manuscript sitting in small type
under floats -- before anyone measured it.

TWO THINGS ARE CHECKED.

Length.  A caption identifies what is shown and, for a panelled figure, says
what each panel is.  That fits in about forty words and never needs seventy.
The ceiling is held in caption_ceiling.txt so that raising it is a decision
rather than an accident.

Orphaned numbers.  A number whose only occurrence in either document is inside
a caption is a number the reader cannot find again, and -- worse for this
suite -- a number some check in audit_paper.py may be reading there.  Trimming
such a caption silently removes the check's anchor.  Every number printed in a
caption must therefore also be printed outside one.  When the trim of this
session dropped thirteen captions from 1460 words to 788, this is the rule that
said which sentences had to move into the body first.

Numbers that identify the float's own contents -- a count of panels, a band
edge -- are exempt through caption_numbers_allowed.txt, one per line.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
CEILING = os.path.join(HERE, "caption_ceiling.txt")
ALLOWED = os.path.join(HERE, "caption_numbers_allowed.txt")


def captions(path):
    """Every caption in a file, with the label that follows it."""
    s = io.open(path, encoding="utf-8").read()
    out, spans = [], []
    for m in re.finditer(CL + r"caption\{", s):
        i, d, j = m.end(), 1, m.end()
        while j < len(s) and d:
            d += (s[j] == "{") - (s[j] == "}")
            j += 1
        lab = re.search(CL + r"label\{([^}]*)\}", s[j:j + 400])
        out.append((lab.group(1) if lab else "?", s[i:j - 1]))
        spans.append((m.start(), j))
    # the same file with every caption removed, which is where a number has to
    # appear a second time
    body, prev = "", 0
    for a, b in spans:
        body += s[prev:a]
        prev = b
    return out, body + s[prev:]


# Control sequences are markup, not words; the count is of what is read aloud.
words = lambda t: len(re.sub(CL + "[a-zA-Z]+", " ", t).split())
# A number as the reader meets it, with the thin-space separator and the math
# delimiters stripped so that $102\,464$ and 102464 are one token.
flat = lambda t: re.sub(r"\s+", " ", t.replace(BS + ",", "").replace("$", ""))
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w]|\.\d)")

ceiling = 75
if os.path.exists(CEILING):
    ceiling = int(io.open(CEILING).read().split()[0])
allowed = set()
if os.path.exists(ALLOWED):
    allowed = {ln.split("#")[0].strip()
               for ln in io.open(ALLOWED, encoding="utf-8")
               if ln.split("#")[0].strip()}

caps, bodies = [], []
for name in ("mssp.tex", "supplement.tex"):
    c, b = captions(os.path.join(PAPER, name))
    caps += [(name, lab, txt) for lab, txt in c]
    bodies.append(b)
elsewhere = flat(" ".join(bodies))

print("=" * 92)
print("Is a caption still a caption?")
print("=" * 92)
print("\n%-16s %-16s %5s  %s" % ("file", "label", "words", "opens with"))
print("-" * 92)
long_ones, orphans = [], []
for name, lab, txt in caps:
    w = words(txt)
    mark = ""
    if w > ceiling:
        long_ones.append((lab, w))
        mark = "  <-- over the ceiling"
    print("%-16s %-16s %5d  %s%s"
          % (name, lab, w, flat(txt)[:44], mark))
    for n in NUM.findall(flat(txt)):
        if n in allowed:
            continue
        if not re.search(r"(?<![\w.])" + re.escape(n) + r"(?![\w]|\.\d)",
                         elsewhere):
            orphans.append((lab, n))

tot = sum(words(t) for _, _, t in caps)
print("\n  %d captions, %d words, mean %.0f, longest %d; the ceiling is %d."
      % (len(caps), tot, tot / len(caps),
         max(words(t) for _, _, t in caps), ceiling))

if long_ones:
    print("\n  OVER THE CEILING")
    for lab, w in long_ones:
        print("    %-18s %d words" % (lab, w))
if orphans:
    print("\n  PRINTED IN A CAPTION AND NOWHERE ELSE")
    print("    A reader meets these once, in small type under a float, and a")
    print("    check that reads one there loses its anchor the moment the")
    print("    caption is trimmed.  Move the sentence into the body, or add")
    print("    the number to caption_numbers_allowed.txt with a reason.")
    for lab, n in orphans:
        print("    %-18s %s" % (lab, n))

bad = len(long_ones) + len(orphans)
print("\n" + "=" * 92)
print("CAPTIONS ARE CAPTIONS" if not bad
      else "%d CAPTIONS TO FIX" % bad)
print("=" * 92)
sys.exit(1 if bad else 0)
