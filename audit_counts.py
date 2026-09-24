r"""The fourteenth audit: does a list have as many items as it announces?

A sentence that opens "Four alternative explanations were tested" is a promise
about the paragraph beneath it, and the paragraph is the only thing that can
keep it.  Nothing in this apparatus was checking that pairing, and it had come
apart: the subsection on alternative explanations announced four and listed five,
because a fifth control was added later and the opening sentence was not.  The
count is not decoration -- a reader who takes the four is left believing the
membership of the two families was never controlled for, which is the one the
fifth supplies.

WHAT IS CHECKED.  Wherever a sentence announces a count in words immediately
before a run of emphasised item labels, the announced number is compared against
the number of labels.  The pairing is recognised only when the announcement and
the items sit in the same subsection, so a count belonging to something else
cannot be captured by accident.

WHAT IS NOT CHECKED, and why.  Counts announced over paragraphs rather than
emphasised labels, and counts of things that are not lists at all, are outside
this: the announcement would have to be matched to a structure the file does not
mark, and a check that guesses is worse than no check.  What is here is the case
that failed, made mechanical.
"""
import io
import os
import re
import sys
# The audited surface is the manuscript AND the supplement: material
# moved to the supplement has not been deleted, and a check that read
# only mssp.tex could not tell the difference.  See _source.py.
import _source

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")

WORD = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}

s = re.sub("(?m)^%.*$", "", _source.read_both())

# the units a count could be announced over
heads = [(m.start(), m.group(1)) for m in
         re.finditer(CL + r"(?:sub)*section[*]?\{([^}]*)\}", s)]
heads.append((len(s), ""))

rows = []
for i in range(len(heads) - 1):
    a, title = heads[i]
    b = heads[i + 1][0]
    seg = s[a:b]
    items = re.findall(CL + r"emph\{([A-Z][^}]*\.)\}", seg)
    if len(items) < 3:
        continue                      # not a labelled list
    m = re.search(r"(?i)\b(" + "|".join(WORD) + r")\s+[a-z-]+(?:\s+[a-z-]+)?"
                  r"\s+(?:were|are|was|is)\b", seg)
    if not m:
        continue
    said = WORD[m.group(1).lower()]
    rows.append((title, said, len(items), re.sub(r"\s+", " ", m.group(0)),
                 items))

print("=" * 92)
print("Lists that announce how many items they have")
print("=" * 92)
print()
bad = 0
for title, said, got, phrase, items in rows:
    ok = said == got
    bad += 0 if ok else 1
    print("  %-44s says %-2d has %-2d %s" % (title[:44], said, got,
                                             "" if ok else "  <-- MISMATCH"))
    print("      \"%s\"" % phrase)
    print("      %s" % ", ".join(x.rstrip(".") for x in items))
    print()

print("=" * 92)
print("  %d announced lists, %d disagreeing with themselves" % (len(rows), bad))
print("every list has as many items as it says" if not bad
      else "a list does not have as many items as it says")
print("=" * 92)
sys.exit(0 if not bad else 1)
