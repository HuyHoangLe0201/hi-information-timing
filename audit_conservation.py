r"""The fifteenth audit: did the split move text, or lose it?

Building a supplement is a long sequence of cut-and-paste, and a cut that does
not paste leaves nothing behind to notice.  One did: a mover that wrote the
source before validating the destination deleted eight passages and stored
none, and the only reason it was caught is that the compiler happened to report
an undefined reference from an unrelated repair.  Nothing in fourteen audit
layers could have reported it, because every one of them reads the two files as
they are and asks whether they are consistent; none of them remembers what was
there before.

WHAT IS CHECKED.  Let B be the manuscript as it stood before the split, saved
in .backup/mssp_before_supplement.tex, and let S be the union of the current
mssp.tex and supplement.tex.  Every line of prose in B must appear in S.

A line that does not is not necessarily lost: the split also rewrites, and a
paragraph replaced by a two-sentence summary leaves its old lines behind by
design.  So the count of absent lines is held against a ceiling in
conservation_ceiling.txt, exactly as the orphan and opening audits are, and the
lines themselves are printed so that raising the ceiling is a decision taken
against a list rather than a number.  Raise it when the lines shown are ones
you rewrote; investigate when they are ones you meant to move.

Lines are compared after collapsing whitespace, so rewrapping a moved paragraph
does not read as a loss.  Short lines carry too little to identify a passage and
are skipped.
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
BASE = os.path.join(HERE, ".backup", "mssp_before_supplement.tex")
CEILING = os.path.join(HERE, "conservation_ceiling.txt")

MINLEN = 40


def lines(path):
    s = io.open(path, encoding="utf-8").read()
    s = re.sub("(?m)^%.*$", "", s)
    return [re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n")]


base = [ln for ln in lines(BASE) if len(ln) >= MINLEN]
have = set(lines(os.path.join(PAPER, "mssp.tex")))
have |= set(lines(os.path.join(PAPER, "supplement.tex")))

gone = [ln for ln in base if ln not in have]

ceiling = 0
if os.path.exists(CEILING):
    ceiling = int(io.open(CEILING).read().split()[0])

print("=" * 92)
print("Did the split move the text, or lose it?")
print("=" * 92)
print("\n  %d lines of prose before the split; %d of them are not in either"
      " file now\n" % (len(base), len(gone)))
for ln in gone:
    print("    " + ln[:86])
print("\n  ceiling %d, deliberately rewritten out" % ceiling)
print("=" * 92)
ok = len(gone) <= ceiling
print("no text was lost that was not rewritten" if ok
      else "TEXT IS MISSING: %d lines above the ceiling" % (len(gone) - ceiling))
print("=" * 92)
sys.exit(0 if ok else 1)
