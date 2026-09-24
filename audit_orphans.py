r"""The twelfth audit: is there any stored result that could have produced this?

WHY.  The paragraph comparing three alarm calibrations quoted six figures, and
four of them existed in no result file at all -- they came from runs nobody
kept.  That was found by accident, while looking at something else.  A defect
class found by accident once will be there again, so it needs a layer.

WHAT THIS ASKS.  Not whether a number is right, which is what the numeric audit
does for the numbers it binds, but the weaker and wider question: does the
repository contain ANY stored value that rounds to this printed one?  A number
with no such value anywhere cannot have been checked by anything, and cannot be
checked by anything until the run that produced it is restored.

  MATCH   a printed number with d decimals matches a stored leaf v when
          |v - x| <= 0.5 * 10^-d, that is, when v would print as x.
  ALSO    list lengths and dictionary sizes count as stored values, since a
          count in the text is usually the size of something rather than a
          number written into a file.

WHAT AN ORPHAN IS AND IS NOT.  An orphan is a candidate, not a verdict.  Most
are innocent: a number formed in the sentence itself (a ratio of two stored
ones, a total, a percentage), a quantity spelled from a table cell that is
stored under a different rounding, or a bound stated by the author rather than
measured.  The output is therefore a worklist ordered so that the dangerous
cases surface first: an orphan with a decimal point, in a sentence that also
names a rig or a dataset, is far more likely to be a lost measurement than a
bare integer in a definition.

The count is reported and the list is written out; the audit fails only if the
number of orphans exceeds a recorded ceiling, so that the figure can only be
driven down and never quietly up.
"""
import glob
import io
import json
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
CEILING = os.path.join(HERE, "orphans_ceiling.txt")

# Files that store the manuscript's own numbers would make every match trivial.
SELF = ("audit_coverage.json", "audit_unused.json")


def leaves(o):
    if isinstance(o, dict):
        yield float(len(o))
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        yield float(len(o))
        for v in o:
            yield from leaves(v)
    elif isinstance(o, bool):
        return
    elif isinstance(o, (int, float)):
        yield float(o)


vals = []
nfiles = 0
for f in sorted(glob.glob(os.path.join(HERE, "*.json"))):
    if os.path.basename(f) in SELF:
        continue
    try:
        d = json.load(io.open(f, encoding="utf-8"))
    except Exception:
        continue
    nfiles += 1
    vals.extend(v for v in leaves(d) if abs(v) < 1e12)
vals = sorted(set(vals))

# a sorted array plus bisection, because there are of the order of a hundred
# thousand stored values and several hundred printed ones
import bisect
# The audited surface is the manuscript AND the supplement: material
# moved to the supplement has not been deleted, and a check that read
# only mssp.tex could not tell the difference.  See _source.py.
import _source


def stored(x, dec):
    tol = 0.5 * 10 ** (-dec) + 1e-12
    i = bisect.bisect_left(vals, x - tol)
    return i < len(vals) and vals[i] <= x + tol


s = _source.read_both()
s = re.sub("(?m)^%.*$", "", s)
s = re.sub(CL + r"orcidlink{[^}]*}", " ", s)
s = re.sub(CL + r"(cite|ref|eqref|label|ead|address|affiliation)"
           r"(\[[^\]]*\])?{[^}]*}", " ", s)
# A thousands separator is typeset as a thin space, so "102\,464" is one number
# and not two.  Removing the separator before scanning was worth doing on the
# first run alone: nine of the sixteen unaccounted numbers were halves of one.
s = re.sub(r"(\d)" + CL + r",(\d)", r"\1\2", s)
body = s[s.find(BS + "begin{abstract}"):s.find(BS + "begin{thebibliography}")]
# Two regions are excluded.  The proofs appendix states algebra rather than
# measurements, and the verification note describes this apparatus rather than
# the records: its numbers count checks, and writing them into the paper changes
# them, so scanning them would make the audit chase its own tail.  Both are cut
# for those reasons and not because they are inconvenient.
#
# They are cut as SPANS and not by truncation.  Truncating at whichever came
# first was correct only while everything worth scanning came before them, and
# it stopped being correct the moment the robustness theory moved into
# appendices that follow the note: a prefix cut would have dropped seventeen
# pages of measurements out of the scan without changing a single line of
# output.
_drop = []
_v = body.find(BS + "section*{Verification}")
if _v > 0:
    _e = body.find(BS + "appendix", _v)
    _drop.append((_v, _e if _e > 0 else len(body)))
_p = body.find(BS + "section{Omitted proofs}")
if _p > 0:
    _drop.append((_p, len(body)))


def _cut(i):
    return any(a <= i < b for a, b in _drop)

RIGS = ("PRONOSTIA", "XJTU", "FD001", "FD004", "bearing", "turbofan", "cell",
        "record", "unit", "rig", "fleet")
orphans = []
seen = 0
for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w]|\.\d)", body):
    t = m.group(1)
    if len(t.replace(".", "")) < 2 or re.match(r"^(19|20)\d\d$", t):
        continue
    if _cut(m.start()):
        continue
    seen += 1
    dec = len(t.split(".")[1]) if "." in t else 0
    # A printed number is not always the number that was stored.  Two
    # transformations are routine in this manuscript and both were producing
    # false orphans on the first run: a mantissa written against a power of ten,
    # and a percentage whose result file holds the fraction.  The candidate
    # values are tried in order and any match accounts for the number.
    tail = body[m.end():m.end() + 26]
    cands = [(float(t), dec)]
    _p = re.match(r"\s*(?:\$\s*)?" + CL + r"times\s*10\^[{]?(-?\d+)[}]?", tail)
    if _p:
        e = int(_p.group(1))
        cands.append((float(t) * 10.0 ** e, dec - e))
    _t = tail.lstrip()
    if _t.startswith("%") or _t.startswith(BS + "%") or _t.startswith("per cent"):
        cands.append((float(t) / 100.0, dec + 2))
    if any(stored(v, d) for v, d in cands):
        continue
    ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 110):m.end() + 55]).strip()
    orphans.append(dict(value=t, decimals=dec, at=m.start(), context=ctx,
                        near_rig=any(w in ctx for w in RIGS)))

# Dangerous first: a decimal number in a sentence naming a rig is the shape of a
# lost measurement; a bare integer in a definition is the shape of a constant.
orphans.sort(key=lambda o: (-(o["decimals"] > 0), -o["near_rig"],
                            -o["decimals"], o["at"]))

print("=" * 92)
print("Numbers the repository cannot account for")
print("=" * 92)
print("""
This does not ask whether a number is right.  It asks whether any stored result
contains a value that would print as it.  A number with none cannot have been
checked by anything.
""")
print("  result files searched         : %d" % nfiles)
print("  distinct stored values        : %d" % len(vals))
print("  numbers printed in the paper   : %d" % seen)
print("  of those, with no stored match : %d  (%.0f per cent)"
      % (len(orphans), 100.0 * len(orphans) / max(seen, 1)))

show = [o for o in orphans if o["decimals"] > 0][:25]
if show:
    print("\n  the ones shaped like a lost measurement, worst first:\n")
    for o in show:
        print("   %-8s %s" % (o["value"], o["context"][-96:]))
rest = len(orphans) - len(show)
if rest > 0:
    print("\n  and %d more, mostly bare integers; the full list is in "
          "orphan_numbers.json" % rest)

json.dump(dict(searched=nfiles, stored=len(vals), printed=seen,
               orphans=orphans),
          io.open(os.path.join(HERE, "orphan_numbers.json"), "w",
                  encoding="utf-8"), indent=1)

# --- the ceiling, which may fall and may not rise ----------------------------
prev = None
if os.path.exists(CEILING):
    try:
        prev = int(io.open(CEILING, encoding="utf-8").read().split()[0])
    except Exception:
        prev = None
now = len(orphans)
if prev is None:
    io.open(CEILING, "w", encoding="utf-8").write(
        "%d\n\nThe number of printed values with no stored match, at the last\n"
        "run that was accepted.  It may be lowered by accounting for a number\n"
        "and may not be raised: a rise means a sentence now quotes something\n"
        "the repository cannot produce.\n" % now)
    print("\n  ceiling recorded at %d" % now)
    ok = True
else:
    print("\n  ceiling %d, now %d" % (prev, now))
    ok = now <= prev
    if ok and now < prev:
        io.open(CEILING, "w", encoding="utf-8").write(
            "%d\n\nThe number of printed values with no stored match, at the\n"
            "last run that was accepted.  It may be lowered by accounting for\n"
            "a number and may not be raised: a rise means a sentence now quotes\n"
            "something the repository cannot produce.\n" % now)
        print("  lowered to %d" % now)

print()
print("=" * 92)
print("every printed number has a stored value that could have produced it"
      if now == 0 else
      ("the unaccounted numbers are at or below the ceiling" if ok else
       "MORE numbers are unaccounted for than at the last accepted run"))
print("=" * 92)
sys.exit(0 if ok else 1)
