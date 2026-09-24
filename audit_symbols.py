r"""The eighteenth audit: does a symbol mean one thing?

A reading of the finished manuscript turned up four symbols carrying two
meanings each, and none of the seventeen existing layers could have seen any of
them, because every one of them checks numbers, structure or references and a
symbol collision is none of those.  What was found:

  \varepsilon  the observation noise in the model, AND the precision demanded
               of the clock, two lines apart in the notation table;
  \sigma       the local noise scale everywhere, AND the standard error of an
               age estimate in the replacement proposition;
  c            the censoring age in three propositions, AND the information
               demand in the two beside them;
  D            the degradation trend throughout, AND the matrix of sensitivity
               rows in the leverage proposition.

WHAT IS CHECKED.  The notation table is the manuscript's own declaration of what
each symbol means.  This layer collects those symbols and then reports every
place in the text that reads like a SECOND declaration of one of them --- "let
$X$ be", "write $X$ for", "with $X$ the", "$X$ denotes" --- because a symbol
that the table has already defined should not be defined again.

It cannot decide whether a redeclaration is a collision or a harmless restatement
in the same sense; that is a judgement.  So it prints them and holds the count
against a ceiling, as the orphan and opening layers do.  Raise the ceiling only
after reading the list: a rise means a symbol has been given a second meaning
somewhere, which is exactly the thing the layer exists to surface.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
CEILING = os.path.join(HERE, "symbols_ceiling.txt")

sys.path.insert(0, HERE)
import _source  # noqa: E402

s = re.sub("(?m)^%.*$", "", _source.read_both())
flat = re.sub(r"\s+", " ", s)

# --- the symbols the notation table declares ---------------------------------
tab = re.search(CL + "label{tab:notation}(.*?)" + CL + "end{tabular}", flat)
assert tab, "the notation table was not found"
symbols = []
for row in tab.group(1).split(BS + BS):
    cell = row.split("&")[0].strip()
    for m in re.finditer(r"\$([^$]+)\$", cell):
        t = m.group(1).strip()
        # the table lists variants in one cell ("$D(\tau)$, $D'$"); each counts
        if t and t not in symbols:
            symbols.append(t)

# The bare letter is what a second declaration would use, so a symbol written
# with an argument or a subscript is reduced to its head.
def head(t):
    t = re.sub(r"[_^].*$", "", t)
    t = re.sub(r"\(.*$", "", t)
    return t.strip()


heads = []
for t in symbols:
    h = head(t)
    if h and h not in heads:
        heads.append(h)

# --- second declarations -----------------------------------------------------
VERBS = [r"[Ll]et", r"[Ww]rite", r"[Ww]ith", r"[Dd]enote", r"and let"]
hits = []
for h in heads:
    pat = (r"(?:" + "|".join(VERBS) + r")\s+" + re.escape("$" + h)
           + r"(?:\^[^$]*|_[^$]*|\([^$]*\))?\$\s+(?:be|for|the|denotes?)")
    for m in re.finditer(pat, flat):
        a = max(0, m.start() - 40)
        hits.append((h, flat[a:m.end() + 60]))

ceiling = 0
if os.path.exists(CEILING):
    ceiling = int(io.open(CEILING).read().split()[0])

print("=" * 92)
print("Does a symbol mean one thing?")
print("=" * 92)
print("\n  the notation table declares %d symbols, %d distinct heads"
      % (len(symbols), len(heads)))
print("  %d of them are declared a second time in the text\n" % len(hits))
for h, ctx in hits:
    print("  %-12s ...%s" % ("$" + h + "$", ctx.strip()[:88]))
print("\n  ceiling %d, now %d" % (ceiling, len(hits)))
print("=" * 92)
ok = len(hits) <= ceiling
print("no symbol is declared more often than the ceiling allows" if ok
      else "A SYMBOL IS DECLARED AGAIN: read the list above before raising it")
print("=" * 92)
sys.exit(0 if ok else 1)
