r"""The twenty-first audit: does anything arrive before what it carries?

The order audit counts forward references and fails only when one sits in a
section's opening sentence.  That is too coarse for the defect a reader actually
feels, which is meeting a thing before the paper has supplied it.  Two kinds are
checkable against ground truth the manuscript already states.

  (SYMBOL)   The notation table names, for every symbol, the section that
             introduces it.  A symbol whose first appearance in the prose comes
             before that section arrives early, and the table is then a promise
             the text has already broken.

  (MEASURED) A section that measures something on a set of objects defined in a
             later section asks the reader to accept a measurement over things
             they have not met.  These are found by looking for a reference to a
             later section inside a sentence whose verb is one of measurement.

Neither is automatically a fault.  A paper may legitimately promise, and the
notation table exists precisely so a reader can look a symbol up.  What the
layer does is hold the count against a ceiling, so that the number of times the
reader is asked to wait cannot grow without someone deciding it should.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"
HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
CEILING = os.path.join(HERE, "arrival_ceiling.txt")

s = re.sub("(?m)^%.*$", "", io.open(TEX, encoding="utf-8").read())
cut = s.find(BS + "begin{thebibliography}")
if cut > 0:
    s = s[:cut]

# --- where each section begins ------------------------------------------------
sec_at, order = {}, []
for m in re.finditer(CL + "(sub)?section" + r"\{([^}]*)\}", s):
    lab = re.search(CL + r"label\{(sec:[^}]*)\}", s[m.end():m.end() + 160])
    if lab:
        sec_at[lab.group(1)] = m.start()
        order.append(lab.group(1))

# --- the notation table: symbol -> the section it says introduces it ----------
tab = re.search(CL + r"label\{tab:notation\}(.*?)" + CL + r"end\{tabular\}",
                s, re.S)
assert tab, "the notation table was not found"
table_end = tab.end()
promises = []
for row in tab.group(1).split(BS + BS):
    cells = row.split("&")
    if len(cells) < 3:
        continue
    lab = re.search(CL + r"ref\{(sec:[^}]*)\}", cells[-1])
    if not lab or lab.group(1) not in sec_at:
        continue
    for m in re.finditer(r"\$([^$]+)\$", cells[0]):
        t = m.group(1).strip()
        if len(t) > 1 and t not in [p[0] for p in promises]:
            promises.append((t, lab.group(1)))

early = []
for sym, sec in promises:
    where = sec_at[sec]
    hit = None
    for m in re.finditer(re.escape("$" + sym + "$"), s[table_end:]):
        hit = table_end + m.start()
        break
    if hit is not None and hit < where:
        early.append((sym, sec, hit, where))

# --- a measurement quoted over objects defined later --------------------------
# Floats are scanned out first: a caption may legitimately name a later section,
# and the notation table names one for every symbol by design.  Only the running
# prose is asked whether it measures something over objects not yet defined.
prose = s
for env in ("figure", "table", "tikzpicture"):
    prose = re.sub(CL + r"begin\{" + env + r"\*?\}.*?" + CL + r"end\{" + env
                   + r"\*?\}", lambda m: " " * (m.end() - m.start()),
                   prose, flags=re.S)

# The road map is excluded.  Its whole function is to point forward, so every
# section it names is by construction defined later; counting those would bury
# the thing this layer exists to find, which is a MEASUREMENT quoted over
# objects the reader has not met.
_rm = prose.find(BS + "paragraph{Contributions and organisation")
if _rm > 0:
    _rmend = prose.find(BS + "section{", _rm)
    prose = prose[:_rm] + " " * (_rmend - _rm) + prose[_rmend:]

VERB = r"(measured|computed|read|quoted|taken|evaluated|built)"
mearly = []
for m in re.finditer(CL + r"ref\{(sec:[^}]*)\}", prose):
    tgt = m.group(1)
    if tgt not in sec_at or sec_at[tgt] <= m.start():
        continue
    a = prose.rfind(".", 0, m.start())
    sent = prose[a + 1:m.end() + 60]
    if re.search(VERB + r"\b[^.]{0,80}$", sent[:m.start() - a], re.I):
        mearly.append((tgt, re.sub(r"\s+", " ", sent).strip()[:110]))

ceiling = 0
if os.path.exists(CEILING):
    ceiling = int(io.open(CEILING).read().split()[0])
total = len(early) + len(mearly)

print("=" * 92)
print("Does anything arrive before what it carries?")
print("=" * 92)
print("\n  %d symbols carry a promise in the notation table; %d appear before it"
      % (len(promises), len(early)))
for sym, sec, hit, where in early:
    print("      $%s$  first used %d characters before %s" % (sym, where - hit, sec))
print("\n  %d measurements are quoted over objects a later section defines"
      % len(mearly))
for tgt, sent in mearly:
    print("      -> %-18s %s" % (tgt, sent))
print("\n  ceiling %d, now %d" % (ceiling, total))
print("=" * 92)
ok = total <= ceiling
print("nothing arrives earlier than the ceiling allows" if ok
      else "MORE arrives early than at the last accepted run")
print("=" * 92)
sys.exit(0 if ok else 1)
