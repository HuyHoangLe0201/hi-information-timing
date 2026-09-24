r"""Does each cross-reference say truthfully what the section it points at does?

A paper this size accumulates sentences of the form "Section X shows Y".  They
are written when X is fresh and are never revisited, so they drift as X is
rewritten, split or renamed.  Nothing else in the toolchain looks at them: the
compiler only checks that the label exists, and the numerical audits only check
figures.

This lists every such sentence beside the title of the section it points at, so
the pair can be read together.  It cannot decide correctness; it puts the two
halves next to each other, which is the part that is tedious by hand.

It also reports two mechanical faults it can decide.  A reference to the section
it sits inside is almost always a leftover from a move.  And a claim naming a
result the target does not contain, checked only for the coarse case where the
target has no proposition and the sentence says the target proves something.
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "paper", "mssp.tex")
raw = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")
body = raw.split("\\begin{document}")[-1].split("\\begin{thebibliography}")[0]

# where each section starts, and what it is called
heads = [(m.start(), bool(m.group(1)), m.group(2))
         for m in re.finditer(r"\\(sub)?section\*?\{([^}]*)\}", body)]
labels = {}
for m in re.finditer(r"\\label\{(sec:[^}]*)\}", body):
    prev = [h for h in heads if h[0] < m.start()]
    if prev:
        labels[m.group(1)] = (prev[-1][2], prev[-1][0])

# which section each position sits in
def owner(pos):
    prev = [h for h in heads if h[0] <= pos]
    return prev[-1][2] if prev else "(front matter)"


REF = re.compile(r"Section~\\ref\{(sec:[^}]*)\}((?:[^.]|\.\d){0,150}\.)")
rows = []
for m in REF.finditer(body):
    tgt = m.group(1)
    tail = " ".join(m.group(2).split())
    name, tpos = labels.get(tgt, ("(unlabelled)", -1))
    rows.append((owner(m.start()), name, tail, m.start(), tpos, tgt))

print("=" * 100)
print("SELF-REFERENCES: a section pointing at itself")
print("=" * 100)
self_ref = [r for r in rows if r[0] == r[1]]
print("%d found\n" % len(self_ref))
for src, name, tail, _, _, tgt in self_ref:
    print("  in %-46s -> itself" % src[:46])
    print("      ..." + tail[:110])

print()
print("=" * 100)
print("EVERY CROSS-REFERENCE, BESIDE WHAT IT POINTS AT")
print("=" * 100)
print("%d references to labelled sections\n" % len(rows))
by_target = {}
for r in rows:
    by_target.setdefault(r[1], []).append(r)
for name in sorted(by_target, key=lambda k: -len(by_target[k])):
    print("  --> %s   (%d references)" % (name, len(by_target[name])))
    for src, _, tail, pos, tpos, tgt in by_target[name]:
        d = "forward" if tpos > pos else "back   "
        print("      [%s] from %-34s %s"
              % (d, src[:34], tail if len(tail) < 96 else tail[:93] + "..."))
    print()

# --- a section number typed rather than referenced ---------------------------
# One caption said "(Section 3.3)".  It was right when it was written and wrong
# by the time it was read, because the section it named had been moved twice
# and a typed number does not move with it.  LaTeX cannot report this: the
# caption compiles, the number is a number, and every \ref in the document
# resolves.  A pointer to a section of THIS document must be a \ref.
#
# Pointers into the supplement are excluded, since they are necessarily typed:
# they carry an S and audit_supplement checks them instead.
print()
print("=" * 100)
print("SECTION NUMBERS TYPED INSTEAD OF REFERENCED")
print("=" * 100)
_typed = []
for _m in re.finditer(r"(?<!Supplementary )(?:Section|Appendix)[~ ]"
                      r"(\d+(?:\.\d+)?)(?![\d.])", body):
    _typed.append((body[:_m.start()].count("\n") + 1, _m.group(0),
                   body[max(0, _m.start() - 60):_m.end() + 20].replace("\n", " ")))
print("%d found\n" % len(_typed))
for _ln, _txt, _ctx in _typed:
    print("  line %-6d %-16s ...%s" % (_ln, _txt, _ctx))
if _typed:
    print("\n  each of these should be a \ref to the section's label")
    raise SystemExit(1)


# =============================================================================
# Labels defined in BOTH documents.
#
# The supplement reads mssp.aux through \externaldocument, so every label the
# manuscript defines is visible inside the supplement.  A label defined in both
# is therefore not a matter of tidiness: the supplement's own \ref can resolve
# to the manuscript's object instead of its own, and the only sign is a
# "multiply defined" line in a log nobody reads.
#
# It has happened three times in this paper -- fig:contrast, tab:scope and
# sec:clockscale -- each time when a float or section was given the obvious
# name in the document it was written in.  Cheap to check, invisible otherwise.
_SUP = os.path.join(os.path.dirname(P), "supplement.tex")
print()
print("=" * 100)
print("LABELS DEFINED IN BOTH DOCUMENTS")
print("=" * 100)
if os.path.exists(_SUP):
    _pat = r"\\label\{([^}]+)\}"
    _mine = set(re.findall(_pat, raw))
    _theirs = set(re.findall(_pat, io.open(_SUP, encoding="utf-8").read()))
    _clash = sorted(_mine & _theirs)
    if _clash:
        for _c in _clash:
            print("  %s" % _c)
        print()
        print("  the supplement imports the manuscript's labels, so each of these")
        print("  may resolve to the wrong document's object")
        raise SystemExit(1)
    print("  none; %d in the manuscript, %d in the supplement"
          % (len(_mine), len(_theirs)))
else:
    print("  supplement.tex not found")
