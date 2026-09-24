r"""The eleventh audit: does the front matter agree with the paper behind it?

WHY.  The coverage report of audit_paper.py shows where its checks do not reach,
and the largest single block is not an obscure appendix: it is the abstract and
the introduction.  Those numbers are restatements, so no result file backs them
directly, and every numeric check is anchored on the sentence in the body that
first computes the quantity.  The consequence is that the two pages a reader and
a referee actually read are the least verified in the manuscript.  Correct a
figure in Section 12 and nothing at all reports that the abstract still carries
the old one.

WHAT IS CHECKED.  Let F be the numbers appearing in the abstract and the
introduction and B those appearing from the first numbered section after the
introduction onwards.  Two conditions:

  (ECHO)    every number in F appears in B;
  (ANCHOR)  every number in F appears in B in a sentence that names the same
            quantity, judged by requiring a shared content word within the
            surrounding clause.

ECHO is a hard requirement and a failure is a defect.  ANCHOR is advisory: a
shared word is a weak test and it is reported for reading rather than counted,
because the alternative -- a table of hand-written pairings -- would itself need
maintaining and would fail silently the moment a sentence was rewritten.

WHAT IS NOT CHECKED, and why.  Author identifiers, street addresses, dataset
names and years are numbers that carry no claim, and they are excluded by
pattern rather than by hand so that the exclusion cannot quietly grow.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
# The abstract and the introduction restate numbers computed later, and
# "later" now includes the supplement: a figure quoted in the discussion
# whose measurement moved out of the manuscript is still a restatement and
# not a new claim.  The front-matter side is the manuscript alone.
import _source
s = re.sub("(?m)^%.*$", "", _source.read_both())

# --- the two regions ---------------------------------------------------------
# The front matter runs from the abstract to the end of the introduction; the
# body runs from the next numbered section to the bibliography.
a0 = s.find(BS + "begin{abstract}")
secs = [m.start() for m in re.finditer(CL + "section{", s)]
assert a0 > 0 and len(secs) > 2, "the manuscript does not have the expected shape"
front = s[a0:secs[1]]
body = s[secs[1]:s.find(BS + "begin{thebibliography}")]

# --- numbers that carry a claim ----------------------------------------------
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
YEAR = re.compile(r"^(19|20)\d\d$")


def numbers(text, keep_context=False):
    # strip what cannot carry a claim: identifiers, addresses, citation keys and
    # labels, all of which contain digits that mean nothing about the results
    t = re.sub(CL + r"orcidlink{[^}]*}", " ", text)
    t = re.sub(CL + r"(cite|ref|eqref|label|ead|address|affiliation)"
               r"(\[[^\]]*\])?{[^}]*}", " ", t)
    t = re.sub(CL + r"(author|affiliation)\[[^\]]*\]", " ", t)
    out = []
    for m in NUM.finditer(t):
        v = m.group(0)
        if len(v.replace(".", "")) < 2 or YEAR.match(v):
            continue
        if keep_context:
            out.append((v, re.sub(r"\s+", " ",
                                  t[max(0, m.start() - 95):m.end() + 45]).strip()))
        else:
            out.append(v)
    return out


F = numbers(front, keep_context=True)
B = set(numbers(body))
Bctx = numbers(body, keep_context=True)

STOP = set("""a an the and or of in on at to for is are was were be been by with
from that this those these it its as not no than then so such which what when
where how any all both each other same one two three more most less least than
over under about into onto per up down out off very can may might will would
shall should must do does did have has had they them their there here we our
us you your he she his her him i if but nor yet only also just even still
number numbers value values figure figures table tables section sections
paper papers result results record records unit units""".split())


def words(t):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in STOP}


missing, weak = [], []
for v, ctx in F:
    if v not in B:
        missing.append((v, ctx))
        continue
    fw = words(ctx)
    if not any(fw & words(c) for w, c in Bctx if w == v):
        weak.append((v, ctx))

print("=" * 92)
print("Does the front matter agree with the paper behind it?")
print("=" * 92)
print("""
Every number in the abstract and the introduction is a restatement of one
computed later, so none of them is checked by a result file.  This compares them
against the body instead.
""")
print("  numbers carrying a claim in the front matter : %d" % len(F))
print("  distinct numbers in the body                 : %d" % len(B))

print("\n  ECHO: ", end="")
if not missing:
    print("every front-matter number appears in the body")
else:
    print("%d appear nowhere in the body" % len(missing))
    for v, ctx in missing:
        print("     %-9s ...%s" % (v, ctx[-88:]))

print("  ANCHOR: ", end="")
if not weak:
    print("each also appears in a body sentence sharing a content word")
else:
    print("%d appear in the body but in no sentence sharing a content word,"
          % len(weak))
    print("          which is advisory and not a failure:")
    for v, ctx in weak:
        print("     %-9s ...%s" % (v, ctx[-88:]))

# --- the other end of the paper, which also restates -------------------------
# The abstract and the introduction are not the only sections that repeat a
# figure computed elsewhere: the discussion and the conclusion do it too, and
# nothing was checking them.  That cost a defect.  The discussion said the
# serial correction moves an age "by up to 0.281 of a lifetime" where the
# section it summarises measures 0.283, and no layer reported it: the numeric
# audit had not bound the sentence, and the orphan layer passed it because an
# unrelated result file happens to hold 0.28122.
#
# The test cannot be as strict here, because the discussion also computes things
# of its own -- the retention null, the sweeps over the pipeline's free choices,
# the relaxed FD004 fleet -- and those legitimately appear nowhere else.  So the
# numbers appearing nowhere else are listed and held under a recorded ceiling
# that may fall and may not rise.  A restatement that drifts becomes the
# ceiling-plus-one and is reported.
CEIL = os.path.join(HERE, "restated_ceiling.txt")
d0 = s.find(BS + "section{Discussion}")
d1 = s.find(BS + "section*{Declaration of competing interest")
tail = s[d0:d1] if 0 < d0 < d1 else ""
elsewhere = set(numbers(s[:d0] + s[d1:s.find(BS + "begin{thebibliography}")]))
own = [(v, c) for v, c in numbers(tail, keep_context=True) if v not in elsewhere]

print("\n  the discussion and the conclusion restate as well as compute")
print("  numbers there                                : %d"
      % len(numbers(tail)))
print("  appearing nowhere else in the paper          : %d" % len(own))
prev = None
if os.path.exists(CEIL):
    try:
        prev = int(io.open(CEIL, encoding="utf-8").read().split()[0])
    except Exception:
        prev = None
tail_ok = prev is None or len(own) <= prev
if prev is None or len(own) < prev:
    io.open(CEIL, "w", encoding="utf-8").write(
        "%d\n\nNumbers in the discussion and the conclusion that appear nowhere\n"
        "else in the paper, at the last run that was accepted.  Those are the\n"
        "ones those sections compute for themselves; a rise means one of their\n"
        "restatements no longer matches what it restates.\n" % len(own))
if prev is not None:
    print("  ceiling %d, now %d" % (prev, len(own)))
if not tail_ok:
    print("\n  new numbers, one of which is a restatement that has drifted:")
    for v, c in own:
        print("     %-9s ...%s" % (v, c[-84:]))

print()
print("=" * 92)
ok = (not missing) and tail_ok
print("every restating section is backed by the sections that compute" if ok
      else "a restating section states numbers the rest of the paper does not")
print("=" * 92)
sys.exit(0 if ok else 1)
