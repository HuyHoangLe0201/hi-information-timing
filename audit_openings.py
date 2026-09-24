r"""The thirteenth audit: can a reader tell what each section is for?

The order audit checks that no section opens by assuming something the reader
has not been given.  That is a statement about dependency, not about legibility:
a section can open with nothing unexplained and still not say what it is for.
This reads the first sentence of every section and appendix and asks four
mechanical questions about it.

  LEAD      does the section have an opening sentence at all, or is its first
            content a subsection heading?  Three sections failed this on the
            first run, and one of them was the section carrying the paper's
            applied result: a reader arriving at the conclusion of the work met a
            subsection title and a table before any sentence saying what was
            being claimed.  This is the only one of the four that is a defect
            rather than a symptom, and it is the one worth having.
  SUBJECT   does the opening share a content word with the section's own title,
            or with the title of the section it follows?  An opening that shares
            neither is starting somewhere the reader has no reason to expect.
  FORM      does the section open with a display, a proposition or a table
            rather than a sentence?  A reader who meets a formula first has to
            infer the question it answers.
  LENGTH    is the opening sentence longer than forty-five words?  A first
            sentence that long is doing the work of a paragraph.

None of these is a defect on its own, which is why they are reported as a table
and counted against a ceiling rather than failed individually.  The table is the
point: it is the paper's spine written out, one line per section, and it can be
read in half a minute to see whether the argument still follows.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
CEIL = os.path.join(HERE, "openings_ceiling.txt")

s = io.open(TEX, encoding="utf-8").read()
s = re.sub("(?m)^%.*$", "", s)

secs = []
for m in re.finditer(CL + r"section[*]?\{([^}]*)\}", s):
    secs.append([m.start(), m.end(), m.group(1)])
for i, x in enumerate(secs):
    x.append(secs[i + 1][0] if i + 1 < len(secs) else len(s))

# The declarations are forms of words the journal sets, not sections an argument
# opens; the generative-AI one is matched on its first words because its title
# breaks across a source line.
SKIP = ("CRediT authorship contribution statement",
        "Declaration of competing interest", "Data availability", "Funding",
        "Declaration of generative AI")

STOP = set("""a an the and or of in on at to for is are was were be been by with
from that this those these it its as not no than then so such which what when
where how any all both each other same one two three more most less least over
under about into onto per up down out off very can may might will would shall
should must do does did have has had they them their there here we our us you
your only also just even still what does what is""".split())


def words(t):
    return {w for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in STOP}


def first_sentence(seg):
    """The first real sentence, skipping labels and any float that precedes it."""
    body = seg
    body = re.sub(CL + r"label\{[^}]*\}", " ", body)
    for env in ("figure", "table"):
        body = re.sub(CL + r"begin\{" + env + r"[*]?\}.*?" + CL + r"end\{"
                      + env + r"[*]?\}", " ", body, flags=re.S)
    body = body.strip()
    opens_with = None
    m = re.match(CL + r"begin\{(proposition|equation|remark|definition)", body)
    if m:
        opens_with = m.group(1)
    m = re.match(CL + r"(sub)?section[*]?\{[^}]*\}", body)
    if m:
        body = body[m.end():]
    body = re.sub(CL + r"label\{[^}]*\}", " ", body).strip()
    m2 = re.match(CL + r"begin\{(proposition|equation|remark|definition)", body)
    if m2 and opens_with is None:
        opens_with = m2.group(1)
    plain = re.sub(CL + r"[a-zA-Z]+", " ", body)
    plain = plain.translate({ord(c): " " for c in "{}$" + BS})
    plain = re.sub(r"\s+", " ", plain).strip()
    m3 = re.search(r"(.+?[.?])(\s|$)", plain)
    return (m3.group(1) if m3 else plain[:200]), opens_with


rows = []
for i, (a, e, title, b) in enumerate(secs):
    if title in SKIP or title.startswith(SKIP[-1]):
        continue
    sent, opens = first_sentence(s[e:b])
    prev = secs[i - 1][2] if i else ""
    w = words(sent)
    subject = bool(w & words(title)) or bool(w & words(prev))
    nwords = len(sent.split())
    head = re.sub(CL + r"label\{[^}]*\}", " ", s[e:e + 900]).strip()
    lead = not head.startswith(BS + "subsection")
    rows.append(dict(title=title, sentence=sent, opens=opens, lead=lead,
                     subject=subject, nwords=nwords))

print("=" * 96)
print("The spine, one line per section: what does each one open by saying?")
print("=" * 96)
print()
flags = 0
for r in rows:
    mark = []
    if not r["lead"]:
        mark.append("NO LEAD")
    if not r["subject"]:
        mark.append("SUBJECT")
    if r["opens"]:
        mark.append("OPENS WITH A " + r["opens"].upper())
    if r["nwords"] > 45:
        mark.append("%d WORDS" % r["nwords"])
    flags += len(mark)
    print("  %-46s %s" % (r["title"][:46], ("  <-- " + ", ".join(mark))
                          if mark else ""))
    print("      %s" % r["sentence"][:150])
    print()

print("=" * 96)
print("  %d sections read, %d flags" % (len(rows), flags))
prev = None
if os.path.exists(CEIL):
    try:
        prev = int(io.open(CEIL, encoding="utf-8").read().split()[0])
    except Exception:
        prev = None
# A missing lead is not a matter of degree, so it fails outright; the other
# three are held under a ceiling.
noleads = sum(1 for r in rows if not r["lead"])
ok = (prev is None or flags <= prev) and noleads == 0
if prev is None or flags < prev:
    io.open(CEIL, "w", encoding="utf-8").write(
        "%d\n\nFlags raised by the opening-sentence audit at the last run that\n"
        "was accepted.  It may be lowered by rewriting an opening and may not\n"
        "be raised.\n" % flags)
if prev is not None:
    print("  ceiling %d, now %d" % (prev, flags))
if noleads:
    print("  %d sections open on a subsection heading with no lead"
          % noleads)
print("every section opens by saying what it is for" if flags == 0
      else ("the openings are at or below the ceiling" if ok
            else "MORE openings are unclear than at the last accepted run"))
print("=" * 96)
sys.exit(0 if ok else 1)
