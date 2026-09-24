r"""Does the paper's argument run forwards?

Splitting Section 2 exposed a dependency that had been invisible while one
section held twenty-two propositions: Section 5 opened by invoking a proposition
stated two sections later.  Nothing caught it, because a forward reference is
legal LaTeX and reads as ordinary prose.

A forward reference is not automatically a fault.  Floats move, so a figure or
table cited before it appears is normal typesetting, and a paper may legitimately
promise a later result.  What should be rare is a section or a proposition whose
ARGUMENT needs something the reader has not been given, and rarest of all is one
in an opening sentence, where the reader has nothing else to hold on to.

So this reports forward references by kind, and separately those that sit in the
first sentence of a section, which is where they do the most damage.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
s = io.open(TEX, encoding="utf-8").read()
s = re.sub(r"(?m)^%.*$", "", s)

# where each label is defined
label_at = {}
for m in re.finditer(r"\\label\{([^}]*)\}", s):
    label_at.setdefault(m.group(1), m.start())

# section starts, to attribute each reference to a section and find openings
secs = []
for m in re.finditer(r"\\section\*?\{([^}]*)\}", s):
    secs.append((m.start(), m.group(1)))
secs.sort()


def section_of(pos):
    cur = ("", "front matter")
    for at, title in secs:
        if at <= pos:
            cur = (at, title)
        else:
            break
    return cur


KIND = [("sec", "section"), ("prop", "proposition"), ("eq", "equation"),
        ("tab", "table"), ("fig", "figure"), ("rem", "remark")]
rows = []
for m in re.finditer(r"\\(?:eq)?ref\{([^}]*)\}", s):
    tgt = m.group(1)
    if tgt not in label_at:
        continue
    if label_at[tgt] <= m.start():
        continue                       # backward or self: fine
    kind = next((k for p, k in KIND if tgt.startswith(p + ":")), "other")
    at, title = section_of(m.start())
    # is it in the section's opening sentence?
    opening = False
    if at != "":
        head = s[at:m.end()]
        body = re.sub(r"\\(?:sub)*section\*?\{[^}]*\}|\\label\{[^}]*\}", "",
                      head).strip()
        opening = body.count(".") <= 1
    rows.append(dict(target=tgt, kind=kind, section=title, opening=opening,
                     at=m.start()))

print("=" * 92)
print("References that point forwards")
print("=" * 92)
print("""
A reference is forward when its target is defined later in the source than the
sentence citing it.  Floats are exempt in practice and are counted separately;
sections and propositions are the ones that shape how the paper reads.
""")
by_kind = {}
for r in rows:
    by_kind.setdefault(r["kind"], []).append(r)
print("  %-14s %8s %10s" % ("kind", "forward", "in an opening"))
print("  " + "-" * 38)
for k in ("section", "proposition", "equation", "remark", "table", "figure",
          "other"):
    v = by_kind.get(k, [])
    if v:
        print("  %-14s %8d %10d"
              % (k, len(v), sum(1 for r in v if r["opening"])))

hard = [r for r in rows if r["kind"] in ("section", "proposition")]
print("\n  the %d forward references to sections and propositions" % len(hard))
print("  " + "-" * 88)
for r in sorted(hard, key=lambda r: r["at"]):
    mark = "  <-- in the opening sentence" if r["opening"] else ""
    print("  %-26s from %-44s%s" % (r["target"], r["section"][:44], mark))

_op = [r for r in hard if r["opening"]]
print("""
  %d of %d forward references to a section or a proposition sit in the sentence
  that opens a section.  Those are the ones worth removing: a reader meeting a
  section's first sentence has nowhere to look up what it assumes.
""" % (len(_op), len(hard)))
