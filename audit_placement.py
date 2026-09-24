r"""The twenty-fourth audit: where does each float actually land?

audit_figorder.py checks the SOURCE -- that a float's environment begins after
the first \ref to it.  That is necessary and not sufficient, because LaTeX moves
floats.  A table whose environment sits in the right paragraph can still be
carried to the top of an earlier page by `[t]`, and the reader then meets the
picture before the argument that needs it.  tab:crb was printed one page ahead
of the sentence that cites it while every source-order check passed.

This reads the typeset PDF instead.  For every float it reports

    cited      the first page whose text sends the reader to it
    lands      the page the float is printed on
    drift      lands - cited, in pages

A drift of 0 is ideal and +1 is normal and usually invisible.  A negative drift
means the picture arrives before the reader is told why.  A drift of +2 or more
means the reference has no referent in view.  Both fail, unless the float is
listed in placement_allowed.txt with the reason it cannot be helped.

Run with no argument to check both documents, or name one.
"""
import io
import os
import re
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
CEIL = os.path.join(HERE, "placement_allowed.txt")

allowed = set()
if os.path.exists(CEIL):
    allowed = {ln.split("#")[0].strip()
               for ln in io.open(CEIL, encoding="utf-8")
               if ln.split("#")[0].strip()}

stems = sys.argv[1:] or ["mssp", "supplement"]
unexplained = []

for stem in stems:
    doc = fitz.open(os.path.join(PAPER, stem + ".pdf"))
    pages = [p.get_text() for p in doc]

    # The .aux records the printed number and the page for every label.
    aux = io.open(os.path.join(PAPER, stem + ".aux"), encoding="utf-8",
                  errors="replace").read()
    floats = {}
    for m in re.finditer(
            r"\\newlabel\{((?:fig|tab):[^}]*)\}\{\{([^}]*)\}\{(\d+)\}", aux):
        floats[m.group(1)] = (m.group(2), int(m.group(3)))

    # Source order, so the report reads down the paper.
    src = io.open(os.path.join(PAPER, stem + ".tex"), encoding="utf-8").read()
    order = []
    for m in re.finditer(r"\\label\{((?:fig|tab):[^}]*)\}", src):
        if m.group(1) not in order:
            order.append(m.group(1))

    print("=" * 84)
    print("Where each float lands, against where the reader is sent to it  "
          "(%s.pdf)" % stem)
    print("=" * 84)
    print("%-18s %-8s %6s %6s %6s   %s"
          % ("label", "printed", "cited", "lands", "drift", ""))
    print("-" * 84)

    worst = []
    for lab in order:
        if lab not in floats:
            print("%-18s %s" % (lab, "NOT IN .aux -- recompile"))
            continue
        num, land = floats[lab]
        kind = "Table" if lab.startswith("tab") else "Figure"
        # The caption prints "Figure 2: ...", so a colon right after the number
        # is the float announcing itself, not the text pointing at it.
        pat = re.compile(r"%s[\s\u00a0]*%s(?![\d.])(?!\s*:)"
                         % (kind, re.escape(num)))
        cite = None
        for i, txt in enumerate(pages, 1):
            if pat.search(txt.replace("\n", " ")):
                cite = i
                break
        if cite is None:
            print("%-18s %-8s %6s %6d %6s   never referred to in the text"
                  % (lab, num, "--", land, "--"))
            worst.append(lab)
            continue
        drift = land - cite
        note = ""
        if drift < 0:
            note = "<-- arrives before the reader is told why"
            worst.append(lab)
        elif drift >= 2:
            note = "<-- reference has no referent in view"
            worst.append(lab)
        print("%-18s %-8s %6d %6d %+6d   %s"
              % (lab, num, cite, land, drift, note))

    print("-" * 84)
    print("%d floats, %d placed more than a page from their citation\n"
          % (len(order), len(worst)))
    unexplained += [l for l in worst if l not in allowed]

print("=" * 84)
print("every float is printed where the reader is sent to it" if not unexplained
      else "not accounted for in placement_allowed.txt: "
           + ", ".join(unexplained))
print("=" * 84)
sys.exit(1 if unexplained else 0)
