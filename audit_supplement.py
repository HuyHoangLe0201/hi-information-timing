r"""The seventeenth audit: do the pointers into the supplement point anywhere?

The manuscript sends a reader to the supplement by number -- "Supplementary
Section~S4" -- and that number is plain text, not a \ref, because LaTeX will not
resolve a label across two documents in the direction that matters here.  Plain
text does not fail loudly: inserting one section in the supplement renumbers
every pointer after it, and the manuscript still compiles, still passes every
other layer, and now sends the reader to the wrong place.

WHAT IS CHECKED.

  (EXISTS)  every SN named in paper/mssp.tex is a section of the supplement;
  (USED)    every section of the supplement is named at least once, so nothing
            was moved out of the paper and then orphaned;
  (SHAPE)   the manuscript contains no "Supplementary Table~S..." or
            "Supplementary Figure~S..." pointer, because table and figure
            numbers in the supplement depend on float placement and are not
            stable enough to quote from another document.

The manuscript is read with its line breaks collapsed, since the pointer is
often broken across a line as "Supplementary\nSection~S7".
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
MAIN = os.path.join(PAPER, "mssp.tex")
SUPP = os.path.join(PAPER, "supplement.tex")

main = re.sub(r"\s+", " ", io.open(MAIN, encoding="utf-8").read())
supp = io.open(SUPP, encoding="utf-8").read()

titles = re.findall(CL + "section{([^}]*)}", supp)
# the frontmatter carries no \section, so every hit is a numbered section
have = {i + 1: t for i, t in enumerate(titles)}

named = [int(n) for n in
         re.findall(r"Supplementary Section~?S(\d+)", main)]
floats = re.findall(r"Supplementary (?:Table|Figure)~?S\d+", main)

print("=" * 92)
print("Do the pointers into the supplement point anywhere?")
print("=" * 92)
print("\n  the supplement has %d sections; the manuscript makes %d pointers\n"
      % (len(have), len(named)))
print("  %-5s %-64s %s" % ("", "supplement section", "pointers"))
print("  " + "-" * 84)
for n, t in have.items():
    c = named.count(n)
    print("  %-5s %-64s %s"
          % ("S%d" % n, t[:64], c if c else "NEVER NAMED"))

dangling = sorted({n for n in named if n not in have})
orphan = [n for n in have if n not in named]

print("\n  EXISTS: ", end="")
print("every pointer names a section that exists" if not dangling
      else "NO SUCH SECTION: " + ", ".join("S%d" % n for n in dangling))
print("  USED:   ", end="")
print("every section is pointed to" if not orphan
      else "NEVER NAMED: " + ", ".join("S%d" % n for n in orphan))
print("  SHAPE:  ", end="")
print("no pointer quotes a float number" if not floats
      else "QUOTES A FLOAT: " + ", ".join(sorted(set(floats))))

print()
print("=" * 92)
ok = not dangling and not orphan and not floats
print("the manuscript and the supplement agree on their numbering" if ok
      else "the manuscript points into the supplement incorrectly")
print("=" * 92)
sys.exit(0 if ok else 1)
