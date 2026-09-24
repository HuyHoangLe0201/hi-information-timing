r"""The audited surface, which is now two files.

Material moved out of the manuscript and into the supplement has not been
deleted, but every audit that reads only paper/mssp.tex cannot tell the two
apart: a number relocated to the supplement reports as MISSING, and the honest
repair -- retiring the check -- would quietly discard the guarantee.  So the
text audits read a surface built from both files.

The supplement's body is spliced in AHEAD of the manuscript's bibliography, not
after it.  audit_paper excludes everything from "\begin{thebibliography}" to
the end of the string, so appending would have excluded the whole supplement
and the coverage figure would have improved as the guarantee was lost.

The splice is delimited in supplement.tex by two comment markers rather than by
finding the frontmatter, so what is audited is stated in the file itself.
"""
import io
import os
import re

BS = chr(92)
CL = "[" + BS + BS + "]"          # a character class matching one backslash
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
MAIN = os.path.join(PAPER, "mssp.tex")
SUPP = os.path.join(PAPER, "supplement.tex")

# A schematic drawn in TikZ is full of coordinates, and a coordinate is not a
# claim about the world: the orphan audit read forty of them as unaccounted
# figures the moment the first diagram was added.  The picture bodies are
# therefore removed from the audited surface.  Captions are NOT removed --- they
# sit outside egin{tikzpicture} --- so every number a figure states is still
# checked, and only the numbers that place ink on the page are dropped.
_PIC = re.compile(CL + r"begin\{tikzpicture\}.*?" + CL + r"end\{tikzpicture\}",
                  re.S)

A = "%%% BODY BEGINS %%%"
B = "%%% BODY ENDS %%%"


def supplement_body():
    """The part of the supplement that is prose, or "" if there is no file."""
    if not os.path.exists(SUPP):
        return ""
    s = io.open(SUPP, encoding="utf-8").read()
    a, b = s.find(A), s.find(B)
    assert a > 0 and b > a, "supplement.tex has lost its body markers"
    return s[a + len(A):b]


def read_both():
    """The manuscript with the supplement's body spliced in before the refs."""
    s = _PIC.sub(" ", io.open(MAIN, encoding="utf-8").read())
    sup = supplement_body().strip()
    if not sup:
        return s
    i = s.find(BS + "begin{thebibliography}")
    if i < 0:
        return s + "\n" + sup + "\n"
    return s[:i] + sup + "\n\n" + s[i:]
