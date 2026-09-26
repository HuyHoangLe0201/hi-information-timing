r"""Assemble the Mechanical Systems and Signal Processing submission.

Two things go in the archive and they are for different readers.

  submission/   what the editor uploads: the manuscript, the supplementary
                material, the highlights, the graphical abstract, a cover
                letter, and the LaTeX source a production office needs to
                rebuild the PDF.
  code/         the analysis the paper says is released with it: every script
                that produces a number, every result file the audits check the
                prose against, and the thirty-four audit layers themselves.

WHAT IS DELIBERATELY LEFT OUT.  work/.backup is 23 MB of superseded copies of
the manuscript and is nobody's business but this project's.  The .npz feature
caches are 22 MB of intermediate arrays derived from five public datasets; the
scripts that build them are included and the datasets are cited, so the caches
are reproducible rather than primary.  code/README.md says how to rebuild them.
The unused leftovers in paper/ -- the CAS class files, figflow.pdf -- are not
shipped either, because neither document loads them.

THE ARCHIVE IS NAMED FOR ITS STATE.  While a declaration in the manuscript
still reads UNFILLED the file is called ..._DRAFT.zip and this script says why.
A submission with an unwritten CRediT statement or competing-interest statement
is returned, and naming the archive "final" would be the packaging telling a
comfortable lie about the paper.

Every file is checked to exist and to be non-empty before it is added, and the
manifest prints a SHA-256 for the two PDFs so the upload can be verified.
"""
import hashlib
import io
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper")

# ---- what the editor uploads -------------------------------------------------
FIGURES = ["fig1.pdf", "fig2.pdf", "fig3.pdf", "fig4.pdf", "fig5.pdf",
           "figdistort.pdf", "figmodel.pdf"]
SUBMISSION = [
    (os.path.join(PAPER, "mssp.pdf"), "submission/Manuscript.pdf",
     "Manuscript, compiled from the source below"),
    (os.path.join(PAPER, "supplement.pdf"),
     "submission/Supplementary_material.pdf", "Supplementary material"),
    (os.path.join(PAPER, "highlights.txt"), "submission/Highlights.txt",
     "Highlights, five bullets within the 85-character limit"),
    (os.path.join(PAPER, "gabs.pdf"), "submission/Graphical_abstract.pdf",
     "Graphical abstract, vector PDF"),
    (os.path.join(HERE, "Cover_letter.md"), "submission/Cover_letter.md",
     "Cover letter (DRAFT: the authors must confirm the declarations)"),
    (os.path.join(HERE, "Cover_Letter_MSSP.docx"),
     "submission/Cover_Letter_MSSP.docx",
     "Cover letter, Word, one page (make_cover_letter_docx.py)"),
    (os.path.join(PAPER, "mssp.tex"), "submission/source/mssp.tex",
     "Manuscript source"),
    (os.path.join(PAPER, "supplement.tex"), "submission/source/supplement.tex",
     "Supplementary source"),
    (os.path.join(PAPER, "_suppbib.tex"), "submission/source/_suppbib.tex",
     "Supplement bibliography, \\input by supplement.tex"),
    (os.path.join(PAPER, "gabs.tex"), "submission/source/gabs.tex",
     "Graphical abstract source"),
    (os.path.join(PAPER, "elsarticle-num.bst"),
     "submission/source/elsarticle-num.bst",
     "Bibliography style named by supplement.tex"),
] + [(os.path.join(PAPER, f), "submission/source/" + f, "Figure")
     for f in FIGURES]

# ---- the analysis ------------------------------------------------------------
SKIP_DIRS = {".backup", "__pycache__", ".git"}
# .png are preview renders made while the figures were being drawn; .tex here
# are stale fragments no document inputs, the live ones going to
# submission/source; the caches and build litter speak for themselves.
SKIP_EXT = {".npz", ".pyc", ".pdf", ".png", ".tex", ".log", ".aux", ".docx"}
# Project notes rather than analysis: how the paper came to be split, what the
# outline was. README.md is written for whoever opens the released code.
SKIP_NAMES = {"Cover_letter.md", "OUTLINE.md", "SPLIT.md",
              "make_cover_letter_docx.py"}


def code_files():
    out = []
    for name in sorted(os.listdir(HERE)):
        p = os.path.join(HERE, name)
        if os.path.isdir(p) or name in SKIP_DIRS or name in SKIP_NAMES:
            continue
        if os.path.splitext(name)[1].lower() in SKIP_EXT:
            continue
        # the helpers that edited the manuscript, and their captured output;
        # _source.py is kept because the audit layers import it
        if name.startswith("_") and name != "_source.py":
            continue
        out.append((p, "code/" + name))
    return out


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


# ---- is the manuscript submissible? -----------------------------------------
tex = io.open(os.path.join(PAPER, "mssp.tex"), encoding="utf-8").read()
unfilled = len(re.findall(r"UNFILLED", tex))
state = "DRAFT" if unfilled else "final"
ZIP = os.path.join(ROOT, "MSSP_submission_%s.zip" % state)

missing = [p for p, _, _ in SUBMISSION
           if not (os.path.isfile(p) and os.path.getsize(p))]
if missing:
    print("cannot build: these are missing or empty")
    for p in missing:
        print("   " + p)
    sys.exit(1)

if os.path.exists(ZIP):
    os.remove(ZIP)
items = [(p, a) for p, a, _ in SUBMISSION] + code_files()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for src, arc in items:
        z.write(src, arc)

print("=" * 78)
print("MSSP submission package")
print("=" * 78)
print("\nsubmission/ -- what the editor uploads")
for p, a, what in SUBMISSION:
    print("  %-42s %7.1f KB  %s" % (a, os.path.getsize(p) / 1024.0, what))
code = code_files()
print("\ncode/ -- the analysis released with the paper")
print("  %d files, %.1f MB: the scripts that produce every number, the result"
      % (len(code), sum(os.path.getsize(p) for p, _ in code) / 1048576.0))
print("  files the audits read, and the thirty-four audit layers.")
print("\n%s  %.1f MB" % (os.path.basename(ZIP),
                         os.path.getsize(ZIP) / 1048576.0))
for f in ("mssp.pdf", "supplement.pdf"):
    print("  sha256 %-18s %s" % (f, sha(os.path.join(PAPER, f))))

if unfilled:
    print("\n" + "!" * 78)
    print("NOT SUBMISSIBLE. %d declaration(s) in the manuscript still read"
          " UNFILLED." % unfilled)
    print("The remaining declarations are statements about the authors; an editor")
    print("returns a manuscript that leaves them blank.  The archive is named")
    print("DRAFT for that reason.  Fill them, rebuild, and run this again.")
    print("!" * 78)
sys.exit(0)
