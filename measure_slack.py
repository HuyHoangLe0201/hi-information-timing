"""
How much can still be ADDED before the Letter spills onto a fifth page?

Position-based estimates are unreliable (headers, float placement), so this
measures it directly: inject filler sentences just before the Conclusion,
bisect on the count, and report the largest number that still compiles to 4
pages. That number is the real review-round budget.
"""
import os
import re
import shutil
import subprocess
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "lsens_letter_v2")
TEXNAME = "lsens_letter.tex"
LATEX = r"D:\texlive\2026\bin\windows\pdflatex.exe"

FILLER = ("The estimator was additionally checked against an independent "
          "implementation, and the reported figures were reproduced to within "
          "the stated tolerances in every case considered here. ")


def pages_with(n_sentences, workdir):
    tex = open(os.path.join(SRC, TEXNAME), encoding="utf-8").read()
    if n_sentences:
        block = "\n\n" + (FILLER * n_sentences).strip() + "\n\n"
        tex = tex.replace(r"\section{Conclusion}", block + r"\section{Conclusion}")
    p = os.path.join(workdir, TEXNAME)
    open(p, "w", encoding="utf-8").write(tex)
    for _ in range(2):
        subprocess.run([LATEX, "-interaction=nonstopmode", TEXNAME],
                       cwd=workdir, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    log = open(os.path.join(workdir, "lsens_letter.log"),
               encoding="utf-8", errors="replace").read()
    m = re.search(r"Output written on .*?\((\d+) pages", log)
    return int(m.group(1)) if m else -1


with tempfile.TemporaryDirectory() as wd:
    for f in os.listdir(SRC):
        if f.endswith((".cls", ".pdf", ".py")):
            shutil.copy(os.path.join(SRC, f), wd)

    base = pages_with(0, wd)
    print(f"baseline: {base} pages")
    if base != 4:
        raise SystemExit("baseline is not 4 pages; aborting")

    lo, hi = 0, 1
    while pages_with(hi, wd) == 4 and hi < 64:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pages_with(mid, wd) == 4:
            lo = mid
        else:
            hi = mid
    words = len(FILLER.split()) * lo
    print(f"\nfits: {lo} filler sentences  ({words} words) before page 5")
    print(f"a body line is ~11 words, so that is roughly {words // 11} free lines")
    print(f"i.e. about {words / 2974 * 100:.0f}% of the current body length "
          f"({2974} words) can still be added.")
