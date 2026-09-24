"""
Assemble the IEEE Sensors Letters submission package.

Contents are the manuscript PDF, the LaTeX source needed to rebuild it, the
three figure files, the graphical abstract and the cover letter. Build
artefacts, analysis scripts and datasets are deliberately excluded: they are not
part of a submission, and the analysis code is offered on acceptance instead.

The zip is rebuilt from scratch every run, and every file is verified to exist
and be non-empty before it goes in.
"""
import os
import zipfile
import shutil
import hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "lsens_letter_v2")
SUB = os.path.join(ROOT, "submission")
ZIP = os.path.join(ROOT, "LSENS_submission.zip")

# (source path, name in the package, what it is)
ITEMS = [
    (os.path.join(SUB, "Cover_Letter_LSENS.docx"), "Cover_Letter_LSENS.docx",
     "Cover letter"),
    (os.path.join(SRC, "lsens_letter.pdf"), "lsens_letter.pdf",
     "Manuscript, 4 pages, compiled from the source below"),
    (os.path.join(SRC, "lsens_letter.tex"), "source/lsens_letter.tex",
     "LaTeX source"),
    (os.path.join(SRC, "IEEE_lsens.cls"), "source/IEEE_lsens.cls",
     "IEEE Sensors Letters class file (official template)"),
    (os.path.join(SRC, "fig1.pdf"), "source/fig1.pdf",
     "Fig. 1, vector PDF, single column"),
    (os.path.join(SRC, "fig2.pdf"), "source/fig2.pdf",
     "Fig. 2, vector PDF, double column"),
    (os.path.join(SRC, "gabs.pdf"), "source/gabs.pdf",
     "Graphical abstract, vector PDF"),
]

missing = [p for p, _, _ in ITEMS if not (os.path.isfile(p) and os.path.getsize(p))]
if missing:
    raise SystemExit("missing or empty:\n  " + "\n  ".join(missing))

manifest = [
    "IEEE Sensors Letters - Regular Letter submission",
    "",
    "Title    : A Closed-Form Minimum-Window Rule for Sensor-Driven",
    "           Remaining-Useful-Life Prediction",
    "Authors  : Huy Hoang Le, Kim-Anh Nguyen",
    "Corresp. : K.-A. Nguyen, nkanh@dut.udn.vn",
    "Subject  : Sensor Signal Processing",
    "",
    "Contents",
    "--------",
]
for _, name, what in ITEMS:
    manifest.append(f"  {name:<32} {what}")
manifest += [
    "",
    "Notes",
    "-----",
    "  The manuscript compiles with pdflatex from source/ ; it needs the newtx,",
    "  helvetic and courier packages in addition to a standard TeX Live.",
    "  Figures are vector PDFs and carry no rasterised text.",
    "  The PRONOSTIA dataset analysed in Section V is publicly available; the",
    "  analysis code will be released upon acceptance.",
]
man_path = os.path.join(SUB, "MANIFEST.txt")
open(man_path, "w", encoding="utf-8").write("\n".join(manifest) + "\n")

if os.path.exists(ZIP):
    os.remove(ZIP)
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(man_path, "MANIFEST.txt")
    for path, name, _ in ITEMS:
        z.write(path, name)

print(f"wrote {ZIP}\n")
with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    print(f"integrity: {'OK' if bad is None else 'CORRUPT at ' + bad}")
    print(f"{'entry':<34}{'bytes':>10}")
    print("-" * 46)
    total = 0
    for i in z.infolist():
        total += i.file_size
        print(f"{i.filename:<34}{i.file_size:>10,}")
    print("-" * 46)
    print(f"{'uncompressed total':<34}{total:>10,}")
    print(f"{'zip on disk':<34}{os.path.getsize(ZIP):>10,}")
