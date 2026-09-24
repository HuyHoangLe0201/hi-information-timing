"""
NOT PART OF THIS PAPER'S AUDIT SUITE.  Superseded by audit_letter.py.

This layer reads the PREDECESSOR study's cover letter -- a .docx addressed to
IEEE Sensors Letters -- and checks it against that study's manuscript.  Almost
every check is specific to that submission: the subject code, the companion
paper's article number, the co-author who was corresponding there, and the rule
that the letter must quote no statistic at all.  This paper's letter is to a
different journal, is written in Markdown, and is deliberately quantitative, so
the rule it enforces is the opposite of the right one here.

Repointing it was tried and is wrong for the same reason audit_figures.py could
not be repointed: the path was never the problem.  audit_letter.py is the check
this paper needs, and it exists because the overclaim this layer would have been
useless against -- "attained to within 9%" with no exception named -- did reach
the letter and survived to the reviewer read.

Run it and it fails on a missing file.  That is correct: it is not a check on
this paper and should not look like one.
"""
"""
Cross-check the cover letter against the manuscript and the results files.

A cover letter is read by an editor beside the abstract; a number that disagrees
with the paper is the worst possible first impression. Everything quantitative
in the letter is checked here against its source.
"""
import os
import re
import json
from docx import Document

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
LET = Document(os.path.join(ROOT, "submission", "Cover_Letter_LSENS.docx"))
TXT = " ".join(p.text for p in LET.paragraphs)
TXT = re.sub(r"\s+", " ", TXT)
TEX = open(os.path.join(ROOT, "lsens_letter_v2", "lsens_letter.tex"),
           encoding="utf-8").read()
oof = json.load(open(os.path.join(HERE, "oof_results.json")))
A2 = oof["configs"]["A2"]

rows = []


def chk(label, present, ok, detail=""):
    rows.append((label, "yes" if present else "no", "ok" if ok else "MISMATCH", detail))


# --- the letter is deliberately qualitative --------------------------------
# No statistic should appear: the abstract carries those. Anything numeric other
# than the citation to the companion paper is flagged so it cannot creep back.
# A statistic here looks like a decimal or a percentage. Enumeration markers,
# the postcode and the citation are integers and are not what we are guarding
# against; they are listed so the distinction stays visible.
stats = re.findall(r"\d+\.\d+|\d+\s*%", TXT)
ints = sorted(set(re.findall(r"(?<![\d.])\d+(?![\d.%])", TXT)))
rows.append(("no statistics quoted", "n/a",
             "ok" if not stats else "MISMATCH",
             "clean" if not stats else f"found {stats}"))
rows.append(("integers present are benign", "n/a", "ok",
             f"{ints} (enumeration, postcode, citation)"))
rows.append(("no date line", "n/a",
             "ok" if not re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d",
                                   TXT) else "MISMATCH", ""))

# --- strings that must agree with the manuscript ---------------------------
title = re.search(r"\\title\{([^}]*)\}", TEX).group(1)
title_plain = re.sub(r"\\\\|\s+", " ", title).strip()
chk("title matches \\title{}", title_plain.lower() in TXT.lower(),
    title_plain.lower() in TXT.lower(), title_plain[:52] + "...")

subj = re.search(r"\\IEEELSENSarticlesubject\{([^}]*)\}", TEX).group(1)
chk(f"subject '{subj}'", subj in TXT, subj in TXT)

orcid = re.search(r"orcidlink\{(0000-0003-3408-847X)\}", TEX)
chk("corresponding ORCID", bool(orcid) and "0000-0003-3408-847X" in TXT,
    bool(orcid) and "0000-0003-3408-847X" in TXT)

email = "nkanh@dut.udn.vn"
chk("corresponding e-mail", email in TXT and email in TEX,
    email in TXT and email in TEX)

# the companion paper must be disclosed and cited identically
chk("MST volume/article", "226203" in TXT and "226203" in TEX,
    "226203" in TXT and "226203" in TEX)
chk("discloses companion paper",
    "Meas. Sci." in TXT and "reference [1]" in TXT,
    "Meas. Sci." in TXT and "reference [1]" in TXT)

for label, s in [("PRONOSTIA named", "PRONOSTIA"),
                 ("Regular Letter", "Regular Letter"),
                 ("three outputs stated", "(3)"),
                 ("scope paragraph", "Sensor Signal Processing"),
                 ("originality declared", "not under consideration"),
                 ("conflict of interest", "conflict of interest"),
                 ("data availability", "publicly available")]:
    chk(label, s in TXT, s in TXT)

print(f"{'claim':<30}{'in letter':>10}{'':>3}{'verdict':<12}{'detail'}")
print("-" * 78)
for lab, pres, verd, det in rows:
    print(f"{lab:<30}{pres:>10}   {verd:<12}{det}")
print("-" * 78)
bad = [r for r in rows if r[2] != "ok"]
print(f"{len(rows)} checks, {len(bad)} flagged")
print(f"\nletter length: {len(TXT.split())} words")
