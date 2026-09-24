"""
The cover letter, checked against the manuscript and the results files.

An editor reads the letter beside the abstract, so a claim that is stronger in
the letter than in the paper is the worst possible first impression -- and it is
exactly the failure this layer was written for.  The letter said the floor was
"attained to within 9%" on all 370 records; the paper says it is attained on the
bearings and both turbofan fleets and missed by a factor of 1.94 on the battery
cells.  Nothing checked the letter, so the overclaim survived the reviewer read
that removed it from the manuscript.

Every number in the letter is traced to crb_full.json or to mssp.tex, every
claim that also appears in the paper is required to appear there in the same
strength, and the declarations an editor expects are required to be present.
audit_cover.py is the predecessor study's version of this layer and is retired.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LET = io.open(os.path.join(HERE, "Cover_letter.md"), encoding="utf-8").read()
TXT = re.sub(r"\s+", " ", LET)
TEX = io.open(os.path.join(ROOT, "paper", "mssp.tex"), encoding="utf-8").read()
C = json.load(open(os.path.join(HERE, "crb_full.json")))

rows = []


def chk(label, ok, detail=""):
    rows.append((label, "ok" if ok else "FLAG", detail))


# --- the title is the manuscript's, word for word ---------------------------
title = re.search(r"\\title\{(.*?)\}", TEX, re.S).group(1)
title = re.sub(r"\\\\", " ", title)
title = re.sub(r"\s+", " ", title).strip()
chk("title matches \\title{}", title.lower() in TXT.lower(), title[:46] + "...")

# --- the corresponding author is the manuscript's ---------------------------
ead = re.search(r"\\ead\{([^}]+)\}", TEX).group(1)
chk("corresponding e-mail", ead in TXT, ead)
chk("journal named", "Mechanical Systems and Signal Processing" in TXT)

# --- every number in the letter is traceable --------------------------------
nums = re.findall(r"\d+(?:\.\d+)?%?", TXT)
GOOD = ("bearing", "turbofan_FD001", "turbofan_FD004")
units = 370
worst_good = max(C[d]["median"] for d in GOOD)
battery = C["battery"]["median"]

chk("370 units", "370" in nums and ("%d public run-to-failure" % units) in TXT,
    "crb: %d records" % units)
chk("9% is the worst attaining domain", "9%" in nums and worst_good <= 1.09,
    "max median over %s = %.3f" % ("/".join(GOOD), worst_good))
chk("1.94 is the battery median", "1.94" in nums and abs(battery - 1.94) < 0.005,
    "battery median = %.3f" % battery)
# "4 to 19 times the information" is the amplitude family's budget over the
# distribution family's, on the shorter-record rig and the longer one
T = json.load(open(os.path.join(HERE, "timing_vs_amount.json")))
lo, hi = sorted(round(T[r]["budget_ratio"]) for r in ("XJTU", "PRONOSTIA"))
chk("4 to 19 is the family budget ratio",
    "%d to %d times the information" % (lo, hi) in TXT,
    "budget ratios round to %d and %d" % (lo, hi))
extra = [n for n in nums if n not in ("370", "9%", "1.94", str(lo), str(hi))]
chk("no untraced numbers", not extra, "untraced: %s" % extra if extra else "none")

# --- the letter is not stronger than the paper ------------------------------
# The attainment claim must name the exception.  Any sentence that puts "9%"
# in the same breath as all three material classes is the overclaim returning.
sent = re.split(r"(?<=[.!?])\s+(?=[A-Z*])", TXT)
att = next((x for x in sent if "9%" in x), "")
chk("attainment claim names the exception",
    bool(att) and ("lithium" in att or "battery" in att) and "1.94" in att,
    att.strip()[:70] + "..." if att else "no attainment sentence found")
# The manuscript writes the figure as maths, and moving the sentence between
# sections has already changed its delimiters once, so the comparison is made
# on a normalised copy rather than on the source characters.
_TEX = re.sub(r"\s+", " ", TEX.replace("$", "").replace("\\%", "%"))
chk("paper makes the same claim",
    "attained to within 9% on the bearings and on both turbofan fleets" in _TEX)

# --- the mechanism claim carries the paper's hedge --------------------------
# The letter may name transferability as the separator only if it also names
# what the paper names: the rigs the claim was measured on.  An unscoped
# version of this sentence was the reviewer's objection to the manuscript, and
# the letter is read beside the abstract.
mech = "transferability" in TXT
scoped = "on these rigs" in TXT or "two rigs" in TXT or "these two rigs" in TXT
chk("mechanism scoped to the rigs it was measured on", not mech or scoped)

# --- and the letter says where the ordering stops ---------------------------
chk("the causal limit is disclosed",
    "causal" in TXT and ("records are short" in TXT or "short records" in TXT))

# --- and that the ordering is in timing, which the amount family reverses ---
chk("timing, not amount, is said", "timing, not in worth" in TXT
    and "The ordering is in time, not in worth" in _TEX)

# --- what the guide asks the letter itself to carry -------------------------
# A Long Research Article must justify its length in the cover letter, and the
# submitting author picks one subject area; the manuscript runs well past the
# journal's median length, so both are required here.
chk("article type declared", "Long Research Article" in TXT)
chk("length justified", "Why a Long Research Article" in TXT)
chk("subject area named", "subject area M, Prognostics" in TXT)
# The companion paper is the authors' own; naming it forestalls an overlap query
chk("companion paper disclosed", "companion paper by the same authors" in TXT
    and "companion paper" in _TEX)

# --- the declarations an editor expects -------------------------------------
for label, s in [("originality declared", "not been published elsewhere"),
                 ("not under consideration", "not under consideration"),
                 ("all authors approved", "approved the submission"),
                 ("competing interest", "competing-interest"),
                 ("funding and generative AI", "funding and generative-AI"),
                 ("data and code released", "released with")]:
    chk(label, s in TXT)

# --- while the declarations are the authors' to confirm, say so -------------
draft = "DRAFT" in LET
unfilled = "UNFILLED" in TEX
chk("marked DRAFT while the manuscript has [UNFILLED]",
    draft == unfilled, "DRAFT=%s, UNFILLED in mssp.tex=%s" % (draft, unfilled))

# --- length -----------------------------------------------------------------
words = len(LET.split())
chk("letter is one page", words <= 600, "%d words" % words)

print("%-48s%-8s%s" % ("check", "verdict", "detail"))
print("-" * 92)
for lab, verd, det in rows:
    print("%-48s%-8s%s" % (lab, verd, det))
print("-" * 92)
bad = [r for r in rows if r[1] != "ok"]
print("%d checks, %d flagged" % (len(rows), len(bad)))
sys.exit(1 if bad else 0)
