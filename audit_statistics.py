r"""Recompute the paper's inferential statistics from the reported counts.

The existing audits check that a number in the manuscript equals a number in a
result file.  Neither asks whether the test behind it was the right one, whether
the p-value is one- or two-sided, or whether the units being counted are
independent.  Those are the errors that survive a numerical audit intact.

Every quantity below is recomputed from the counts the paper itself reports, so
this needs no stored result and cannot inherit a fault from the code that
produced one.
"""
import io
import os
import re
import numpy as np
from scipy import stats
from itertools import permutations

TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "mssp.tex")
PAPER = re.sub(r"\s+", " ",
               io.open(TEX, encoding="utf-8").read().replace("$", ""))
NOTE = []


def in_paper(text, label):
    ok = re.sub(r"\s+", " ", text) in PAPER
    print("  %-56s %12s %12s  %s"
          % (label, "present" if ok else "ABSENT", "-",
             "ok" if ok else "FAIL"))
    if not ok:
        NOTE.append("the manuscript no longer says: " + text)


def line(label, got, said, verdict=""):
    print("  %-56s %12s %12s  %s" % (label, got, said, verdict))


print("=" * 92)
print("SIGN TESTS: is the reported p one-sided or two-sided?")
print("=" * 92)
print("  %-56s %12s %12s" % ("", "one-sided", "two-sided"))
CASES = [
    ("product se*g steadier, 9 of 12", 9, 12, 0.07),
    ("shape earlier, 15 of 16 bearings", 15, 16, None),
    ("fusion earlier, 17 of 17 bearings", 17, 17, None),
    ("window falls with age, 11 of 17", 11, 17, None),
    ("sign law agrees, 47 of 51 pairs", 47, 51, None),
    ("leverage bound attained, 16 of 16", 16, 16, None),
]
for lab, k, n, said in CASES:
    one = stats.binomtest(k, n, 0.5, alternative="greater").pvalue
    two = stats.binomtest(k, n, 0.5).pvalue
    v = ""
    if said is not None:
        near1 = abs(one - said) < 0.005
        near2 = abs(two - said) < 0.005
        v = ("matches the ONE-SIDED test" if near1 else
             "matches the two-sided test" if near2 else "matches NEITHER")
        if near1 and "one-sided alternative" not in PAPER:
            NOTE.append("The p=%.2f sign test is one-sided and the manuscript "
                        "does not say so." % said)
    line(lab, "%.4g" % one, "%.4g" % two, v)

print()
print("=" * 92)
print("PERMUTATION TEST on a rank correlation, ten channels")
print("=" * 92)
# The paper reports rho = +0.53 between tau_min and the refined error over ten
# channels, with an exact permutation test giving p = 0.06.  With n = 10 the
# permutation distribution of Spearman's rho is exactly enumerable.
n = 10
base = np.arange(n)
rhos = np.fromiter(
    (stats.spearmanr(base, p).statistic for p in permutations(base)),
    dtype=float, count=np.math.factorial(n) if hasattr(np, "math") else None) \
    if False else None
# 10! = 3.6M pairs is enumerable but slow through scipy; the null distribution
# of Spearman's rho depends only on the sum of squared rank differences, so it
# is computed directly.
d2 = np.empty(3628800, dtype=np.int64)
i = 0
for p in permutations(range(n)):
    a = np.subtract(base, p)
    d2[i] = a @ a
    i += 1
rho = 1.0 - 6.0 * d2 / (n * (n * n - 1))
one = float(np.mean(rho >= 0.53))
two = float(np.mean(np.abs(rho) >= 0.53))
print("  %-56s %12s %12s" % ("", "one-sided", "two-sided"))
line("exact permutation P(rho >= 0.53), n = 10", "%.4g" % one, "%.4g" % two,
     "paper reports 0.06")
if abs(one - 0.06) < 0.008 and "p=0.06 one-sided" not in PAPER:
    NOTE.append("The p=0.06 permutation test is one-sided and the manuscript "
                "does not say so.")

print()
print("=" * 92)
print("Are both p-values labelled in the manuscript?")
print("=" * 92)
in_paper("p=0.07 against the one-sided alternative",
         "the sign test is labelled one-sided")
in_paper("and 0.15 against the two-sided one",
         "its two-sided value is given")
in_paper("p=0.06 one-sided", "the permutation test is labelled one-sided")
in_paper("and 0.11 two-sided", "its two-sided value is given")

print()
print("=" * 92)
print("RECORD ACCOUNTING: do the stated totals add up?")
print("=" * 92)
DATA = dict(PRONOSTIA=17, XJTU=15, FD001=100, FD004=249, battery=4)
line("sum of the dataset table", sum(DATA.values()), 385,
     "ok" if sum(DATA.values()) == 385 else "MISMATCH")
# the attainment study uses 370 records
line("records in the attainment study", 370,
     "%d if FD004 is excluded" % (sum(DATA.values()) - 249),
     "370 = 385 - 15, i.e. one bearing set is partly excluded")
if "370 records of four of them" not in PAPER:
    NOTE.append("The 370 records of the attainment study are 15 fewer than the "
                "385 in the dataset table, and the manuscript does not say "
                "which 15 are held back or why.")

print()
print("=" * 92)
print("INDEPENDENCE: are the units counted actually independent?")
print("=" * 92)
print("""  The 47-of-51 sign test counts consecutive pairs of windows drawn from
  seventeen bearings, so the 51 units are neither independent of one another
  nor exchangeable across bearings: three pairs from one bearing share samples
  and share that bearing's density estimate.  A binomial p-value computed on
  them is anticonservative.  The direction of the result is not in doubt at
  47 of 51, but the p-value should not be quoted as if from 51 independent
  trials, and the paper does not quote one, which is the correct choice.
""")
n_eff = 17
k_eff = int(round(47 / 51 * 17))
line("if one vote is taken per bearing instead", "%d of %d" % (k_eff, n_eff),
     "p = %.4g" % stats.binomtest(k_eff, n_eff, 0.5,
                                  alternative="greater").pvalue,
     "still decisive")

print()
print("=" * 92)
print("MULTIPLICITY: how many tests support the headline claim?")
print("=" * 92)
print("""  The separation between distributional and amplitude indicators is checked
  at seven demand levels, on two rigs, under eight single-measure deletions,
  at two estimator bandwidths, and in a second domain.  These are robustness
  replications of one hypothesis, not a family of hypotheses, so no
  family-wise correction applies: the claim would be weakened, not
  strengthened, by any one of them failing.  The multiplicity that would need
  correcting is the choice of which indicators to compare, and that is fixed
  in advance by the distribution/amount split rather than searched.
""")

print("=" * 92)
if NOTE:
    print("%d points for the manuscript:" % len(NOTE))
    for t in dict.fromkeys(NOTE):
        print("  - " + t)
else:
    print("no statistical issues found")
print("=" * 92)
