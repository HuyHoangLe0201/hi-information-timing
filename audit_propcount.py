r"""The sixteenth audit: does the paper count its own propositions correctly?

The omitted-proofs appendix opens by saying how many propositions the paper has,
how many are proved there, and how many carry their derivation inside the
statement.  Those three numbers were right when they were written and are the
first thing a split breaks: moving one proposition to the supplement changes all
three, and no other layer can see it, because audit_paper reads the manuscript
and the supplement together and the sentence is about the manuscript alone.

WHAT IS CHECKED.  The propositions are counted in paper/mssp.tex;
the sentence that reports the counts is in the supplement, which is where the
long proofs now live:

  (TOTAL)   the number of \begin{proposition} environments;
  (THERE)   how many distinct propositions the supplement's proofs section
            proves,
            counting a proof that covers two of them as two;
  (INLINE)  how many are followed immediately by their own proof;
  (REST)    TOTAL - THERE - INLINE, the ones whose derivation is in the
            statement.

Each is compared with the number the appendix's opening sentence prints, and
the three must also add up.  The words are spelled out in the sentence, so the
comparison is against a spelled-out form rather than a digit.
"""
import io
import os
import re
import sys

BS = chr(92)
CL = "[" + BS + BS + "]"

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
SUPP = os.path.join(os.path.dirname(HERE), "paper", "supplement.tex")

WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
        7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
        12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
        16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
        20: "twenty", 21: "twenty-one", 22: "twenty-two",
        23: "twenty-three", 24: "twenty-four", 25: "twenty-five",
        26: "twenty-six", 27: "twenty-seven", 28: "twenty-eight",
        29: "twenty-nine", 30: "thirty", 31: "thirty-one",
        32: "thirty-two", 33: "thirty-three", 34: "thirty-four"}

# The propositions are counted in the manuscript; the long proofs now live in
# the supplement, so the appendix side of the comparison is read from there.
body = re.sub(r"\s+", " ", re.sub("(?m)^%.*$", "",
              io.open(MAIN, encoding="utf-8").read()))
supp = re.sub(r"\s+", " ", re.sub("(?m)^%.*$", "",
              io.open(SUPP, encoding="utf-8").read()))
# found by its label: the title now also names the results the body states in
# words, and a title is the first thing a reader asks to have reworded
cut = supp.find(BS + "label{sec:proofs}")
assert cut > 0, "the proofs section was not found in the supplement"
app = supp[cut:]

total = len(re.findall(CL + "begin{proposition}", body))
inline = len(re.findall(CL + "end{proposition} " + CL + "begin{proof}", body))

there = set()
# The literal square brackets of the optional argument have to be escaped as
# "\[" and "\]": writing them bare turned "[Proof of ..." into a character
# class and the pattern would not compile at all.
_PROOFHEAD = CL + r"begin{proof}\[Proof of ([^\]]*)\]"
for head in re.findall(_PROOFHEAD, app):
    there.update(re.findall(r"prop:[A-Za-z0-9]+", head))
rest = total - len(there) - inline

# The three numbers the sentence prints.
m_total = re.search(r"of the ([a-z-]+) propositions", app)
m_there = re.search(r"([A-Z][a-z-]+) of the [a-z-]+ propositions", app)
m_rest = re.search(r"the ([a-z-]+) that remain", app)
m_inline = re.search(r"([A-Z][a-z-]+) are proved in the main paper "
                     r"immediately after their statement", app)

rows = [
    ("propositions in the manuscript", total,
     m_total.group(1) if m_total else "NOT STATED"),
    ("proved in the omitted-proofs appendix", len(there),
     m_there.group(1).lower() if m_there else "NOT STATED"),
    ("carrying the derivation in the statement", rest,
     m_rest.group(1) if m_rest else "NOT STATED"),
]

print("=" * 92)
print("Does the paper count its own propositions correctly?")
print("=" * 92)
print()
print("  %-44s %8s  %-14s %s" % ("", "counted", "printed", ""))
bad = 0
for label, n, printed in rows:
    ok = printed == WORD.get(n, str(n))
    bad += not ok
    print("  %-44s %8d  %-14s %s"
          % (label, n, printed, "ok" if ok else "<-- MISMATCH, should be "
             + WORD.get(n, str(n))))
_pin = m_inline.group(1).lower() if m_inline else "NOT STATED"
_okin = _pin == WORD.get(inline, str(inline))
bad += not _okin
print("  %-44s %8d  %-14s %s"
      % ("proved beside the statement", inline, _pin,
         "ok" if _okin else "<-- MISMATCH, should be "
         + WORD.get(inline, str(inline))))

adds = len(there) + inline + rest == total
print("\n  the three classes account for every proposition"
      if adds else "\n  THE THREE CLASSES DO NOT ADD UP")
print("=" * 92)
ok = not bad and adds
print("the paper's count of its own propositions is right" if ok
      else "the paper miscounts its own propositions")
print("=" * 92)
sys.exit(0 if ok else 1)
