r"""Measure what makes the prose hard to follow, sentence by sentence.

Three things obstruct a reader here and all three are countable.  The em dash
lets a writer postpone deciding whether an aside is a parenthesis, a clause or
its own sentence, so a paper accumulates them wherever the argument was hard.
Sentence length past about thirty-five words stops being a style question and
becomes a memory one.  And a sentence carrying several subordinate clauses can
be short and still unreadable, so clause count is measured separately from
length.

The output is a work-list ordered by cost, not a score.
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "paper", "mssp.tex")
s = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")

body = s.split("\\section{Introduction}")[-1].split("\\begin{thebibliography}")[0]
# Keep the em dashes, drop everything that is not running prose.
clean = re.sub(r"\\begin\{(table|figure|equation|align|tabular)\*?\}.*?"
               r"\\end\{\1\*?\}", " ", body, flags=re.S)
clean = re.sub(r"\$[^$]*\$", " X ", clean)          # math is one token
# An environment boundary ends a sentence.  Without this the last sentence of a
# paragraph runs into the first line of the proposition that follows it, and the
# merged pair is reported as one unreadable ninety-word sentence that nobody
# wrote.
clean = re.sub(r"\\(?:begin|end)\{[^}]*\}", ". ", clean)
clean = re.sub(r"\\(?:sub)?(?:section|paragraph)\*?\{[^}]*\}", ". ", clean)
clean = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", clean)
clean = re.sub(r"[{}]", " ", clean)
clean = re.sub(r"[ \t\n]+", " ", clean)

sent = [x.strip() for x in re.split(r"(?<=[.!?]) +(?=[A-Z(])", clean)
        if len(x.strip()) > 3]

DASH = clean.count("---")
print("em dashes in running prose : %d" % DASH)
print("also in headings and floats: %d"
      % (s.count("---") - DASH))
print()

lens = [len(x.split()) for x in sent]
lens.sort()
n = len(lens)
print("%d sentences" % n)
print("  median %d words, mean %.1f" % (lens[n // 2], sum(lens) / n))
for cut in (30, 40, 50, 60):
    k = sum(1 for L in lens if L > cut)
    print("  longer than %2d words: %3d  (%.0f%%)" % (cut, k, 100.0 * k / n))
print()

# Subordination: commas plus the usual subordinators, as a proxy for how many
# things must be held open at once.
SUB = re.compile(r"\b(which|that|because|since|so that|whereas|while|although|"
                 r"though|where|when|if|unless|rather than|as|and|but)\b", re.I)


def load(x):
    return x.count(",") + len(SUB.findall(x)) + x.count(";") + x.count("---")


worst = sorted(sent, key=lambda x: -(len(x.split()) + 3 * load(x)))
print("the twenty-five heaviest sentences (length + 3x subordination):\n")
for i, x in enumerate(worst[:25], 1):
    print("%2d. [%d words, %d joins, %d dash] %s"
          % (i, len(x.split()), load(x), x.count("---"),
             x if len(x) < 200 else x[:197] + "..."))
    print()

many = [x for x in sent if x.count("---") >= 2]
print("sentences carrying a dashed aside (%d):\n" % len(many))
for x in many[:20]:
    print("   " + (x if len(x) < 170 else x[:167] + "..."))
