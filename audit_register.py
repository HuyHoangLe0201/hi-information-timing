r"""Find the sentences that still read as journalism rather than as a paper.

Reading a seventeen-thousand-word manuscript for register does not work: the
ear adjusts after two pages and stops hearing the habit.  These are the
constructions that carry the informal register, each searched for explicitly, so
the list is a work-list rather than an impression.

  opener      a sentence beginning with And, But, So, Now, Then, Yet
  fragment    a very short declarative used for emphasis
  clause_hd   a heading that is a question or a subordinate clause
  narrative   the authors appearing as characters in their own method
  rhetorical  a question asked in running text and answered by the author
  deictic     "That is ...", "This is ..." opening a sentence as a verdict
  metaphor    residual figurative verbs for what a number does
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "paper", "mssp.tex")
s = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")

# Work on the body only, and drop environments that are not running prose.
body = s.split("\\section{Introduction}")[-1]
body = body.split("\\begin{thebibliography}")[0]   # author initials are not prose
body = re.sub(r"\\begin\{(table|figure|equation|align|tabular)\*?\}.*?"
              r"\\end\{\1\*?\}", " ", body, flags=re.S)

heads = re.findall(r"\\(?:sub)?(?:section|paragraph)\*?\{([^}]*)\}", body)
prose = re.sub(r"\\(?:sub)?(?:section|paragraph)\*?\{[^}]*\}", " ", body)
prose = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", prose)
prose = re.sub(r"[{}]", " ", prose).replace("\n", " ")
prose = re.sub(r"\s+", " ", prose)

sent = [x.strip() for x in re.split(r"(?<=[.!?]) +", prose) if x.strip()]

FLAGS = []


def flag(kind, text):
    FLAGS.append((kind, text if len(text) < 118 else text[:115] + "..."))


for h in heads:
    if h.rstrip(".").endswith("?") or re.match(
            r"^(Why|How|What|When|Where|Whether|If|Is|Does|Do|Can) ", h):
        flag("clause_hd", h)

for x in sent:
    w = x.split()
    if not w:
        continue
    if re.match(r"^(And|But|So|Now|Then|Yet|Still|Indeed)\b", x):
        flag("opener", x)
    if (len(w) <= 6 and x.endswith(".")
            and "~" not in x and not re.search(r"[A-Za-z]:[a-z]", x)):
        flag("fragment", x)
    if re.search(r"\bwe (are|were|could not|cannot|do not|did not|report|take|"
                 r"say|call|know|found|find|leave|prefer|use)\b", x, re.I):
        flag("narrative", x)
    if "?" in x:
        flag("rhetorical", x)
    if re.match(r"^(That|This) (is|was|does|did|leaves|puts|makes|means)\b", x):
        flag("deictic", x)
    if re.search(r"\b(tells?|says?|wants?|cares?|knows?|admits?|refuses?|"
                 r"insists?|misses|catches|survives? every)\b", x):
        flag("metaphor", x)

order = ["clause_hd", "rhetorical", "fragment", "opener", "deictic",
         "narrative", "metaphor"]
print("%d sentences of running prose, %d headings\n" % (len(sent), len(heads)))
for k in order:
    hits = [t for kk, t in FLAGS if kk == k]
    if not hits:
        continue
    print("== %s (%d) ==" % (k, len(hits)))
    for t in hits:
        print("   " + t)
    print()
print("total flagged: %d" % len(FLAGS))
