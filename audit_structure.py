r"""Structural faults that are visible in the outline but not in the prose.

Five are checked, because each misleads a reader in a different way.

  no preamble    a section that opens straight into its first subsection gives
                 no orientation; the reader meets a detail before a purpose
  lone child     a section with exactly one subsection: either the subsection
                 is not a division at all, or a sibling is missing
  runt           a section too small to be one, which a reader will look for in
                 the contents and find as a heading over a paragraph
  giant          a subsection long enough to need its own divisions
  forward ref    a claim that depends on a section not yet read

Nothing here is automatically wrong.  The output is a list of places to make a
decision, not a list of errors.
"""
import io
import os
import re

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "paper", "mssp.tex")
s = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")
s = s.split("\\begin{thebibliography}")[0]

HEAD = re.compile(r"\\(sub)?section\*?\{([^}]*)\}")
hits = [(m.start(), bool(m.group(1)), m.group(2)) for m in HEAD.finditer(s)]


def words(a, b):
    seg = re.sub(r"\\begin\{(table|figure|tabular)\*?\}.*?\\end\{\1\*?\}", " ",
                 s[a:b], flags=re.S)
    seg = re.sub(r"\\[a-zA-Z]+\*?", " ", seg)
    seg = re.sub(r"[{}$\\]", " ", seg)
    return len([t for t in seg.split() if any(c.isalnum() for c in t)])


# section index -> (name, own text before first child, children)
secs = []
for i, (pos, sub, name) in enumerate(hits):
    if sub:
        continue
    end = next((h[0] for h in hits[i + 1:] if not h[1]), len(s))
    kids = [(h[0], h[2]) for h in hits[i + 1:] if h[0] < end and h[1]]
    pre_end = kids[0][0] if kids else end
    secs.append(dict(name=name, pos=pos, end=end, kids=kids,
                     total=words(pos, end), preamble=words(pos, pre_end)))

print("%-52s %6s %6s %4s" % ("section", "words", "preamb", "subs"))
print("-" * 72)
for d in secs:
    print("%-52s %6d %6d %4d"
          % (d["name"][:52], d["total"], d["preamble"], len(d["kids"])))
print()

flags = []
for d in secs:
    if d["kids"] and d["preamble"] < 60:
        flags.append(("no preamble", "%s: %d words before its first subsection"
                      % (d["name"], d["preamble"])))
    if len(d["kids"]) == 1:
        flags.append(("lone child", "%s has exactly one subsection (%s)"
                      % (d["name"], d["kids"][0][1])))
    if d["total"] < 250 and not d["kids"]:
        flags.append(("runt", "%s is %d words" % (d["name"], d["total"])))
    for j, (kp, kn) in enumerate(d["kids"]):
        ke = d["kids"][j + 1][0] if j + 1 < len(d["kids"]) else d["end"]
        kw = words(kp, ke)
        if kw > 850:
            flags.append(("giant", "%s / %s is %d words" % (d["name"], kn, kw)))

# Forward references: a \ref to a label defined later in the file.
label_at = {m.group(1): m.start()
            for m in re.finditer(r"\\label\{(sec:[^}]*)\}", s)}
fwd = {}
for m in re.finditer(r"\\ref\{(sec:[^}]*)\}", s):
    tgt = label_at.get(m.group(1))
    if tgt is not None and tgt > m.start():
        fwd[m.group(1)] = fwd.get(m.group(1), 0) + 1
for k, v in sorted(fwd.items(), key=lambda kv: -kv[1]):
    if v >= 3:
        flags.append(("forward ref", "%s is cited %d times before it appears"
                      % (k, v)))

for kind in ("no preamble", "lone child", "runt", "giant", "forward ref"):
    hits2 = [t for k, t in flags if k == kind]
    if hits2:
        print("== %s ==" % kind)
        for t in hits2:
            print("   " + t)
        print()
print("%d structural points to decide" % len(flags))
