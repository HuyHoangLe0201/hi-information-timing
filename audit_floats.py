r"""Is every float referred to from the text, and do its panels line up?

Three of five figures were captioned and never mentioned in the body, so a
reader had no cue about when to look at them.  A float nothing points at is a
float that gets skipped, and most journals require every one to be cited.

The panel check asks whether the body cites a panel the caption does not
describe.

A sentence pointing at panel (d) of a three-panel figure is invisible to the
compiler and to every numerical audit, and is exactly the kind of thing that
survives a figure being redrawn.
"""
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
s = io.open(P, encoding="utf-8").read().replace("\r\n", "\n")

def brace_match(text, cmd):
    """The argument of `cmd`, counting braces.

    A regex with one level of nesting stops at the first \emph{...} inside a
    caption and silently returns a truncated string, which made this tool report
    a figure whose caption describes no panels when it describes three.
    """
    i = text.find(cmd + "{")
    if i < 0:
        return None
    i += len(cmd) + 1
    depth, j = 1, i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i:j - 1]


figs = re.findall(r"\\begin\{figure\}(.*?)\\end\{figure\}", s, re.S)
print("%-12s %-30s %-22s %s" % ("figure", "panels named in caption",
                                "label", "referred to in body"))
print("-" * 96)
for f in figs:
    g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", f)
    lab = re.search(r"\\label\{([^}]*)\}", f)
    cap = brace_match(f, "\\caption")
    name = g.group(1) if g else "?"
    label = lab.group(1) if lab else "?"
    incap = sorted(set(re.findall(r"\(([a-e])\)", cap or "")))
    inbody = sorted(set(re.findall(
        r"\\ref\{%s\}\(([a-e])\)" % re.escape(label), s)))
    flag = ""
    if inbody and not set(inbody) <= set(incap):
        flag = "  <-- body cites a panel the caption does not describe"
    print("%-12s %-30s %-22s %s%s"
          % (name, ",".join(incap) or "(none)", label,
             ",".join(inbody) or "(none)", flag))

# Do the figure files exist, and are they newer than nothing obvious?
print()
for f in figs:
    g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", f)
    if not g:
        continue
    p = os.path.join(os.path.dirname(P), g.group(1))
    print("  %-12s %s" % (g.group(1), "present" if os.path.exists(p)
                          else "MISSING"))


# --- is every float cited at all? --------------------------------------------
# Three of five figures were captioned and never mentioned, which neither the
# compiler nor any numerical audit can see.
print()
missing = []
for m in re.finditer(r"\\label\{((?:fig|tab):[^}]*)\}", s):
    lab = m.group(1)
    if not re.search(r"\\ref\{%s\}" % re.escape(lab), s):
        missing.append(lab)
print("floats never referred to from the text: %s"
      % (", ".join(missing) if missing else "none"))

# A layer that prints its verdict and exits zero cannot fail, and every
# runner that reads exit codes reports it as passing while it flags.
# audit_paper.py sat that way through a pass for sentence variety, with
# two checks reading sentences that had been rewritten around them.
import sys
sys.exit(1 if missing else 0)
