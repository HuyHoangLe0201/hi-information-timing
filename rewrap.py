r"""Rewrap prose paragraphs to eighty columns.

Successive edits leave the source ragged: a replaced sentence sits on a short
line beside a full one, and a later reader cannot tell an intentional break from
an accident.  LaTeX ignores all of it, which is exactly why it needs doing by
tool rather than by eye.

Only paragraphs that are unambiguously prose are touched.  A block is skipped if
it sits inside any environment other than the few that contain running text, or
if any of its lines carries a construct where a line break is not free: a table
cell, a row terminator, an unescaped comment character, a list item, or a
preamble declaration.  The test that this was safe is that the compiled page
count and all three numerical audits are unchanged afterwards.
"""
import io
import os
import re
import textwrap

P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "paper", "mssp.tex")
raw = io.open(P, encoding="utf-8", newline="").read()
CRLF = "\r\n" in raw
lines = raw.replace("\r\n", "\n").split("\n")

# Environments whose body is running text and may be rewrapped.
PROSE_ENV = {"document", "abstract", "frontmatter", "proof",
             "proposition", "remark", "keyword"}
# Anything on a line that makes a break unsafe or meaningless.
UNSAFE = re.compile(r"(?<!\\)%|&|\\\\|\\item|\\begin|\\end|\\bibitem|\\caption"
                    r"|\\includegraphics|\\label|\\documentclass|\\usepackage"
                    r"|\\newcommand|\\title|\\author|\\ead|\\affiliation"
                    r"|\\cortext|\\journal|\\newtheorem|\\theoremstyle"
                    r"|\\AtBeginDocument|\\section|\\subsection|\\bibliographystyle")

out, buf, stack, changed = [], [], [], 0


def flush():
    global changed
    if not buf:
        return
    ok = (all(e in PROSE_ENV for e in stack)
          and not any(UNSAFE.search(ln) for ln in buf))
    if ok:
        para = " ".join(ln.strip() for ln in buf)
        new = textwrap.wrap(para, width=80, break_long_words=False,
                            break_on_hyphens=False)
        if new != buf:
            changed += 1
        out.extend(new)
    else:
        out.extend(buf)
    del buf[:]


for ln in lines:
    b = re.match(r"\s*\\begin\{([^}]*)\}", ln)
    e = re.match(r"\s*\\end\{([^}]*)\}", ln)
    if ln.strip() == "":
        flush()
        out.append(ln)
        continue
    if b or e:
        flush()
        out.append(ln)
        if b:
            stack.append(b.group(1))
        elif stack and stack[-1] == e.group(1):
            stack.pop()
        continue
    buf.append(ln)
flush()

s = "\n".join(out)
if CRLF:
    s = s.replace("\n", "\r\n")
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("rewrapped %d paragraphs" % changed)
