"""
Where is the space going, and where is it cheapest to reclaim?

Reports words per section, then the longest sentences in the body -- long
sentences are where tightening buys the most lines for the least content.
Also measures how much of the last page is actually free.
"""
import os
import re
import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "lsens_letter_v2", "lsens_letter.tex")
PDF = os.path.join(ROOT, "lsens_letter_v2", "lsens_letter.pdf")
raw = open(TEX, encoding="utf-8").read()

body = raw.split(r"\begin{abstract}")[1].split(r"\begin{thebibliography}")[0]
body = re.sub(r"(?m)^\s*%.*$", "", body)
body = re.sub(r"(?<!\\)%.*", "", body)


def clean(s):
    s = re.sub(r"\$[^$]*\$", " M ", s)
    s = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " M ", s, flags=re.S)
    s = re.sub(r"\\(?:eq)?ref\{[^}]*\}", " R ", s)
    s = re.sub(r"\\cite(?:\[[^\]]*\])?\{[^}]*\}", " C ", s)
    s = re.sub(r"\\(?:emph|textit|textbf)\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    return re.sub(r"\s+", " ", s.replace("{", " ").replace("}", " ")).strip()


# ---- split into labelled blocks -------------------------------------------
blocks = []
cur = ["abstract", ""]
for line in body.splitlines():
    m = re.match(r"\s*\\section\{([^}]*)\}", line)
    if m:
        blocks.append(cur); cur = [m.group(1), ""]
        continue
    m = re.match(r"\s*\\begin\{(remark|proposition|theorem|assumption)\}(\[[^\]]*\])?", line)
    if m:
        blocks.append(cur)
        lab = m.group(1) + (m.group(2) or "")
        cur = [f"  .{lab[:42]}", ""]
        continue
    if re.match(r"\s*\\end\{(remark|proposition|theorem|assumption)\}", line):
        blocks.append(cur); cur = [cur[0].replace("  .", "  (cont) "), ""]
        continue
    if re.match(r"\s*\\begin\{(table|figure\*?)\}", line):
        blocks.append(cur); cur = ["  [float]", ""]
        continue
    cur[1] += " " + line
blocks.append(cur)

print(f"{'block':<50}{'words':>7}")
print("-" * 58)
tot = 0
for name, txt in blocks:
    n = len(clean(txt).split())
    if n < 3:
        continue
    tot += n
    print(f"{name[:50]:<50}{n:>7}")
print("-" * 58)
print(f"{'TOTAL body words':<50}{tot:>7}")

# ---- longest sentences ----------------------------------------------------
flat = clean(body)
sents = [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", flat) if s.strip()]
sents.sort(key=lambda s: -len(s.split()))
print(f"\n=== 10 longest sentences (tightening targets) ===")
for s in sents[:10]:
    print(f"\n  {len(s.split()):>3}w  {s[:190]}")

# ---- free space on the last page -----------------------------------------
d = fitz.open(PDF)
last = d[len(d) - 1]
blocks_pdf = [b for b in last.get_text("blocks") if b[4].strip()]
lowest = max(b[3] for b in blocks_pdf)
h = last.rect.height
print(f"\n=== last page ===")
print(f"  page height {h:.0f} pt, lowest text at {lowest:.0f} pt")
print(f"  free at bottom: {h - lowest:.0f} pt "
      f"({(h - lowest) / h * 100:.0f}% of page height)")
print(f"  a body line is ~10.5 pt, so that is ~{(h - lowest) / 10.5:.0f} free lines")
print(f"  NOTE: a two-column page holds ~2 x 47 = 94 lines, so one full page")
print(f"  of slack would need ~94 lines freed.")
