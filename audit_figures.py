"""
NOT PART OF THIS PAPER'S AUDIT SUITE.

This layer reads numbers back out of the RENDERED figures of the PREDECESSOR
study -- its colourbar, its beta_eff panel, its bar geometry -- and none of
those exists in this manuscript.  Repointing it at paper/mssp.tex was tried and
is wrong: the path was never the problem, the figures are simply different ones.
It is kept because the idea is worth reviving -- parsing a figure PDF catches a
plotting script that is self-consistent and still wrong, which no other layer
here can -- but it would have to be rewritten against the six figures this paper
actually has.  The same applies to count_abstract.py.

Run it and it fails on a missing file.  That is the correct behaviour: it is not
a check on this paper, and it should not look like one.
"""
"""
Figure audit -- read the numbers back OUT of the rendered figure PDFs and
compare them with the results files and with the text.

Trusting make_fig*.py would only prove the plotting script is self-consistent.
Instead this parses fig1.pdf / fig2.pdf directly: text labels via PyMuPDF, and
for the bar panel the actual rectangle geometry, calibrated against the axis
tick positions, so a bar of the wrong length is caught even if its printed
label is right.
"""
import os
import re
import json
import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LET = os.path.join(ROOT, "lsens_letter_v2")
HERE = os.path.dirname(os.path.abspath(__file__))
prof = json.load(open(os.path.join(HERE, "indicator_profiles.json")))
oof = json.load(open(os.path.join(HERE, "oof_results.json")))


def deitalic(s):
    """pgf sets math letters as Mathematical Italic code points; fold to ASCII
    so plain regexes work against the extracted text."""
    out = []
    for ch in s:
        o = ord(ch)
        if 0x1D44E <= o <= 0x1D467:          # italic a-z
            out.append(chr(ord('a') + o - 0x1D44E))
        elif 0x1D434 <= o <= 0x1D44D:        # italic A-Z
            out.append(chr(ord('A') + o - 0x1D434))
        else:
            out.append(ch)
    return "".join(out)


def words(pdf):
    d = fitz.open(pdf)
    p = d[0]
    return [(deitalic(w[4]), (w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
            for w in p.get_text("words")], p


# ------------------------------------------------------------------ fig 2
print("=== fig2.pdf: printed labels vs results files ===")
w2, page2 = words(os.path.join(LET, "fig2.pdf"))
txt2 = " ".join(t for t, _, _ in w2)
txt2 = re.sub(r"\s+", " ", txt2)

A2 = oof["configs"]["A2"]
checks = [
    ("panel (a) slope", rf"slope\s*{A2['slope']:.2f}", f"{A2['slope']:.2f}"),
    ("panel (a) r", rf"r\s*=\s*{A2['corr']:.2f}", f"{A2['corr']:.2f}"),
]
for name, pat, val in checks:
    ok = re.search(pat, txt2) is not None
    print(f"  {name:<22} expects {val:>6}   {'found' if ok else 'NOT FOUND in figure'}")

# The colourbar legend must name the variable actually mapped to colour; a stale
# label here silently mislabels 1500 points.
import numpy as _np
_cells = _np.load(os.path.join(HERE, "cells_A2.npz"))
_mapped = "tau0" if "tau0" in _cells.files else "w"
_says_age = bool(re.search(r"age", txt2)) and not re.search(r"window\s*w", txt2)
print(f"  colourbar variable     data={_mapped:<6}  label says "
      f"{'age' if _says_age else 'window'}   "
      f"{'ok' if (_mapped == 'tau0') == _says_age else 'MISLABELLED'}")

# beta_eff labels printed inside panel (c)
printed = sorted(float(x) for x in re.findall(r"eff\s*=\s*([0-9]+\.[0-9])", txt2))
expected = sorted(round(prof["indicators"][k]["beta_eff"], 1) for k in
                  ["rms", "peak", "kurtosis", "lf 0-2kHz", "mf 2-6kHz",
                   "hf 4-10kHz", "vhf 10-12.8kHz"])
print(f"\n  panel (c) beta_eff printed : {printed}")
print(f"  panel (c) beta_eff expected: {expected}")
print(f"  -> {'match' if printed == expected else 'MISMATCH'}")

# ---- bar geometry: recover tau_min from the drawn rectangles ----
print("\n=== fig2(c): bar LENGTHS re-measured from the vector drawing ===")
drawings = page2.get_drawings()
# panel (c) occupies the right third of the figure
xs_all = [d["rect"].x0 for d in drawings]
xmax = max(d["rect"].x1 for d in drawings)
panel_c_left = xmax * 0.62
cand = [d["rect"] for d in drawings
        if d.get("fill") and d["rect"].x0 > panel_c_left
        and d["rect"].width > 3 and 2 < d["rect"].height < 30]
# Real bars all start on the axis origin and share a height; anything else
# (legend swatches, annotation markers) is discarded by that signature.
if cand:
    from collections import Counter
    x0_mode = Counter(round(r.x0, 1) for r in cand).most_common(1)[0][0]
    h_mode = Counter(round(r.height, 1) for r in cand).most_common(1)[0][0]
    bars = [r for r in cand
            if abs(r.x0 - x0_mode) < 0.5 and abs(r.height - h_mode) < 0.5]
    print(f"  {len(cand)} filled rects in panel (c); {len(bars)} share the bar "
          f"signature (x0={x0_mode}, h={h_mode})")
else:
    bars = []
# calibrate with the tick labels 0.00 and 1.00 under panel (c)
ticks = {}
for t, cx, cy in w2:
    if t in ("0.00", "0.25", "0.50", "0.75", "1.00") and cx > panel_c_left:
        ticks[float(t)] = cx
if len(ticks) >= 2:
    lo_v, hi_v = min(ticks), max(ticks)
    lo_x, hi_x = ticks[lo_v], ticks[hi_v]
    scale = (hi_v - lo_v) / (hi_x - lo_x)
    bars.sort(key=lambda r: r.y0)
    # The panel sorts bars by tau_min, longest at the top; derive the expected
    # order the same way rather than hardcoding it, so a re-sort in the figure
    # cannot silently disagree with the audit.
    order = sorted(prof["indicators"],
                   key=lambda k: prof["indicators"][k]["tau_min_obs"],
                   reverse=True)
    print(f"  axis calibration from ticks {sorted(ticks)}: "
          f"{scale:.5f} data units per pt")
    print(f"  {'band':<17}{'drawn':>9}{'expected':>10}{'|diff|':>9}")
    print("  " + "-" * 45)
    worst = 0.0
    for r, key in zip(bars, order):
        drawn = lo_v + (r.x1 - lo_x) * scale
        exp = prof["indicators"][key]["tau_min_obs"]
        worst = max(worst, abs(drawn - exp))
        print(f"  {key:<17}{drawn:>9.3f}{exp:>10.3f}{abs(drawn-exp):>9.3f}")
    print("  " + "-" * 45)
    print(f"  worst discrepancy {worst:.3f} -> "
          f"{'bars match the data' if worst < 0.02 else 'BARS DISAGREE WITH DATA'}")
    print(f"  ({len(bars)} filled bars found, {len(order)} expected)")
else:
    print("  could not calibrate panel (c) axis")

# ------------------------------------------------------------------ fig 1
print("\n=== fig1.pdf: printed tick vs predicted tau_min ===")
w1, _ = words(os.path.join(LET, "fig1.pdf"))
txt1 = re.sub(r"\s+", " ", " ".join(t for t, _, _ in w1))
tau_pred = prof["tau_min_pow3"]
print(f"  predicted tau_min (beta=3, q*=0.35) = {tau_pred:.4f}")
print(f"  '0.81' highlighted on the axis      : "
      f"{'yes' if '0.81' in txt1 else 'NO -- check'}")
print(f"  legend lists all three classes      : "
      f"{'yes' if all(s in txt1 for s in ['Str.-exp.', 'Exponential', 'Power-law']) else 'NO'}")

# ------------------------------------------------------------------ text
print("\n=== numbers the TEXT quotes from the figures ===")
tex = open(os.path.join(LET, "lsens_letter.tex"), encoding="utf-8").read()
for label, pat, exp in [
    ("slope 0.97", r"log-log slope \$0\.97\$", A2["slope"]),
    ("r = 0.98", r"\$r=0\.98\$", A2["corr"]),
    ("beta_eff 0.74 to 30", r"\$0\.74\$ to \$30\$", None),
    ("tau_min 0.81 in text", r"\(q\^\\ast\)\^\{1/5\}=0\.81", None),
]:
    print(f"  {label:<22} in tex: {'yes' if re.search(pat, tex) else 'NOT FOUND'}")
