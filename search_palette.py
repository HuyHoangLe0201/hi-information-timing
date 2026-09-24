"""
Snap-to-passing: search for a 3-hue categorical set (plus a neutral reference)
that clears every gate under the all-pairs test, instead of hand-picking one.

Candidate hues are drawn from published CVD-safe qualitative sets rather than
invented: Okabe-Ito, ColorBrewer Dark2/Set1, and Tol's bright scheme. The
objective is to maximise the worst pair distance across normal, protan and
deuteran vision, subject to the lightness band, the chroma floor and >= 3:1
contrast on white.
"""
from itertools import combinations
import numpy as np
from validate_palette import oklch, dE, simulate, contrast, SURFACE

CANDIDATES = {
    # Okabe-Ito
    "OI orange": "#E69F00", "OI sky": "#56B4E9", "OI green": "#009E73",
    "OI blue": "#0072B2", "OI vermillion": "#D55E00", "OI purple": "#CC79A7",
    # ColorBrewer Dark2
    "CB teal": "#1B9E77", "CB orange": "#D95F02", "CB violet": "#7570B3",
    "CB pink": "#E7298A", "CB green": "#66A61E", "CB gold": "#E6AB02",
    # ColorBrewer Set1
    "S1 red": "#E41A1C", "S1 blue": "#377EB8", "S1 green": "#4DAF4A",
    "S1 purple": "#984EA3", "S1 orange": "#FF7F00", "S1 brown": "#A65628",
    # Tol bright
    "T blue": "#4477AA", "T red": "#EE6677", "T green": "#228833",
    "T yellow": "#CCBB44", "T cyan": "#66CCEE", "T purple": "#AA3377",
}
GRAY = "#8C8C8C"          # the linear reference line, deliberately neutral


def legal(hx):
    L, C, _ = oklch(hx)
    return 0.43 <= L <= 0.77 and C >= 0.10 and contrast(hx, SURFACE) >= 3.0


pool = {k: v for k, v in CANDIDATES.items() if legal(v)}
print(f"{len(pool)} of {len(CANDIDATES)} candidates clear L-band, chroma and contrast")


def worst(colours):
    """Worst pair distance over normal + both CVD models, all pairs."""
    wn, wc = 1e9, 1e9
    for a, b in combinations(colours, 2):
        wn = min(wn, dE(a, b))
        for k in ("protanopia", "deuteranopia"):
            wc = min(wc, dE(simulate(a, k), simulate(b, k)))
    return wn, wc


best = []
for trio in combinations(pool.items(), 3):
    names = [t[0] for t in trio]
    cols = [t[1] for t in trio] + [GRAY]      # the gray must separate too
    wn, wc = worst(cols)
    if wn >= 15 and wc >= 8:
        best.append((min(wn / 15, wc / 8), wn, wc, names, [t[1] for t in trio]))

best.sort(reverse=True)
print(f"{len(best)} triples clear every gate against each other and the gray\n")
print(f"{'worst normal':>13}{'worst CVD':>11}   slots")
print("-" * 74)
for score, wn, wc, names, hexes in best[:12]:
    print(f"{wn:>13.1f}{wc:>11.1f}   " + ", ".join(f"{n} {h}" for n, h in zip(names, hexes)))

if best:
    _, wn, wc, names, hexes = best[0]
    print("\nbest by worst-pair margin:")
    for n, h in zip(names, hexes):
        L, C, _ = oklch(h)
        print(f"  {n:<15}{h}   L={L:.3f} C={C:.3f} contrast={contrast(h, SURFACE):.2f}:1")
    print(f"  {'gray ref':<15}{GRAY}")
    print(f"  worst normal dE {wn:.1f} (gate 15), worst CVD dE {wc:.1f} (target 8)")
