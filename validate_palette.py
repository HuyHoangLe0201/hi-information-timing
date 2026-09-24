"""
The six categorical checks, computed rather than eyeballed.

Node is not available here, so this is a Python implementation of the same
standard: OKLab/OKLCH per Ottosson, colour-vision deficiency simulated with
Machado-Oliveira-Fernandes 2009 at severity 1.0 (the model the thresholds are
calibrated to), Euclidean distance in OKLab x100.

Gates, light mode:
  L band      0.43 - 0.77
  chroma      >= 0.10
  CVD dE      >= 8 target, >= 6 floor (floor legal only with secondary encoding)
  normal dE   >= 15  (hard gate)
  contrast    >= 3:1 vs surface
Sequential ramps are NOT judged by these; for those the check is monotone
lightness, which is run separately at the bottom.
"""
import numpy as np

SURFACE = "#ffffff"


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def oklab(hexc):
    lin = srgb2lin(hex2rgb(hexc))
    lms = np.cbrt(M1 @ lin)
    return M2 @ lms


def oklch(hexc):
    L, a, b = oklab(hexc)
    return L, float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360)


# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB
CVD = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
}


def simulate(hexc, kind):
    lin = srgb2lin(hex2rgb(hexc))
    out = lin2srgb(CVD[kind] @ lin)
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in out)


def dE(h1, h2):
    return float(np.linalg.norm((oklab(h1) - oklab(h2)) * 100))


def relL(hexc):
    r, g, b = srgb2lin(hex2rgb(hexc))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(h1, h2):
    a, b = sorted([relL(h1), relL(h2)], reverse=True)
    return (a + 0.05) / (b + 0.05)


# --------------------------------------------------------------- the palette
PAL = [("Exponential", "#1F4E79"),
       ("Power-law",   "#B7302B"),
       ("Str.-exp.",   "#127A6E"),
       ("Linear ref.", "#8C8C8C")]


def _report():
    print("=== checks 2, 3, 5: per colour ===")
    print(f"{'slot':<14}{'hex':<10}{'L':>7}{'C':>7}{'band':>7}{'chroma':>8}"
          f"{'contrast':>10}{'':>3}")
    print("-" * 66)
    fails = []
    for name, hx in PAL:
        L, C, _ = oklch(hx)
        cr = contrast(hx, SURFACE)
        band = "ok" if 0.43 <= L <= 0.77 else "FAIL"
        chrom = "ok" if C >= 0.10 else ("gray" if name.endswith("ref.") else "FAIL")
        con = "ok" if cr >= 3.0 else "WARN"
        if "FAIL" in (band, chrom) or con == "WARN":
            fails.append(name)
        print(f"{name:<14}{hx:<10}{L:>7.3f}{C:>7.3f}{band:>7}{chrom:>8}"
              f"{cr:>9.2f}:1 {con}")

    print("\n=== check 4: pair separation (OKLab x100) ===")
    print("  all-pairs is used: fig1 draws four curves that cross, so any two can be")
    print("  adjacent on the page.")
    names = [n for n, _ in PAL]
    hexes = [h for _, h in PAL]
    print(f"\n{'pair':<28}{'normal':>9}{'protan':>9}{'deuter':>9}   verdict")
    print("-" * 68)
    worst_n = worst_c = 1e9
    for i in range(len(PAL)):
        for j in range(i + 1, len(PAL)):
            n1, n2 = names[i], names[j]
            h1, h2 = hexes[i], hexes[j]
            dn = dE(h1, h2)
            dp = dE(simulate(h1, "protanopia"), simulate(h2, "protanopia"))
            dd = dE(simulate(h1, "deuteranopia"), simulate(h2, "deuteranopia"))
            worst_n = min(worst_n, dn)
            worst_c = min(worst_c, dp, dd)
            v = "ok" if min(dp, dd) >= 8 and dn >= 15 else (
                "floor (needs 2nd encoding)" if min(dp, dd) >= 6 and dn >= 15 else "FAIL")
            print(f"{n1+' / '+n2:<28}{dn:>9.1f}{dp:>9.1f}{dd:>9.1f}   {v}")
    print("-" * 68)
    print(f"worst normal-vision dE {worst_n:.1f}  (hard gate 15)")
    print(f"worst CVD dE           {worst_c:.1f}  (target 8, floor 6)")

    print("\n=== sequential ramp for window width w in fig2(a) ===")
    print("judged on monotone lightness, not the categorical checks")


    def ramp_report(name, hexes):
        Ls = [oklch(h)[0] for h in hexes]
        mono = all(b > a for a, b in zip(Ls, Ls[1:])) or all(b < a for a, b in zip(Ls, Ls[1:]))
        dL = [abs(b - a) for a, b in zip(Ls, Ls[1:])]
        hues = [oklch(h)[2] for h in hexes]
        span = max(hues) - min(hues)
        light_cr = max(contrast(hexes[0], SURFACE), contrast(hexes[-1], SURFACE))
        print(f"  {name:<12} L {Ls[0]:.2f}->{Ls[-1]:.2f}  monotone={mono}  "
              f"min dL={min(dL):.3f}  hue span={span:.0f} deg  "
              f"light-end contrast={min(contrast(hexes[0], SURFACE), contrast(hexes[-1], SURFACE)):.2f}:1")
        return mono, span


    import matplotlib
    for cmap_name, lo, hi in [("viridis", 0.0, 1.0), ("Blues", 0.35, 1.0)]:
        cm = matplotlib.colormaps[cmap_name]
        hs = [matplotlib.colors.to_hex(cm(x)) for x in np.linspace(lo, hi, 6)]
        ramp_report(cmap_name, hs)
    print("\n  a single-hue ramp keeps hue span near 0; a multi-hue ramp does not.")


if __name__ == "__main__":
    _report()
