r"""How many of the eleven certificates survive their own sampling error?

The certificate is a per-record statement: this bearing's pair tolerates a
contrast of T, the corrections impose at most R, and the record is certified when
R < T.  Both sides are estimated from one record of finite length, and
length_screen.py has just measured how imprecise per-record contrasts are -- a
sampling spread of 0.30 in log units for the serial correction at three hundred
samples, against a typical log-contrast of 0.48.  A count of eleven certified
records is therefore a count of estimates, and the paper reports it as a count of
facts.

The margin is what matters, log T - log R, and its sampling error is measured the
way reversal_resolution.py measures the family margin's: decimate a long record by
k, recompute the whole certificate for each of the k phases, and take the spread
across phases.  Record, channels and pipeline are held fixed; only length moves.

Then each real record's margin is compared with the spread at its own length.  A
certificate whose margin is smaller than that spread is not a wrong statement, but
it is one the data cannot support at the length it was computed on, and the count
should say so.
"""
import os
import json
import numpy as np
from scipy.signal import savgol_filter

from pipeline import robust_scale, _win, local_scale, LOCAL_BW
from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(11235)
OUT = {}
Q = 0.35
THETA = Q / (1.0 - Q)
MIN_SHARE = 0.02
H_HALF, WIDE = 0.04, 4.0
CUTS = (0.3, 0.5, 0.7)
DISTORTIONS = ("serial", "tail", "derivative")
AMOUNT = ("rms", "peak")
DISTRIB = ("b4_6", "b6_8", "b8_10")
KS = (1, 2, 4, 8)


def rolling_phi(res, bw=0.20):
    r = np.asarray(res, float)
    m = len(r)
    h = max(8, int(bw * m))
    lo = np.maximum(0, np.arange(m) - h)
    hi = np.minimum(m, np.arange(m) + h + 1)

    def win(a, k=0):
        cc = np.concatenate(([0.0], np.cumsum(a)))
        return cc[np.minimum(hi - k, len(a))] - cc[np.minimum(lo, len(a))]

    cnt = (hi - lo).astype(float)
    s1, s2 = win(r), win(r * r)
    mu = s1 / np.maximum(cnt, 1)
    var = s2 - cnt * mu * mu
    cross = win(r[:-1] * r[1:], 1) - np.maximum(cnt - 1, 0) * mu * mu
    return np.clip(np.where(var > 0, cross / np.maximum(var, 1e-300), 0.0),
                   -0.9, 0.9)


def pieces(x, min_n=60):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < min_n:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate", rng, reps=6),
                0.0, None)
    if d.sum() <= 0:
        return None
    s = robust_scale(x)
    D = (x - np.median(x)) / s
    w_ = _win(n)
    sl = np.clip(local_scale(res, max(5, int(LOCAL_BW * n))), 1e-12, None)
    out = {}
    ph = rolling_phi(res)
    out["serial"] = (1.0 - ph) / (1.0 + ph)
    h = max(8, int(LOCAL_BW * n))
    lo = np.maximum(0, np.arange(n) - h)
    hi = np.minimum(n, np.arange(n) + h + 1)
    cc = np.concatenate(([0.0], np.cumsum(res)))
    c2 = np.concatenate(([0.0], np.cumsum(res * res)))
    cnt = (hi - lo).astype(float)
    mu = (cc[hi] - cc[lo]) / cnt
    var = np.maximum((c2[hi] - c2[lo]) / cnt - mu * mu, 0.0)
    out["tail"] = 1.0 / np.clip((np.sqrt(var) / sl) ** 2, 1e-3, 1e3)
    wf = max(11, int(WIDE * 0.08 * n))
    wf += (wf % 2 == 0)
    if wf < n:
        dt = savgol_filter(D, w_, 2, deriv=1, delta=1.0 / n)
        d3 = savgol_filter(D, wf, 4, deriv=3, delta=1.0 / n)
        cor = ((dt - (H_HALF ** 2 / 10.0) * d3) / sl) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            out["derivative"] = np.clip(
                np.where(dens > 0, cor / np.maximum(dens, 1e-300), 1.0),
                0.2, 5.0)
    return d, out


def contrast(w, g, ic):
    tot = float(np.sum(g))
    if tot <= 0:
        return np.nan
    if (float(np.sum(g[:ic + 1])) / tot < MIN_SHARE
            or float(np.sum(g[ic + 1:])) / tot < MIN_SHARE):
        return np.nan
    a = float(np.sum(w[:ic + 1] * g[:ic + 1]) / np.sum(g[:ic + 1]))
    b = float(np.sum(w[ic + 1:] * g[ic + 1:]) / np.sum(g[ic + 1:]))
    return a / b if b > 0 else np.nan


def age_at_odds(F, x):
    p = x / (1.0 + x)
    if p >= 1.0:
        return np.inf
    i = int(np.searchsorted(F, p))
    return np.inf if i >= len(F) else (i + 1) / len(F)


def certificate(block, FN, min_n=60):
    """(tolerated, worst contrast) for one record, or None."""
    best = {}
    for grp, names in (("d", DISTRIB), ("a", AMOUNT)):
        cand = []
        for nm in names:
            if nm not in FN:
                continue
            p = pieces(block[:, FN.index(nm)], min_n)
            if p is None:
                continue
            g, ws = p
            F = np.cumsum(g) / g.sum()
            a = age_at_odds(F, THETA)
            if np.isfinite(a):
                cand.append((a, g, F, ws))
        if cand:
            best[grp] = min(cand, key=lambda t: t[0])
    if "d" not in best or "a" not in best:
        return None
    ad, dd, Fd, wd = best["d"]
    aa, da, Fa, wa = best["a"]
    if not (ad < aa):
        return None
    lo, hi = 1.0, 40.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if age_at_odds(Fd, mid ** 2 * THETA) < age_at_odds(Fa, THETA / mid ** 2):
            lo = mid
        else:
            hi = mid
    worst = 0.0
    for ws, g in ((wd, dd), (wa, da)):
        for dist in DISTORTIONS:
            if dist not in ws:
                continue
            w = np.clip(ws[dist], 1e-9, None)
            for cf in CUTS:
                v = contrast(w, g, int(cf * len(g)) - 1)
                if np.isfinite(v) and v > 0:
                    worst = max(worst, max(v, 1.0 / v))
    if worst <= 0:
        return None
    return float(lo ** 2), float(worst)


z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
units = [k for k in z.files if k != "featnames"]

print("=" * 92)
print("1.  The certificate margin, and its sampling error against length")
print("=" * 92)
print("""
Each long bearing is decimated by k and the whole certificate recomputed for each
phase.  The margin is log(tolerated) - log(worst contrast), so a positive margin
is a certificate and its size is what the certificate is worth.
""")
LONG = [u for u in units if z[u].shape[0] >= 2200]
print("  %-8s %9s %10s %14s %10s"
      % ("k", "length", "records", "spread(log)", "phases"))
print("  " + "-" * 56)
rows = []
for k in KS:
    spreads, ph = [], 0
    for u in LONG:
        got = []
        for off in range(k):
            c = certificate(z[u][off::k], FN)
            if c is not None:
                got.append(np.log(c[0]) - np.log(c[1]))
        ph += len(got)
        if len(got) >= 2:
            spreads.append(float(np.std(got, ddof=1)))
    ln = int(np.median([z[u].shape[0] for u in LONG]) / k)
    r = dict(k=k, length=ln, records=len(LONG),
             spread=float(np.median(spreads)) if spreads else 0.0, phases=ph)
    rows.append(r)
    print("  %-8d %9d %10d %14.4f %10d"
          % (k, ln, len(LONG), r["spread"], ph))
OUT["decimation"] = rows
OUT["long_records"] = len(LONG)


def spread_at(n):
    xs = np.array([r["length"] for r in rows if r["spread"] > 0], float)
    ys = np.array([r["spread"] for r in rows if r["spread"] > 0], float)
    if len(xs) < 2:
        return np.nan
    o = np.argsort(xs)
    return float(np.interp(np.log(n), np.log(xs[o]), ys[o]))


print()
print("=" * 92)
print("2.  Every record's margin against the spread at its own length")
print("=" * 92)
ref = json.load(open(os.path.join(HERE, "contrast_identity.json")))
print("""
The tolerated and worst contrasts are the ones already reported; only the
resolution is new.  A record is certified when the margin is positive and
resolved when it exceeds the spread at that record's length.
""")
print("  %-14s %7s %10s %10s %9s %8s %8s %10s"
      % ("record", "n", "tolerated", "worst", "margin", "spread", "z",
         "verdict"))
print("  " + "-" * 84)
cert = []
for r in ref["per_bearing"]:
    n = int(z[r["unit"]].shape[0])
    m = float(np.log(r["tolerated"]) - np.log(r["worst_contrast"]))
    s = spread_at(n)
    zz = m / s if s > 0 else np.nan
    # Three classes, not two.  A record whose margin is negative by less than
    # its own sampling error is no more "refused" than one certified by less
    # than its own error is "certified"; both are undecidable at that length,
    # and reporting only the certified side of that would be the same mistake
    # the screen made.
    verdict = ("undecidable" if abs(zz) <= 1.0 else
               ("resolved" if m > 0 else "refused"))
    cert.append(dict(unit=r["unit"], n=n, tolerated=r["tolerated"],
                     worst=r["worst_contrast"], margin=m, spread=s, z=zz,
                     verdict=verdict))
    print("  %-14s %7d %10.3f %10.3f %9.3f %8.3f %8.2f %10s"
          % (r["unit"], n, r["tolerated"], r["worst_contrast"], m, s, zz,
             verdict))
OUT["records"] = cert
OUT["n_records"] = len(cert)
OUT["n_certified"] = sum(1 for c in cert if c["margin"] > 0)
OUT["n_resolved"] = sum(1 for c in cert if c["verdict"] == "resolved")
OUT["n_refused"] = sum(1 for c in cert if c["verdict"] == "refused")
OUT["n_undecidable"] = sum(1 for c in cert if c["verdict"] == "undecidable")
OUT["n_certified_undecidable"] = sum(
    1 for c in cert if c["verdict"] == "undecidable" and c["margin"] > 0)
_res = [c for c in cert if c["verdict"] == "resolved"]
OUT["median_z_resolved"] = float(np.median([c["z"] for c in _res])) if _res \
    else None
OUT["spread_at_2400"] = spread_at(2400)
OUT["spread_at_700"] = spread_at(700)
OUT["longest_undecidable"] = max(
    (c["n"] for c in cert if c["verdict"] == "undecidable"), default=0)

print("""
  Of the %d records the distortion results admit, %d are certified on the point
  estimate.  Splitting by whether the margin exceeds its own sampling error, %d
  are certified and resolved, %d are refused and resolved, and %d are
  undecidable at their length -- %d of those undecidable records having a
  positive margin and the rest a negative one.  The margin's spread is %.3f in
  log units at two thousand four hundred samples and %.3f at seven hundred, so
  resolution is set by length; the longest undecidable record runs to %d.

  That does not withdraw the certificate.  It says the eleven is a count of
  estimates, %d of which the data can distinguish from the boundary, and that a
  reader should carry the second figure.  Reporting only the certified side of
  the undecidable band would repeat, in a smaller way, the mistake the length
  screen made.
""" % (OUT["n_records"], OUT["n_certified"], OUT["n_resolved"],
       OUT["n_refused"], OUT["n_undecidable"], OUT["n_certified_undecidable"],
       OUT["spread_at_2400"], OUT["spread_at_700"],
       OUT["longest_undecidable"], OUT["n_resolved"]))

json.dump(OUT, open(os.path.join(HERE, "certificate_resolution.json"), "w"),
          indent=2, default=float)
print("written to certificate_resolution.json")
