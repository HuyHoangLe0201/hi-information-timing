r"""Does tau_q rank indicators by WHEN information arrives or by HOW MUCH?

A reviewer put the sharpest question this paper has had.  The measured quantity

    tau_q = Ghat^{-1}( q Ghat(1) )

is a quantile of the NORMALISED curve, so it divides Ghat(1) out and keeps only
where along the record the information sits.  Two indicators whose budgets
differ by orders of magnitude can share a tau_q, and an indicator carrying very
little can rank early by delivering its little early.  That is a property of the
estimand and Definition 1 says so -- but the paper then reported the result
under the word "usability", which a reader takes to mean usefulness.

The framework's own design quantity is the other one.  Proposition 1(ii) gives
tau_min = G^{-1}(eps^{-2}), the earliest age a STATED PRECISION is reachable,
and that is an absolute demand under which more information is better.  So the
two questions can be asked of the same records, and this asks both:

  relative   the age at which a fixed fraction q of a record's own budget has
             arrived -- what the paper measures
  absolute   the age at which a fixed absolute level L has arrived, the same L
             for every measure on a rig -- what tau_min would ask

L is swept over two decades because a single level would prove nothing.  Where a
record never reaches L its age is censored at 1, which is the honest reading --
the indicator never supplies that much -- and the count is reported with the
age so that a median resting on censored records can be seen to be doing so.

Neither ranking is a certificate.  The absolute one in particular is not a claim
this paper can make: Appendix C measures the fraction of derivative energy
surviving an estimated trend at 0.0013 to 0.0231 across channels, a spread of
17.8, so two budgets are not comparable to better than that factor.  The point
of computing it is to find out whether the two questions have the same answer
here.  They do not.

    python timing_vs_amount.py   ->  timing_vs_amount.json
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr

from nonparam import weighted_density, estimate_floor

HERE = os.path.dirname(os.path.abspath(__file__))
Q = 0.35
EPS = 1e-30
SEED = 20260924
CACHE = os.path.join(HERE, "timing_curves.npz")


# ------------------------------------------------------------------ measures --
def norm(P):
    return P / np.clip(P.sum(axis=1, keepdims=True), EPS, None)


def entropy(P):
    p = norm(P)
    return -(p * np.log(np.clip(p, EPS, None))).sum(axis=1)


def gini(P):
    p = np.sort(norm(P), axis=1)
    m = p.shape[1]
    return (2 * (p * np.arange(1, m + 1)).sum(axis=1)) \
        / np.clip(p.sum(axis=1), EPS, None) / m - (m + 1) / m


MEASURES = [
    ("spectral entropy", "distribution", lambda P, f: entropy(P)),
    ("spectral centroid", "distribution", lambda P, f: (norm(P) * f).sum(axis=1)),
    ("spectral spread", "distribution",
     lambda P, f: np.sqrt((norm(P) * (f - (norm(P) * f).sum(axis=1,
                                                            keepdims=True)) ** 2
                           ).sum(axis=1))),
    ("spectral Gini", "distribution", lambda P, f: gini(P)),
    ("top-8-band share", "distribution",
     lambda P, f: np.sort(norm(P), axis=1)[:, -8:].sum(axis=1)),
    ("high/low band ratio", "distribution",
     lambda P, f: P[:, -8:].sum(axis=1)
     / np.clip(P[:, :8].sum(axis=1), EPS, None)),
    ("total power", "amount", lambda P, f: P.sum(axis=1)),
    ("peak band power", "amount", lambda P, f: P.max(axis=1)),
    ("high-band power", "amount", lambda P, f: P[:, -8:].sum(axis=1)),
    ("log total power", "amount",
     lambda P, f: np.log(np.clip(P.sum(axis=1), EPS, None))),
]
KIND = {m[0]: m[1] for m in MEASURES}
RIGS = {"PRONOSTIA": "fineband.npz", "XJTU": "fineband_xjtu.npz"}


def cumulative(x):
    """The floor-corrected cumulative curve of one record, or None."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        return None
    dens, res = weighted_density(x)
    if dens is None:
        return None
    d = np.clip(dens - estimate_floor(dens, res, "surrogate",
                                      np.random.default_rng(SEED)), 0.0, None)
    if d.sum() <= 0:
        return None
    return np.cumsum(d)


def build():
    """Every (rig, measure, unit) cumulative curve, computed once."""
    store = {}
    for rig, fn in RIGS.items():
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            print("  %s: %s not found" % (rig, fn))
            continue
        z = np.load(path)
        f = z["centres"] / 1000.0
        BK = sorted(k for k in z.files if k != "centres")
        for lab, kind, fun in MEASURES:
            for b in BK:
                c = cumulative(fun(z[b].astype(float), f))
                if c is not None:
                    store["%s|%s|%s" % (rig, lab, b)] = c
        print("  %s: %d curves" % (rig, sum(1 for k in store
                                            if k.startswith(rig + "|"))))
    np.savez_compressed(CACHE, **store)
    return store


def load():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print("  %d curves from cache" % len(z.files))
        return {k: z[k] for k in z.files}
    print("building the curves ...")
    return build()


def age_at_absolute(c, L):
    """Age at which the cumulative reaches L; 1.0 if it never does."""
    k = int(np.searchsorted(c, L))
    return min((k + 1) / len(c), 1.0)


def main():
    store = load()
    out = {}

    for rig in RIGS:
        per = {}
        for key, c in store.items():
            r, lab, unit = key.split("|")
            if r == rig:
                per.setdefault(lab, []).append(c)
        if not per:
            continue

        # ---------------------------------------------- relative, the paper's --
        rows = []
        for lab, cs in per.items():
            taus, tots = [], []
            for c in cs:
                tot = float(c[-1])
                k = int(np.searchsorted(c / tot, Q))
                taus.append(min((k + 1) / len(c), 1.0))
                tots.append(tot)
            rows.append(dict(measure=lab, kind=KIND[lab], units=len(cs),
                             budget=float(np.median(tots)),
                             tau=float(np.median(taus))))
        g = np.array([r["budget"] for r in rows])
        t = np.array([r["tau"] for r in rows])
        rho, pv = spearmanr(t, g)
        med_rel = {k: float(np.median([r["tau"] for r in rows
                                       if r["kind"] == k]))
                   for k in ("distribution", "amount")}

        print("=" * 76)
        print("%s" % rig)
        print("=" * 76)
        print("%-22s%-14s%15s%10s" % ("measure", "family", "budget", "tau_q"))
        print("-" * 76)
        for r in sorted(rows, key=lambda r: r["tau"]):
            print("%-22s%-14s%15.4g%10.3f"
                  % (r["measure"], r["kind"], r["budget"], r["tau"]))
        print("-" * 76)
        print("budgets span x%.1f; median budget: distribution %.3g, amount %.3g"
              % (g.max() / g.min(),
                 np.median([r["budget"] for r in rows
                            if r["kind"] == "distribution"]),
                 np.median([r["budget"] for r in rows if r["kind"] == "amount"])))
        print("Spearman(tau_q, budget) = %+.3f  (p = %.3f)" % (rho, pv))
        print("relative gap, amount - distribution = %+.3f"
              % (med_rel["amount"] - med_rel["distribution"]))

        # ------------------------------------------------ absolute, a sweep --
        base = float(np.median([np.median([c[-1] for c in cs])
                                for cs in per.values()]))
        print()
        print("  the same measures under a fixed absolute demand L")
        print("  %-14s%12s%12s%10s%10s"
              % ("L / base", "dist age", "amount age", "gap", "censored"))
        sweep = []
        for mult in (0.05, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0):
            L = mult * base
            fam, cens = {"distribution": [], "amount": []}, 0
            for lab, cs in per.items():
                ages = [age_at_absolute(c, L) for c in cs]
                cens += sum(1 for c in cs if c[-1] < L)
                fam[KIND[lab]].append(float(np.median(ages)))
            d = float(np.median(fam["distribution"]))
            a = float(np.median(fam["amount"]))
            sweep.append(dict(mult=mult, L=L, distribution=d, amount=a,
                              gap=a - d, censored=cens,
                              total=sum(len(cs) for cs in per.values())))
            print("  %-14.2f%12.3f%12.3f%10.3f%10d"
                  % (mult, d, a, a - d, cens))
        signs = [1 if s["gap"] > 0 else -1 for s in sweep]
        print("  the absolute gap is negative at %d of %d levels"
              % (sum(1 for x in signs if x < 0), len(signs)))

        fam_budget = {k: float(np.median([r["budget"] for r in rows
                                          if r["kind"] == k]))
                      for k in ("distribution", "amount")}
        out[rig] = dict(rows=rows, spearman_tau_budget=float(rho), p=float(pv),
                        budget_span=float(g.max() / g.min()),
                        family_budget=fam_budget,
                        budget_ratio=fam_budget["amount"]
                        / fam_budget["distribution"],
                        relative_gap=med_rel["amount"] - med_rel["distribution"],
                        base=base, absolute_sweep=sweep)
        print()

    json.dump(out, open(os.path.join(HERE, "timing_vs_amount.json"), "w"),
              indent=1)
    print("written to timing_vs_amount.json")


if __name__ == "__main__":
    main()
