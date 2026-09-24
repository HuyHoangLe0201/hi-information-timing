r"""Where the fusion excess comes from, measured on the channels themselves.

Four simulations failed to reproduce the artefact Section 3.4 reports, so the
mechanism is not degeneracy in the abstract.  This decomposes the real quadratic
form instead.

For each bearing and each channel set, the residual correlation matrix is
diagonalised and the quadratic form written as a sum over its eigendirections,

    d' C^-1 d  =  sum_j  (v_j' d)^2 / w_j ,

so each direction's contribution is visible separately.  Three questions follow.
How much of the total does the smallest direction carry?  What is the eigenvalue
there?  And how large is the projection of the sensitivity onto it, relative to
what an unstructured vector would give?

If the artefact is a small eigenvalue meeting an ordinary projection, the guard
is testing the right quantity.  If it is an ordinary eigenvalue meeting a large
projection, or the reverse, it is not.
"""
import os
import json
import numpy as np

from fusion_prep import prepare_fusion

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

z = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
FN = [str(s) for s in z["featnames"]]
I = {k: i for i, k in enumerate(FN)}
units = sorted(k for k in z.files if k != "featnames")

prepared = []
for u in units:
    ch = {}
    for nm in FN:
        p, _ = prepare_fusion(np.asarray(z[u][:, I[nm]], float), nm)
        if p is not None:
            ch[nm] = p
    if len(ch) == len(FN) and len({p["n"] for p in ch.values()}) == 1:
        prepared.append((u, ch))
print("%d bearings carrying all %d channels\n" % (len(prepared), len(FN)))

BANDS = [n for n in FN if n.startswith("b")]
SETS = {"all channels": FN, "band fractions only": BANDS,
        "amplitudes only": [n for n in FN if not n.startswith("b")]}


def decompose(ch, names):
    """Per-direction contributions to the fused information, summed over the
    record.  The sensitivity varies along the record, so the projection is
    accumulated rather than taken at one sample."""
    R = np.column_stack([ch[n]["res"] for n in names])
    D = np.column_stack([ch[n]["dtrend"] for n in names])
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    w, V = np.linalg.eigh(C)
    P = (D @ V) ** 2                       # projection energy per direction
    contrib = P.sum(axis=0) / np.maximum(w, 1e-300)
    return w, P.sum(axis=0), contrib, float(np.linalg.cond(C))


print("%-22s %4s %11s %11s %11s %11s %11s"
      % ("set", "K", "cond", "w_min", "share_min", "proj_min", "proj_even"))
print("-" * 88)
rows = []
for lab, names in SETS.items():
    cn, wm, sh, pj, pe = [], [], [], [], []
    for u, ch in prepared:
        try:
            w, P, contrib, c = decompose(ch, names)
        except Exception:
            continue
        tot = contrib.sum()
        if not np.isfinite(tot) or tot <= 0:
            continue
        cn.append(c)
        wm.append(float(w[0]))
        sh.append(float(contrib[0] / tot))
        # projection energy in the smallest direction, against the average
        # direction's share, which is what an unstructured sensitivity gives
        pj.append(float(P[0] / P.sum()))
        pe.append(1.0 / len(w))
    if not cn:
        continue
    m = lambda a: float(np.median(a))
    r = dict(set=lab, K=len(names), cond=m(cn), w_min=m(wm), share=m(sh),
             proj=m(pj), even=m(pe), records=len(cn))
    rows.append(r)
    print("%-22s %4d %11.2e %11.2e %11.3f %11.4f %11.4f"
          % (lab, r["K"], r["cond"], r["w_min"], r["share"], r["proj"],
             r["even"]))
OUT["sets"] = rows

print("""
Reading the table.  'share_min' is the fraction of the fused information that
the smallest eigendirection alone carries.  'proj_min' is the fraction of the
sensitivity's energy lying in that direction, and 'proj_even' is what an
unstructured sensitivity would put there, namely one over K.
""")
for r in rows:
    ratio = r["proj"] / r["even"]
    print("  %-22s smallest direction carries %.1f%% of the information "
          "from %.2f of the sensitivity energy (%.2f x even)"
          % (r["set"], 100 * r["share"], r["proj"], ratio))

_all = next((r for r in rows if r["set"] == "all channels"), None)
if _all:
    # Two factors oppose each other and their ratio is the mechanism.
    suppress = _all["even"] / _all["proj"]        # how much the identity helps
    amplify = 1.0 / _all["w_min"]                 # how much the eigenvalue hurts
    # a correlation matrix has unit mean eigenvalue, so 1/w_min is the
    # amplification relative to an average direction
    net = amplify / suppress
    OUT["headline"] = dict(share=_all["share"], proj=_all["proj"],
                           w_min=_all["w_min"], cond=_all["cond"],
                           suppression=suppress, amplification=amplify,
                           net=net)
    print("""
  The sensitivity DOES avoid the degenerate direction: it places %.2f per cent
  of its energy there against %.1f per cent for an unstructured vector, a
  suppression of %.0f fold.  The identity is obeyed, approximately.

  It is not obeyed enough.  The eigenvalue there is %.1e against a mean of one
  for a correlation matrix, an amplification of %.0f fold, so the net factor is
  %.0f and the direction ends up carrying %.0f per cent of the fused
  information.

  That is why the four simulations found nothing.  Each built a sensitivity
  EXACTLY orthogonal to the degenerate direction, which makes the contribution
  exactly zero and the artefact impossible.  The real sensitivity is only
  approximately orthogonal, because it is estimated per channel from a smoother
  whose error does not obey the identity, and approximate is not enough when the
  denominator is 1e-5.
""" % (100 * _all["proj"], 100 * _all["even"], suppress,
       _all["w_min"], amplify, net, 100 * _all["share"]))
json.dump(OUT, open(os.path.join(HERE, "rank_guard_data.json"), "w"),
          indent=2, default=float)
print("written to rank_guard_data.json")
