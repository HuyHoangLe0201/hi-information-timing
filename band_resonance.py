"""
If a structural resonance is what makes a band early, it should also be loud.

The earliest usable band falls with centre frequency on both rigs, -0.71 on
PRONOSTIA and -0.43 on XJTU, which is the direction impulsive damage ringing a
resonance predicts. But the two rigs disagree on WHICH band is best -- rank
agreement +0.14, PRONOSTIA choosing 8-10 kHz and XJTU 6-8 kHz -- so the specific
band cannot be recommended. That is expected if the resonance belongs to the
housing and mounting rather than to bearings in general.

The mechanism makes a further prediction that does not require the rigs to
agree. A resonance is loud: the band containing it should carry more energy, and
more of the energy that GROWS over life, than its neighbours. So on each rig
separately the early band should also be the one whose energy rises most.

If the earliest band and the most-growing band coincide on each rig while
differing between rigs, the mechanism is supported and its rig-specificity
explained. If they do not coincide, frequency is ordering the bands for some
other reason.
"""
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EDGES = {"b0_1": 0.5, "b1_2": 1.5, "b2_4": 3.0, "b4_6": 5.0,
         "b6_8": 7.0, "b8_10": 9.0, "b10_12.8": 11.4}


def growth(series):
    """Median ratio of end-of-life band energy to early-life energy."""
    v = []
    for x in series:
        x = np.asarray(x, float)
        n = len(x)
        if n < 40:
            continue
        early = float(np.median(x[:max(3, n // 10)]))
        late = float(np.median(x[-max(3, n // 20):]))
        if early > 0 and np.isfinite(late):
            v.append(late / early)
    return float(np.median(v)) if v else np.nan


def share(series):
    """Median share of total band energy this band carries at end of life."""
    return None


BF = {(r["rig"], r["band"]): r["tau"]
      for r in json.load(open(os.path.join(HERE, "band_frequency.json")))}

for tag, path in (("PRONOSTIA", "feat_raw.npz"), ("XJTU", "xjtu_feat.npz")):
    z = np.load(os.path.join(HERE, path), allow_pickle=True)
    FN = [str(s) for s in z["featnames"]]
    I = {k: i for i, k in enumerate(FN)}
    BK = sorted(k for k in z.files if k != "featnames")
    bands = [b for b in EDGES if b in I]
    tot_end = None
    rows = []
    for b in bands:
        s = [z[k][:, I[b]] for k in BK]
        rows.append(dict(band=b, centre=EDGES[b], growth=growth(s),
                         tau=BF.get((tag, b), np.nan)))
    print(f"\n=== {tag} ===\n")
    print(f"{'band':<12}{'centre kHz':>11}{'energy growth':>15}{'tau(0.35)':>11}")
    print("-" * 49)
    for r in sorted(rows, key=lambda r: r["centre"]):
        print(f"{r['band']:<12}{r['centre']:>11.1f}{r['growth']:>15.1f}"
              f"{r['tau']:>11.3f}")
    print("-" * 49)
    ok = [r for r in rows if np.isfinite(r["growth"]) and np.isfinite(r["tau"])]
    if len(ok) >= 4:
        g = [r["growth"] for r in ok]; t = [r["tau"] for r in ok]
        rc = float(np.corrcoef(np.argsort(np.argsort(g)),
                               np.argsort(np.argsort(t)))[0, 1])
        loudest = max(ok, key=lambda r: r["growth"])
        earliest = min(ok, key=lambda r: r["tau"])
        print(f"rank correlation of growth with tau: {rc:+.2f}")
        print(f"band that grows most : {loudest['band']} "
              f"({loudest['centre']:.1f} kHz, x{loudest['growth']:.0f})")
        print(f"band usable earliest : {earliest['band']} "
              f"({earliest['centre']:.1f} kHz, tau {earliest['tau']:.3f})")
        print(f"-> {'the same band' if loudest['band'] == earliest['band'] else 'DIFFERENT bands'}")
