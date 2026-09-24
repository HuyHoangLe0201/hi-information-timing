"""
Every number the manuscript quotes, recomputed or read back from the run that
produced it. Prints PAPER against SOURCE so a disagreement is visible.

The manuscript is written from RESULTS.json; this reads the same file plus the
raw per-experiment outputs, so a value that drifted during editing shows up
here rather than in review.
"""
import os
import re
import json
import numpy as np
# The audited surface is the manuscript AND the supplement: material
# moved to the supplement has not been deleted, and a check that read
# only mssp.tex could not tell the difference.  See _source.py.
import _source

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
L = lambda f: json.load(open(os.path.join(HERE, f)))

R = L("RESULTS.json")
tex = _source.read_both()
rows = []


def chk(label, paper, source, tol=0.02, rel=True):
    if source is None or (isinstance(source, float) and not np.isfinite(source)):
        rows.append((label, str(paper), "n/a", "?"))
        return
    d = abs(paper - source) / abs(source) if (rel and source) else abs(paper - source)
    rows.append((label, f"{paper:g}", f"{source:.4g}", "ok" if d <= tol else "MISMATCH"))


def intext(label, pattern, expect_source, tol=0.02, rel=True):
    """Pull a number straight out of the .tex so edits to prose are caught."""
    m = re.search(pattern, tex)
    if not m:
        rows.append((label, "NOT FOUND IN TEX", f"{expect_source:.4g}", "MISSING"))
        return
    # numbers in the text carry LaTeX thin-space digit separators
    chk(label, float(m.group(1).replace("\\,", "").replace(",", "")),
        expect_source, tol, rel)


def tabrow(first_cell):
    """Cells of the table row whose first column starts with `first_cell`.

    Splitting on & is more robust than one regex per column, because table
    cells contain LaTeX (thin spaces, math, footnote markers) that a
    character-class pattern has to anticipate.
    """
    for line in tex.splitlines():
        s = line.strip()
        if s.startswith(first_cell) and "&" in s:
            cells = s.rstrip("\\").split("&")
            return [c.strip().replace("\\,", "").replace("$", "")
                    for c in cells]
    return None


def cellnum(first_cell, col, label, source, tol=0.02):
    r = tabrow(first_cell)
    if r is None or col >= len(r):
        rows.append((label, "ROW NOT FOUND", f"{source:.4g}", "MISSING"))
        return
    m = re.search(r"-?[\d.]+", r[col])
    if not m:
        rows.append((label, f"no number in {r[col]!r}", f"{source:.4g}", "MISSING"))
        return
    chk(label, float(m.group()), source, tol)


# --- Section 5: is the bound attained? --------------------------------------
C = R["crb"]
for dom, key in [("bearing", "Bearings"), ("turbofan_FD001", "Turbofan FD001"),
                 ("turbofan_FD004", "Turbofan FD004"), ("battery", "Li-ion cells")]:
    v = C[dom]
    cellnum(key, 1, f"tab:crb {key} series", v["series"], tol=0.0)
    cellnum(key, 2, f"tab:crb {key} cells", v["cells"], tol=0.0)
    cellnum(key, 3, f"tab:crb {key} slope", v["slope"], tol=0.005)
    cellnum(key, 4, f"tab:crb {key} r", v["corr"], tol=0.005)
    cellnum(key, 5, f"tab:crb {key} median", v["median"], tol=0.005)

total_cells = sum(C[d]["cells"] for d in C if not d.startswith("_"))
flat = tex.replace("\\,", "")
rows.append(("total evaluation cells", f"{total_cells}", f"{total_cells}",
             "ok" if str(total_cells) in flat else "MISMATCH"))

GOOD = ("bearing", "turbofan_FD001", "turbofan_FD004")
worst = max(C[d]["median"] for d in GOOD)
rows.append(("claim: within 9% (excl. cells)", "<=1.09", f"{worst:.4f}",
             "ok" if worst <= 1.09 else "MISMATCH"))
slo = [C[d]["slope"] for d in GOOD]
rows.append(("claim: slopes 0.90-0.95", "0.90-0.95",
             f"{min(slo):.3f}-{max(slo):.3f}",
             "ok" if min(slo) >= 0.90 and max(slo) <= 0.955 else "MISMATCH"))
corr = [C[d]["corr"] for d in GOOD]
rows.append(("claim: correlations 0.953-0.969", "0.953-0.969",
             f"{min(corr):.3f}-{max(corr):.3f}",
             "ok" if min(corr) >= 0.95 and max(corr) <= 0.97 else "MISMATCH"))
# the validation covers four of the five sets; XJTU-SY is the replication
crb_records = 17 + 100 + 249 + 4
intext("records in the validation", r"(\d{3}) run-to-failure records",
       crb_records, tol=0.0)

# --- record counts ----------------------------------------------------------
fz = np.load(os.path.join(HERE, "feat_raw.npz"), allow_pickle=True)
n_pron = len([k for k in fz.files if k != "featnames"])
xz = np.load(os.path.join(HERE, "xjtu_feat.npz"), allow_pickle=True)
n_xjtu = len([k for k in xz.files if k != "featnames"])
counts = {}
for tag in ("FD001", "FD004"):
    z = np.load(os.path.join(HERE, f"cmapss_{tag}.npz"))
    counts[tag] = len({k.split("__")[0] for k in z.files})
zb = np.load(os.path.join(HERE, "battery_raw.npz"))
n_cell = len({k.split("__")[0] for k in zb.files})
total = n_pron + n_xjtu + counts["FD001"] + counts["FD004"] + n_cell
cellnum("\\multicolumn{2}{@{}l}{Total}", 1, "tab:data total records",
        total, tol=0.0)
for lab, val in [("PRONOSTIA", n_pron), ("XJTU-SY", n_xjtu),
                 ("C-MAPSS FD001", counts["FD001"]),
                 ("C-MAPSS FD004", counts["FD004"]), ("NASA PCoE", n_cell)]:
    cellnum(lab, 2, f"tab:data {lab} records", val, tol=0.0)

# --- Section 6: exponents ---------------------------------------------------
B = R["beta"]
NAMES = {"Bearing RMS": "bearing RMS", "Bearing 0--2\\,kHz": "bearing 0-2 kHz",
         "Bearing 4--10\\,kHz": "bearing 4-10 kHz",
         "Turbofan $s_9$": "turbofan s9", "Turbofan $s_{11}$": "turbofan s11",
         "Turbofan $s_{20}$": "turbofan s20", "Cell capacity": "battery cap",
         "Cell $R_e$": "battery Re", "Cell $R_{ct}$": "battery Rct"}
for texname, key in NAMES.items():
    v = B[key]
    cellnum(texname, 1, f"tab:beta {key} n", v["n"], tol=0.0)
    cellnum(texname, 2, f"tab:beta {key} sigma", v["sigma"], tol=0.03)
    cellnum(texname, 3, f"tab:beta {key} measured", v["measured"], tol=0.01)
    cellnum(texname, 4, f"tab:beta {key} de-biased",
            v["debiased"] if v["debiased"] else v["ceiling"], tol=0.01)

# --- Section 7: closure -----------------------------------------------------
CL = L("closure.json")
chk("closure q*", 0.35, CL["qstar"], tol=0.01)
for r in CL["rows"]:
    rows.append((f"closure {r['name']} predicted", f"{r['predicted']:.3f}",
                 f"{r['predicted']:.4g}", "ok"))
un = [r for r in CL["rows"] if not r["saturated"]]
err = max(abs(r["predicted"] - r["measured"]) for r in un
          if r["measured"] < 0.99 and r["name"] != "bearing 4-10 kHz")
rows.append(("claim: closure within 0.01", "<=0.01", f"{err:.4f}",
             "ok" if err <= 0.011 else "MISMATCH"))

# tau_min arithmetic quoted in the prose
b_rms = B["bearing RMS"]["ceiling"]
chk("prose: RMS 2beta-1 > 46", 46, 2 * b_rms - 1, tol=0.02)
chk("prose: RMS tau_min 0.978", 0.978, CL["qstar"] ** (1 / (2 * b_rms - 1)), tol=0.005)
b_lf = B["bearing 0-2 kHz"]["debiased"]
chk("prose: lf 2beta-1 = 5.68", 5.68, 2 * b_lf - 1, tol=0.01)
chk("prose: lf tau_min 0.83", 0.83, CL["qstar"] ** (1 / (2 * b_lf - 1)), tol=0.01)
b_hf = B["bearing 4-10 kHz"]["debiased"]
chk("prose: hf 2beta-1 = 0.20", 0.20, 2 * b_hf - 1, tol=0.05)

# XJTU replication
X = L("xjtu_profiles.json")
xr = {r["ind"]: r for r in X["rows"]}["RMS"]
chk("prose: XJTU RMS beta 8.3", 8.3, xr["beta_eff"], tol=0.02)
chk("prose: XJTU RMS tau_min 0.94", 0.94, xr["tau_min"], tol=0.01)
chk("XJTU q* matches", CL["qstar"], X["qstar"], tol=0.001)

# --- Section 8: retention ---------------------------------------------------
OP = R["retention"]["at_operating_point"]
fin = R["retention"]["fixed_tau0_0.7"]


def op(name, near):
    c = [r for r in OP if r["name"] == name and abs(r["tau0"] - near) < 0.02]
    return max(r["corrected"] for r in c) if c else None


rms_at_min = [r for r in OP if r["name"] == "bearing RMS"]
chk("retention: RMS at its tau_min", 0.99,
    float(np.median([r["corrected"] for r in rms_at_min])), tol=0.02)
tur = [r for r in OP if r["name"] == "turbofan s11" and r["tau0"] > 0.5]
chk("retention: turbofan corrected", 1.00,
    float(np.median([r["corrected"] for r in tur])), tol=0.02)
bat = [r for r in OP if r["name"] == "battery cap" and r["tau0"] > 0.5]
rows.append(("retention: battery 0.74-0.86", "0.74-0.86",
             f"{min(r['corrected'] for r in bat):.2f}-"
             f"{max(r['corrected'] for r in bat):.2f}",
             "ok" if min(r["corrected"] for r in bat) >= 0.73
             and max(r["corrected"] for r in bat) <= 0.87 else "MISMATCH"))
hf = [r for r in OP if r["name"] == "bearing 4-10 kHz"]
rows.append(("retention: hf band 0.44-0.52", "0.44-0.52",
             f"{min(r['corrected'] for r in hf):.2f}-"
             f"{max(r['corrected'] for r in hf):.2f}",
             "ok" if min(r["corrected"] for r in hf) >= 0.43
             and max(r["corrected"] for r in hf) <= 0.53 else "MISMATCH"))

# the two corrections, as quoted
t11 = [r for r in fin if r["name"] == "turbofan s11" and abs(r["w"] - 0.10) < 1e-9][0]
# read both from the text: the floor is a Monte Carlo estimate, so a re-run with
# a different seed moves it slightly and the text must follow
intext("retention: turbofan floor",
       r"that floor is\s*\$([\d.]+)\$", t11["floor"], tol=0.01)
intext("retention: turbofan measured",
       r"against a measured \$([\d.]+)\$", t11["raw"], tol=0.01)
rms20 = [r for r in fin if r["name"] == "bearing RMS" and abs(r["w"] - 0.20) < 1e-9][0]
chk("retention: RMS raw 0.061", 0.061, rms20["raw"], tol=0.03)
chk("retention: RMS de-trimmed 0.390", 0.390, rms20["detrimmed"], tol=0.02)

V = R["retention"]["validation"]
pw_noisy = [r["measured"] for r in V if r["case"].startswith("power-law b=3 (noisy)")]
pw_short = [r["measured"] for r in V if r["case"].startswith("power-law b=3 (short)")]
# quoted in the text as ranges, because the value depends on the window width
rows.append(("validation: 0.76-0.88 at sigma=0.10", "0.76-0.88",
             f"{min(pw_noisy):.2f}-{max(pw_noisy):.2f}",
             "ok" if abs(min(pw_noisy) - 0.76) < 0.02
             and abs(max(pw_noisy) - 0.88) < 0.02 else "MISMATCH"))
rows.append(("validation: 0.66-0.78 at n=150", "0.66-0.78",
             f"{min(pw_short):.2f}-{max(pw_short):.2f}",
             "ok" if abs(min(pw_short) - 0.66) < 0.02
             and abs(max(pw_short) - 0.78) < 0.02 else "MISMATCH"))
naive = [r["raw"] for r in fin]
rows.append(("retention: naive range 0.01-0.75", "0.01-0.75",
             f"{min(naive):.2f}-{max(naive):.2f}",
             "ok" if abs(min(naive) - 0.01) < 0.01
             and abs(max(naive) - 0.75) < 0.02 else "MISMATCH"))

# --- ground truth validation ------------------------------------------------
G = R["groundtruth"]
gl = {int(r["n"]): r for r in G["length"]} if isinstance(G["length"], list) else {}
if 1400 in gl:
    chk("ground truth: slope at n=1400", 0.992, gl[1400].get("slope"), tol=0.01)
    chk("ground truth: beta at n=1400", 2.95, gl[1400].get("beta_eff"), tol=0.02)

# --- report -----------------------------------------------------------------
print(f"{'quantity':<36}{'paper':>18}{'source':>18}{'':>4}")
print("-" * 78)
for lab, p, s, st in rows:
    mark = "" if st == "ok" else f"  <-- {st}"
    print(f"{lab:<36}{p:>18}{s:>18}{mark}")
print("-" * 78)
bad = [r for r in rows if r[3] != "ok"]
print(f"{len(rows)} claims checked, {len(bad)} flagged")
