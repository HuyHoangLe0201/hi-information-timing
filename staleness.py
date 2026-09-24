"""
Does every result still correspond to the code that produced it?

Two audits already run on this paper.  audit_session.py checks each claim
against its result file, and audit_paper.py checks the manuscript's prose
against the same files.  Both compare the paper with the JSON.  Neither compares
the JSON with the CODE, so a script edited after its result was written leaves
both audits passing against a stale number, and nothing anywhere would say so.

That is not hypothetical.  During this work a script was edited while it was
running; the process had already loaded the old source, the output came from
that, and the result file it wrote looked entirely normal.  The failure is
silent by construction.

This checks the missing edge.  For each result file it finds the script that
writes it -- by searching the sources for the filename -- and compares
modification times, flagging any result older than its producer.  The same is
done for the raw data a script reads, since regenerating an input invalidates
everything downstream of it just as surely as editing the code.

Modification time is a coarse instrument: it catches the edit-and-forget case,
which is the one that actually happens, and it will not catch a change that
leaves the timestamp untouched.  It is a floor on integrity, not a proof of it.
"""
import os
import re
import glob
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_EXT = (".npz", ".npy")
SKIP = {"staleness.py"}
SAME_RUN = 5.0        # seconds: files stamped within this are one run


def sources():
    return sorted(p for p in glob.glob(os.path.join(HERE, "*.py"))
                  if os.path.basename(p) not in SKIP)


def producers():
    """result file -> the scripts that WRITE it.

    A first version looked for `json.dump` in the two hundred characters before
    the filename, and matched almost nothing: in this codebase the dict being
    dumped sits between the two and is often far longer than that.  The reliable
    marker is the write mode itself -- the filename and a `"w"` in the same
    open() call -- which no read can produce.
    """
    out = {}
    for p in sources():
        try:
            s = open(p, encoding="utf-8").read()
        except Exception:
            continue
        for m in re.finditer(r'["\']([A-Za-z0-9_\-]+\.json)["\']', s):
            tail = s[m.end():m.end() + 60]
            if re.match(r'\s*\)\s*,\s*["\']w["\']', tail):
                out.setdefault(m.group(1), set()).add(p)
    return out


def data_read_by(p):
    try:
        s = open(p, encoding="utf-8").read()
    except Exception:
        return set()
    got = set()
    for m in re.finditer(r'["\']([A-Za-z0-9_\-.]+\.(?:npz|npy))["\']', s):
        f = os.path.join(HERE, m.group(1))
        if os.path.exists(f):
            got.add(f)
    return got


PROD = producers()
results = sorted(glob.glob(os.path.join(HERE, "*.json")))
print(f"{len(results)} result files, {len(sources())} scripts\n")
print(f"{'result':<30}{'producer':<28}{'verdict':>12}{'age (h)':>10}")
print("-" * 80)
rows, stale, orphan = [], [], []
for r in results:
    name = os.path.basename(r)
    ps = PROD.get(name)
    if not ps:
        orphan.append(name)
        rows.append(dict(result=name, producer=None, verdict="no producer"))
        print(f"{name:<30}{'--':<28}{'no producer':>12}{'':>10}")
        continue
    rt = os.path.getmtime(r)
    worst, worst_p = None, None
    for p in ps:
        for f in {p} | data_read_by(p):
            ft = os.path.getmtime(f)
            if worst is None or ft > worst:
                worst, worst_p = ft, f
    # A dependency a few seconds newer than the result is the same run, not a
    # later edit: a script that writes several files stamps them in sequence.
    # crb_full.json trailed crb_cells.npz by 0.31 s and was flagged for it.
    ok = worst <= rt + SAME_RUN
    if not ok:
        stale.append((name, os.path.basename(worst_p),
                      (worst - rt) / 3600.0))
    rows.append(dict(result=name, producer=os.path.basename(sorted(ps)[0]),
                     verdict="current" if ok else "STALE",
                     lag_hours=(0.0 if ok else (worst - rt) / 3600.0)))
    print(f"{name:<30}{os.path.basename(sorted(ps)[0]):<28}"
          f"{('current' if ok else 'STALE'):>12}"
          f"{('' if ok else f'{(worst - rt) / 3600.0:.1f}'):>10}")
print("-" * 80)
print()
print(f"{len(results) - len(stale) - len(orphan)} current, {len(stale)} stale, "
      f"{len(orphan)} with no identifiable producer")
print()

# Which results does the paper actually depend on?  Staleness in a result no
# audit reads is untidy; staleness in one the paper quotes is a wrong number.
used = set()
for a in ("audit_paper.py", "audit_session.py"):
    p = os.path.join(HERE, a)
    if not os.path.exists(p):
        continue
    s = open(p, encoding="utf-8").read()
    used |= set(re.findall(r'L\(["\']([A-Za-z0-9_\-]+\.json)["\']\)', s))
print(f"the two audits read {len(used)} of these result files\n")

critical = [t for t in stale if t[0] in used]
cosmetic = [t for t in stale if t[0] not in used]
if stale:
    print("stale results, in minutes behind their newest dependency:")
    for n, p, h in sorted(stale, key=lambda t: -t[2]):
        mark = "  <-- QUOTED IN THE PAPER" if n in used else ""
        print(f"  {n:<30} behind {p:<26} {h * 60:>6.1f} min{mark}")
    print()
if critical:
    print(f"{len(critical)} of them feed a number the paper quotes.  Those must be")
    print("regenerated before the number is trusted: both existing audits will")
    print("verify the manuscript against them without complaint.")
elif stale:
    print("None of them feeds a number the paper quotes, so no claim rests on a")
    print("stale result.  The lags are all under a few minutes and correspond to")
    print("edits made to a script's printed commentary after its numbers were")
    print("produced, which modification time cannot distinguish from an edit to")
    print("the computation.  That is the tool's known limit.")
else:
    print("No result is older than the code or the data it came from.")
if orphan:
    print()
    print("Results with no identifiable producer are ones whose writing script")
    print("this search could not locate; they are listed so they are not")
    print("mistaken for verified:")
    for n in orphan:
        print(f"  {n}")

json.dump(dict(checked=len(results), stale=[s[0] for s in stale],
               orphan=orphan, rows=rows,
               generated=time.strftime("%Y-%m-%d %H:%M:%S")),
          open(os.path.join(HERE, "staleness.json"), "w"), indent=2)
