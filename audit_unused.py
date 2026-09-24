r"""Which computed quantities does nothing ever read?

The manuscript quoted a count and a margin for "the best transferable linear
combination" and the figures behind them were computed against the linear side
fitted on the record it is scored on.  Every numerical audit passed: the pattern
matched the sentence, the number matched the results file.  What was wrong was
which quantity the sentence named, and no layer checks that.

The signal that exposed it is mechanical.  linear_vs_shape.json carried a
transferable age for every bearing, lin_tr, and nothing in the manuscript or in
any audit ever read it.  A results file that computes a quantity nobody reads is
either dead weight or, as there, evidence that a neighbouring quantity is being
quoted in its place.

So this runs the numerical audits with an instrumented loader that records every
key they touch, and reports the keys they never touch, for the files they use.
It is the coverage check of audit_mathematics.py applied to results instead of
propositions: it does not decide whether a sentence is right, it removes the
place where a wrong one can hide.
"""
import io
import os
import re
import sys
import json
import runpy
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
AUDITS = ("audit_paper.py", "audit_session.py")
READ = {}          # file -> set of key paths touched
PRESENT = {}       # file -> set of leaf key paths
_real_load = json.load


class Watched(dict):
    """A dict that records which of its keys are read, recursively."""

    def __init__(self, data, path, seen):
        super().__init__(data)
        self._path, self._seen = path, seen

    def _mark(self, key):
        self._seen.add("%s/%s" % (self._path, key) if self._path else str(key))

    def __getitem__(self, key):
        self._mark(key)
        return _wrap(super().__getitem__(key),
                     "%s/%s" % (self._path, key) if self._path else str(key),
                     self._seen)

    def get(self, key, default=None):
        self._mark(key)
        if key not in self:
            return default
        return self[key]

    def __contains__(self, key):
        self._mark(key)
        return super().__contains__(key)

    # Iterating a mapping reads all of it.  Without these three the layer
    # reported quantities as unread whenever an audit looped over a file
    # instead of indexing it, which was most of them: criteria_resolution.json
    # came back nine keys present and none read, from an audit that consumes it
    # entirely.
    def items(self):
        for k in list(super().keys()):
            self._mark(k)
        return [(k, self[k]) for k in super().keys()]

    def values(self):
        return [v for _, v in self.items()]

    def keys(self):
        for k in list(super().keys()):
            self._mark(k)
        return super().keys()

    def __iter__(self):
        for k in list(super().keys()):
            self._mark(k)
        return super().__iter__()


def _wrap(v, path, seen):
    if isinstance(v, dict):
        return Watched(v, path, seen)
    if isinstance(v, list):
        return [_wrap(x, path, seen) for x in v]
    return v


def _leaves(v, path, out):
    """Every leaf key path in a result file.

    A list of records is collapsed onto one path: fifteen bearings with the same
    fields are one quantity read fifteen times, not fifteen quantities, and
    counting them separately would bury the signal in per-record noise.
    """
    if isinstance(v, dict):
        for k, x in v.items():
            p = "%s/%s" % (path, k) if path else str(k)
            if isinstance(x, (dict, list)):
                _leaves(x, p, out)
            else:
                out.add(p)
    elif isinstance(v, list):
        for x in v[:1]:
            _leaves(x, path, out)
        if not v or not isinstance(v[0], (dict, list)):
            out.add(path)


def loader(fp, *a, **k):
    name = os.path.basename(getattr(fp, "name", "?"))
    data = _real_load(fp, *a, **k)
    if not name.endswith(".json"):
        return data
    seen = READ.setdefault(name, set())
    pres = PRESENT.setdefault(name, set())
    _leaves(data, "", pres)
    return _wrap(data, "", seen)


print("=" * 92)
print("Quantities the audits compute and never read")
print("=" * 92)
print("""
Each numerical audit is run with a loader that records every key it touches.  A
key present in a results file the audits use, and never touched, is a quantity
the manuscript's checks do not reach.
""")
json.load = loader
for a in AUDITS:
    p = os.path.join(HERE, a)
    if not os.path.exists(p):
        continue
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(p, run_name="__not_main__")
    except SystemExit:
        pass
    except Exception as e:                      # a broken audit is reported,
        print("  %s raised %s: %s" % (a, type(e).__name__, e))
json.load = _real_load

print("  %-32s %8s %8s %8s" % ("results file", "keys", "read", "unread"))
print("  " + "-" * 62)
rows, total_unread = [], 0
for name in sorted(PRESENT):
    pres, seen = PRESENT[name], READ.get(name, set())
    touched = {p for p in pres
               if p in seen or any(s.startswith(p + "/") or p.startswith(s + "/")
                                   for s in seen)}
    unread = sorted(pres - touched)
    if not pres:
        continue
    total_unread += len(unread)
    rows.append(dict(file=name, keys=len(pres), read=len(touched),
                     unread=unread))
    print("  %-32s %8d %8d %8d" % (name, len(pres), len(touched), len(unread)))

OUT = dict(files=len(rows), unread_total=total_unread, detail=rows)
print("""
  %d quantities across %d files are computed and never read.  Most are
  intermediates and listing them all would be a list nobody reads, so the report
  below narrows to the shape the failure actually had.
""" % (total_unread, sum(1 for r in rows if r["unread"])))

# --- the shape that matters: a sibling read, a sibling ignored ---------------
# lin_own and lin_tr sit side by side under per_unit.  One was quoted, the other
# was never touched, and the sentence named the second while the check read the
# first.  A group where some children are read and others are not is where that
# can happen; a group nothing reads is an intermediate and a group read entirely
# is accounted for.
print("=" * 92)
print("Siblings of quantities the audits do read, that they do not")
print("=" * 92)
print("""
Grouped by their parent.  Each line is a quantity computed beside one the
manuscript's checks consume, and left alone: the place where a sentence can name
one and a check read the other.
""")
flagged = []
for r in rows:
    pres, seen = PRESENT[r["file"]], READ.get(r["file"], set())
    def touched(p):
        return p in seen or any(s.startswith(p + "/") or p.startswith(s + "/")
                                for s in seen)
    groups = {}
    for p in pres:
        groups.setdefault(p.rsplit("/", 1)[0] if "/" in p else "", []).append(p)
    for parent, kids in sorted(groups.items()):
        if len(kids) < 2:
            continue
        read_k = [k for k in kids if touched(k)]
        idle = [k for k in kids if not touched(k)]
        if read_k and idle:
            flagged.append(dict(file=r["file"], parent=parent,
                                read=sorted(read_k), unread=sorted(idle)))
for f in sorted(flagged, key=lambda f: (-len(f["unread"]), f["file"])):
    print("  %s  [%s]" % (f["file"], f["parent"] or "top level"))
    print("      read:   %s" % ", ".join(k.rsplit("/", 1)[-1]
                                         for k in f["read"][:8]))
    print("      idle:   %s" % ", ".join(k.rsplit("/", 1)[-1]
                                         for k in f["unread"][:12]))
OUT["siblings"] = flagged
OUT["sibling_groups"] = len(flagged)
OUT["sibling_quantities"] = sum(len(f["unread"]) for f in flagged)
print("""
  %d groups, holding %d quantities, sit beside a quantity the audits consume and
  are never consumed themselves.  Each is a candidate for the failure that
  motivated this layer, and each has to be looked at once by a person: either the
  sentence names the right sibling, or it does not.
""" % (OUT["sibling_groups"], OUT["sibling_quantities"]))

json.dump(OUT, open(os.path.join(HERE, "audit_unused.json"), "w"), indent=2,
          default=str)
print("written to audit_unused.json")
