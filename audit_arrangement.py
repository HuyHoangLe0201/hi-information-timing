r"""The order of the sections as an optimisation, solved exactly.

Reorganising a paper is usually argued.  It can instead be posed and solved.

THE MODEL.  Let the body sections be a set S.  Each cross-reference in section u
to a label defined in section v is an edge u -> v with unit weight, so w(u,v) is
the number of times u asks the reader to know something from v.  An ordering pi
of S makes an edge FORWARD when pi(v) > pi(u): the reader meets the claim before
the thing it rests on.  The cost of an ordering is the total weight of its
forward edges,

    C(pi)  =  sum over pairs u,v of  w(u,v) . 1[ pi(u) < pi(v) ],

and the arrangement problem is to minimise C.  This is the minimum weighted
feedback arc set, NP-hard in general; with fifteen sections it is small enough
to solve EXACTLY by dynamic programming over subsets, so what follows is an
optimum and not a heuristic.

    dp[M] = least forward weight among edges internal to M, when M is a prefix.
    dp[M + v] = min over v not in M of  dp[M] + sum over u in M of w(u,v),

because placing v after every member of M makes exactly the edges from M to v
forward.  Held--Karp over 2^|S| subsets.

WHAT THE MODEL DOES NOT KNOW.  Topical coherence, the convention that a paper
opens with an introduction and closes with a conclusion, and the fact that a
reader will tolerate a forward reference in a roadmap but not in an argument.
The first is why the optimum is read as evidence rather than obeyed; the second
and third are imposed as constraints.  The introduction is pinned first and the
conclusion last, and the discussion immediately before it.

The number worth reporting is not the optimum alone but the GAP: how much of the
achievable improvement the present order already has.
"""
import io
import os
import re
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "mssp.tex")
s = io.open(TEX, encoding="utf-8").read()
s = re.sub(r"(?m)^%.*$", "", s)
# The appendix is not part of the ordering problem: it is read on demand
# from the propositions it proves, not in sequence, so it is cut here.
_app = s.find(chr(92) + "appendix")
if _app > 0:
    s = s[:_app]

# --- the sections, in the order they now stand -------------------------------
secs = []
for m in re.finditer(r"\\section\{([^}]*)\}", s):     # numbered sections only
    secs.append([m.start(), m.group(1), 0])
for i, x in enumerate(secs):
    x[2] = secs[i + 1][0] if i + 1 < len(secs) else len(s)
names = [x[1] for x in secs]
n = len(names)
idx = {t: i for i, t in enumerate(names)}

# --- where every label lives -------------------------------------------------
owner = {}
for m in re.finditer(r"\\label\{([^}]*)\}", s):
    for i, (a, t, b) in enumerate(secs):
        if a <= m.start() < b:
            owner.setdefault(m.group(1), i)
            break

# --- the weighted digraph ----------------------------------------------------
# Floats are excluded: a figure or table is placed by the typesetter, so citing
# one before it appears is not a demand on the reader's memory.
#
# Two edge sets, because the first objective tried here was wrong and the
# optimum said so.  Counting EVERY cross-reference put the construction ninth
# and the scope section second, which is absurd, and the reason is that the
# model treated two different things alike.  A reference to a proposition or an
# equation is DEFINITIONAL: the argument does not go through without it.  A
# reference to a section is usually a POINTER -- "this error is measured in
# Section 11" -- and a reader loses nothing by meeting it early.  Minimising
# pointers rewards moving a section that points forward to the back, which is
# why the naive optimum buried the construction.
def build(kinds):
    M = [[0] * n for _ in range(n)]
    for i, (a, t, b) in enumerate(secs):
        for m in re.finditer(r"\\(?:eq)?ref\{([^}]*)\}", s[a:b]):
            tgt = m.group(1)
            if not tgt.startswith(kinds):
                continue
            j = owner.get(tgt)
            if j is None or j == i:
                continue
            M[i][j] += 1
    return M


ALL = build(("sec:", "prop:", "eq:", "rem:"))
DEF = build(("prop:", "eq:", "rem:"))
W = ALL


def cost(order, M=None):
    M = W if M is None else M
    pos = {v: k for k, v in enumerate(order)}
    return sum(M[u][v] for u in range(n) for v in range(n)
               if M[u][v] and pos[u] < pos[v])


def solve(M, head, tail, before=()):
    """Exact minimum-forward-weight order.

    `head` and `tail` are sections pinned at the front and back, in order; the
    middle is optimised over every permutation by dynamic programming on
    subsets.  `before` is a list of (u, v) pairs requiring u to precede v, which
    is how a hard editorial constraint enters: the corpus must be described
    before the sections that measure on it.
    """
    free = [i for i in range(n) if i not in head and i not in tail]
    k = len(free)
    at = {v: j for j, v in enumerate(free)}
    need = {}                       # bit -> mask of bits that must precede it
    for u, v in before:
        if u in at and v in at:
            need[at[v]] = need.get(at[v], 0) | (1 << at[u])
    dp = [None] * (1 << k)
    dp[0] = (0, ())
    for mask in range(1 << k):
        cur = dp[mask]
        if cur is None:
            continue
        placed = list(head) + [free[j] for j in range(k) if mask >> j & 1]
        for j in range(k):
            if mask >> j & 1:
                continue
            if need.get(j, 0) & ~mask:
                continue            # a required predecessor is not placed yet
            v = free[j]
            add = sum(M[u][v] for u in placed)
            nm = mask | (1 << j)
            cand = (cur[0] + add, cur[1] + (v,))
            if dp[nm] is None or cand[0] < dp[nm][0]:
                dp[nm] = cand
    end = dp[(1 << k) - 1]
    if end is None:
        return None
    return list(head) + list(end[1]) + list(tail)



cur = list(range(n))
tot_all, tot_def = sum(sum(r) for r in ALL), sum(sum(r) for r in DEF)
print("=" * 92)
print("The arrangement problem, solved exactly")
print("=" * 92)
print("""
Every cross-reference between two sections is an edge, and the cost of an order
is the weight of the references that point forwards.  Floats are excluded.
""")
print("  sections %d" % n)
print("  all references          : %3d edges, weight %3d"
      % (sum(1 for u in range(n) for v in range(n) if ALL[u][v]), tot_all))
print("  definitional only       : %3d edges, weight %3d"
      % (sum(1 for u in range(n) for v in range(n) if DEF[u][v]), tot_def))

# Sections are found by a distinctive word rather than by their full title:
# the titles now carry an MSSP-style functional label in front of the phrase
# ("Theoretical foundation: the information budget"), and a lookup on the exact
# string broke the moment that label was added.
def _find(word):
    for t in names:
        if word.lower() in t.lower():
            return idx[t]
    raise KeyError(word)


INTRO = _find("Introduction")
DISC = _find("Discussion")
CONC = _find("Conclusion")
DATA = _find("Data and protocol")
measuring = [idx[t] for t in names
             if not any(w in t.lower() for w in
                        ("introduction", "information budget",
                         "data and protocol"))]
BEFORE = [(DATA, v) for v in measuring]

print("\n" + "=" * 92)
print("1.  Minimising every reference, which is the wrong objective")
print("=" * 92)
naive = solve(ALL, (INTRO,), (DISC, CONC))
print("  present %d, optimum %d" % (cost(cur, ALL), cost(naive, ALL)))
print("  that optimum places:")
for k in (1, 2, 8):
    print("     position %-3d %s" % (k + 1, names[naive[k]]))
print("""
  The construction lands ninth and the scope section second.  The model is at
  fault, not the paper: it counts a pointer -- "this error is measured in
  Section 11" -- the same as a dependency, so it rewards pushing any section
  that points forward towards the back.  A reader loses nothing by meeting a
  pointer early and everything by meeting a proposition late.
""")

print("=" * 92)
print("2.  Minimising definitional references only")
print("=" * 92)
best = solve(DEF, (INTRO,), (DISC, CONC))
print("  present %d, optimum %d, of a total definitional weight %d"
      % (cost(cur, DEF), cost(best, DEF), tot_def))

print("\n" + "=" * 92)
print("3.  With the editorial constraint the last revision established")
print("=" * 92)
con = solve(DEF, (INTRO,), (DISC, CONC), before=BEFORE)
print("  the corpus must precede every section that measures on it")
print("  present %d, constrained optimum %d" % (cost(cur, DEF), cost(con, DEF)))
print("\n  %-4s %-40s %-40s" % ("", "present", "constrained optimum"))
print("  " + "-" * 86)
for k in range(n):
    mark = "" if cur[k] == con[k] else "   <-"
    print("  %-4d %-40s %-40s%s"
          % (k + 1, names[cur[k]][:38], names[con[k]][:38], mark))
moved = sum(1 for k in range(n) if cur[k] != con[k])
gap = cost(cur, DEF) - cost(con, DEF)
print("""
  %d of %d positions differ and the gap is %d references of %d.  The present
  order therefore already realises %.0f per cent of what any admissible
  reordering could achieve on the dependencies that matter.

  The optimum is evidence about structure, not an instruction.  It knows which
  section needs which and nothing about what belongs beside what, so where it
  disagrees the question to ask is whether the dependency it found is real.
""" % (moved, n, gap, tot_def,
       100.0 * (1 - gap / max(tot_def, 1))))

# --- and the actionable part -------------------------------------------------
# A gap of eight is not an instruction to reorder ten sections.  It is eight
# named places where a section uses a result stated later, and each can be
# answered on its own: move the proposition, restate it, or accept the promise.
print("=" * 92)
print("4.  The definitional dependencies that run backwards, one by one")
print("=" * 92)
pos = {v: k for k, v in enumerate(cur)}
worst = []
for i, (a, t, b) in enumerate(secs):
    for m in re.finditer(r"\\(?:eq)?ref\{([^}]*)\}", s[a:b]):
        tgt = m.group(1)
        if not tgt.startswith(("prop:", "eq:", "rem:")):
            continue
        j = owner.get(tgt)
        if j is None or j == i or pos[j] <= pos[i]:
            continue
        ctx = re.sub(r"\s+", " ", s[max(a, a + m.start() - 90):
                                    a + m.end() + 30]).strip()
        worst.append((names[i], tgt, names[j], ctx))
print("  %d found\n" % len(worst))
for src_t, tgt, dst_t, ctx in worst:
    print("  %s" % src_t)
    print("     needs %-22s from %s" % (tgt, dst_t))
    print("     ...%s..." % ctx[-110:])
    print()
