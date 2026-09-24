r"""A fourth audit: the manuscript's own cross-references and bibliography.

The three existing audits check numbers.  Two failures live outside their reach
because they are invisible to LaTeX itself.

The first is a control sequence whose backslash has been eaten.  Editing the
manuscript through a shell heredoc turns the \r of \ref into a carriage return,
leaving `Section~` followed by the plain word `ef{sec:related}`.  LaTeX has no
complaint to make: that is ordinary text, it typesets as "Section efsec:related",
and the compile reports zero undefined references.  Three had survived in this
manuscript and were found only by reading a random page.

The second is a bibliography that has grown past the text that uses it.  An
entry no \cite reaches is padding, and a section that cites nothing is where a
reviewer looks for the surrounding literature and finds the author alone.

Neither is a wrong number, so neither would ever be flagged elsewhere.
"""
import io
import os
import re

TEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "paper", "mssp.tex")
# The supplement cites into the same bibliography, so an entry cited only
# there is cited, not orphaned.  See _source.py.
import _source
s = _source.read_both().replace(chr(13), "")
bad = 0

# --- control sequences whose backslash was consumed --------------------------
# The tell is a break where none belongs, immediately followed by the tail of a
# known command and its brace.  A genuine word ending in these letters is never
# preceded by a bare line break and followed by "{".
STEM = {"ef": "ref", "ite": "cite", "abel": "label", "aragraph": "paragraph",
        "extit": "textit", "extbf": "textbf", "mph": "emph",
        "ection": "section"}
PAT = re.compile(r"(?<![\\A-Za-z])[\r\n](" + "|".join(STEM) + r")\{")
hits = [(s[:m.start()].count("\n") + 1, m.group(1)) for m in PAT.finditer(s)]
print("eaten backslashes: %s" % (("%d" % len(hits)) if hits else "none"))
for ln, tail in hits:
    print("  line %-6d reads %-10s should be \\%s" % (ln, tail + "{", STEM[tail]))
bad += len(hits)

# A stray carriage return inside a line is the same fault in another guise.
stray = sum(1 for ln in s.split("\n") if "\r" in ln)
if stray:
    print("lines with an embedded carriage return: %d" % stray)
bad += stray

# --- bibliography against the text -------------------------------------------
defined = re.findall(r"\\bibitem\{([^}]*)\}", s)
cited = []
for m in re.finditer(r"\\cite\{([^}]*)\}", s):
    cited += [k.strip() for k in m.group(1).replace("\n", "").split(",")
              if k.strip()]

uncited = [k for k in defined if k not in set(cited)]
undefined = sorted({k for k in cited if k not in set(defined)})
dupes = sorted({k for k in defined if defined.count(k) > 1})
print()
print("%d bibitems, %d cite commands, %d distinct keys cited"
      % (len(defined), s.count("\\cite{"), len(set(cited))))
for label, bag in (("uncited entries", uncited), ("undefined keys", undefined),
                   ("duplicate keys", dupes)):
    print("%-16s: %s" % (label, ", ".join(bag) if bag else "none"))
bad += len(uncited) + len(undefined) + len(dupes)

# --- sections standing alone --------------------------------------------------
# The back matter is not expected to cite, and a section under two hundred words
# reporting a null result is not either; everything else is.
# An appendix of proofs is exempt for a stated reason and not by habit: it
# derives results the body has already positioned against the literature,
# so a citation there would point at the section it came from.
# The supplement is exempt as a whole.  Its sections hold the derivations and
# the per-record measurements behind claims the manuscript has already placed
# against the literature, so a citation in one of them would point back at the
# section it supports rather than outward.  This check is about the manuscript,
# so it is run on the manuscript alone even though the citation counts above
# are taken over both files.
BACK = ("Data availability", "Verification", "Conclusion", "Conclusions",
        "Omitted proofs", "Derivations of results stated in the body")
_main = io.open(TEX, encoding="utf-8").read()
secs = [(m.start(), m.group(1))
        for m in re.finditer(r"\\section\{([^}]*)\}", _main)]
bare = []
for i, (pos, name) in enumerate(secs):
    end = secs[i + 1][0] if i + 1 < len(secs) else len(_main)
    seg = _main[pos:end]
    words = len(re.sub(r"[{}$\\]", " ", re.sub(r"\\[a-zA-Z]+", " ", seg)).split())
    if name not in BACK and "\\cite{" not in seg and words > 200:
        bare.append("%s (%d words)" % (name, words))
print()
print("%d numbered sections; %d substantial ones cite nothing%s"
      % (len(secs), len(bare), (": " + "; ".join(bare)) if bare else ""))
bad += len(bare)

# --- a citation group prints in the order it is written ----------------------
# This bibliography is written out by hand, so \cite{a,b} prints the numbers in
# the order the keys are given, not in ascending order.  Six reviews cited as
# one group printed "[7, 8, 9, 10, 6, 5]" on the first page of the
# introduction.  Nothing is wrong with the references and nothing is undefined;
# it simply reads as carelessness, which on the first page is expensive.
_order = {k: i for i, k in enumerate(re.findall(r"bibitem\{([a-z0-9]+)\}", s))}
_flat = re.sub(r"\s+", "", s)
_unsorted = []
for _m in re.finditer(r"cite\{([^}]*)\}", _flat):
    _keys = _m.group(1).split(",")
    _n = [_order.get(k, -1) for k in _keys]
    if len(_keys) > 1 and _n != sorted(_n):
        _unsorted.append((_keys, _n))
print()
if _unsorted:
    print("%d citation group(s) would print out of numerical order:"
          % len(_unsorted))
    for _keys, _n in _unsorted:
        print("   " + ",".join(_keys) + "  ->  "
              + ", ".join(str(x + 1) for x in _n))
    bad += len(_unsorted)
else:
    print("every multi-key citation prints in ascending order")

# --- the bibliography is in order of first citation --------------------------
# The check above reads inside one \cite; this one reads across the whole
# manuscript, and it is the one that was missing.  _reorder_refs.py sorts the
# \bibitem blocks by first citation, and any edit that moves a paragraph or
# adds a citation to an early one silently undoes that: twice in one session a
# new opening paragraph made the first citation on page one print as [25], and
# every layer passed.  Elsevier's numbered style is by order of appearance, so
# this is a defect a copy-editor would return.
_bibi = {k: i for i, k in enumerate(re.findall(r"bibitem\{([a-z0-9]+)\}", s))}
_seen = []
for _m in re.finditer(r"\\cite[a-z]*(?:\[[^\]]*\])?\{([^}]*)\}",
                      _main[:_main.find("\\begin{thebibliography}")]):
    for _k in _m.group(1).split(","):
        _k = _k.strip()
        if _k and _k not in _seen:
            _seen.append(_k)
_pairs = [(a, b) for i, a in enumerate(_seen) for b in _seen[i + 1:]
          if _bibi.get(a, 10 ** 9) > _bibi.get(b, 10 ** 9)]
print()
if _pairs:
    print("%d citation pair(s) appear in the wrong numerical order; the first "
          "citation\nin the manuscript prints as [%d]. Run _reorder_refs.py."
          % (len(_pairs), _bibi.get(_seen[0], -1) + 1))
    for a, b in _pairs[:6]:
        print("   %s [%d] is cited before %s [%d]"
              % (a, _bibi.get(a, -1) + 1, b, _bibi.get(b, -1) + 1))
    bad += 1
else:
    print("the bibliography is in order of first citation")

# --- what has not been checked by a human ------------------------------------
# A reference can be perfectly formed, cited exactly once, resolve without a
# warning, and still name the wrong volume.  No audit can detect that; only
# reading the source can.  This layer therefore does not check the entries, it
# reports which ones nobody has checked, so that the gap stays visible instead
# of being assumed away.  It is printed rather than counted as a problem for the
# same reason the unfilled author declarations are: a missing check is not a
# wrong entry, it is an absent guarantee.
_unv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "refs_unverified.txt")
if os.path.exists(_unv):
    lines = [x.strip() for x in io.open(_unv, encoding="utf-8")
             if x.strip() and not x.startswith("#")]
    memory = [x[1:] for x in lines if x.startswith("!")]
    pending = [x.lstrip("!") for x in lines]
    unknown = [k for k in pending if k not in defined]
    print()
    print("%d of %d entries are not yet confirmed against the publisher's "
          "record," % (len(pending), len(defined)))
    # The from-memory list is empty now that all eleven have been checked, and
    # an empty list printed under a colon reads as a missing answer.
    if memory:
        print("and %d of those were written from memory rather than copied "
              "from a source:" % len(memory))
        print("   " + ", ".join(memory))
    else:
        print("none of them was written from memory; the eleven that were have "
              "all been checked.")
    if unknown:
        print("   listed as pending but not in the bibliography: "
              + ", ".join(unknown))
        bad += len(unknown)

print()
print("%d problems" % bad if bad else "no problems")
raise SystemExit(1 if bad else 0)
