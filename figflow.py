r"""The road map: what the paper replaces, with what, and what follows.

The paper has five figures and every one of them is a measurement.  A reader
meeting forty-four pages and thirty-one propositions has nothing that shows the
shape of the argument, and the introduction's claim that the parts are "four
stages of one argument" is asserted in prose and nowhere drawn.

This builds that drawing.  It is a standalone TikZ document compiled to
figflow.pdf so that the manuscript's preamble is untouched and the figure is
included exactly as the other five are.

Section numbers are read out of mssp.aux rather than typed, so the figure cannot
drift from the document it points into.  If a number is missing the build stops
rather than printing a wrong pointer.
"""
import io
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(os.path.dirname(HERE), "paper")
AUX = os.path.join(PAPER, "mssp.aux")
OUT = os.path.join(PAPER, "figflow.pdf")

PDFLATEX = r"D:\texlive\2026\bin\windows\pdflatex.exe"


def section_numbers():
    """label -> printed number, from the aux file the manuscript just produced."""
    s = io.open(AUX, encoding="utf-8", errors="replace").read()
    out = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", s):
        out[m.group(1)] = m.group(2)
    return out


TEMPLATE = r"""
% standalone.cls and preview.sty are absent from this installation, so the page
% is cut to the picture by hand: geometry sets the paper to the drawing's own
% bounding box, which the \useasboundingbox below fixes explicitly.
\documentclass{article}
\usepackage[paperwidth=164mm,paperheight=104mm,margin=0mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[cmintegrals]{newtxmath}
\renewcommand{\rmdefault}{ptm}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,fit,backgrounds}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\topskip}{0pt}
\setlength{\parskip}{0pt}
\hyphenpenalty=10000
\exhyphenpenalty=10000

% the palette of figstyle.py, so the road map cannot look like a different paper
\definecolor{cLin}{HTML}{8C8C8C}
\definecolor{cPow}{HTML}{D55E00}
\definecolor{cExp}{HTML}{0072B2}
\definecolor{cStr}{HTML}{AA3377}
\definecolor{cAcc}{HTML}{009E73}

\begin{document}
\begin{tikzpicture}[x=1mm,y=1mm,
  font=\fontsize{6.6}{7.6}\selectfont,
  bx/.style 2 args={draw=#1,fill=#1!7,line width=0.45pt,rounded corners=0.6pt,
                    align=center,inner sep=1.6pt,text width=#2,minimum height=7},
  ar/.style={-{Stealth[length=1.5mm,width=1.1mm]},line width=0.45pt,draw=cLin},
  arb/.style={-{Stealth[length=1.5mm,width=1.1mm]},line width=0.5pt,draw=cExp},
  tag/.style={font=\fontsize{5.6}{6.4}\selectfont,text=cLin,inner sep=0.6pt},
  note/.style={font=\fontsize{5.9}{6.9}\selectfont,align=center,text=cLin,
               inner sep=1pt},
  lane/.style={draw=#1!45,fill=#1!4,line width=0.4pt,rounded corners=1pt},
]
\useasboundingbox (0,3) rectangle (164,107);

% ============================ the usual pipeline ============================
\node[tag,anchor=west,text=cLin] at (2,106) {\textsc{the usual pipeline}};
\node[bx={cLin}{30mm}] (u1) at (19,100)  {vibration or\\sensor record};
\node[bx={cLin}{30mm}] (u2) at (61,100)  {health\\indicator};
\node[bx={cLin}{30mm}] (u3) at (103,100) {trained\\RUL model};
\node[bx={cLin}{30mm}] (u4) at (145,100) {reported\\error};
\draw[ar] (u1) -- (u2);  \draw[ar] (u2) -- (u3);  \draw[ar] (u3) -- (u4);
\node[note,text width=150mm] at (82,91.4)
  {the indicator is scored through one model's accuracy on one fleet, so the two
   are judged jointly; screening instead by monotonicity, trendability or
   prognosability certifies good behaviour, and misses a difference of $0.78$ of
   a lifetime in when an indicator reaches its information~(\S@sec:criteria@)};

% ================================= the pivot =================================
\draw[arb] (82,86.6) -- (82,82.2);
\node[anchor=west,font=\fontsize{6.6}{7.6}\selectfont\itshape,text=cExp]
  at (85,84.4) {what does the indicator alone determine?};

% ============================== the instrument ==============================
\node[tag,anchor=west,text=cExp] at (2,80.5) {\textsc{the instrument}};
\node[bx={cExp}{40mm}] (i1) at (30,74.5)
  {the damage clock $\delta$ as the parameter\\$X_k=D(\tau_k-\delta)+\varepsilon_k$};
\node[bx={cExp}{36mm}] (i2) at (82,74.5)
  {information density\\$g=D'(\tau)^{2}/\sigma(\tau)^{2}$};
\node[bx={cExp}{40mm}] (i3) at (134,74.5)
  {cumulative curve\\$G(\tau)=\sum_{\tau_k\le\tau}g(\tau_k)$};
\draw[arb] (i1) -- (i2);  \draw[arb] (i2) -- (i3);
\node[tag,anchor=west] at (12,69.4) {\S@sec:estimand@};
\node[tag,anchor=west] at (66,69.4) {Eq.~(@eq:fisher@)};
\node[tag,anchor=west] at (116,69.4) {Eq.~(@eq:G@)};

% the bus that fans the curve into the three design quantities
\draw[line width=0.5pt,draw=cExp] (134,71) -- (134,66.4);
\draw[line width=0.5pt,draw=cExp] (30,66.4) -- (134,66.4);
\foreach \x in {30,82,134} \draw[arb] (\x,66.4) -- (\x,63.2);

\node[bx={cExp}{46mm}] (q1) at (30,59.5)
  {$\tau_{\min}$: earliest age at which\\a stated precision is reachable};
\node[bx={cExp}{46mm}] (q2) at (82,59.5)
  {$w^{\ast}$: shortest window\\that reaches it};
\node[bx={cExp}{46mm}] (q3) at (134,59.5)
  {$\eta(c)$: what a record\\censored at $c$ still offers};
\node[note,text width=150mm,text=cExp] at (82,52.6)
  {one value, one quantile and one increment of a single curve
   (Proposition~@prop:main@): no shape is assumed of the degradation and no
   parameter is fitted};

% ================================ two lanes =================================
\draw[arb] (41,49.5) -- (41,46.3);
\draw[arb] (123,49.5) -- (123,46.3);

\begin{scope}[on background layer]
  \draw[lane={cAcc}] (2,14.5) rectangle (80,45.5);
  \draw[lane={cStr}] (84,14.5) rectangle (162,45.5);
\end{scope}
\node[font=\fontsize{6.2}{7}\selectfont\itshape,text=cAcc] at (41,43)
  {is the instrument trustworthy?};
\node[font=\fontsize{6.2}{7}\selectfont\itshape,text=cStr] at (123,43)
  {what does it let one ask?};

\node[bx={cAcc}{68mm}] (v1) at (41,37)
  {reading the curve from a finite, noisy record~(\S@sec:estimating@)};
\node[bx={cAcc}{68mm}] (v2) at (41,29)
  {three corrections the residuals force: tail, serial,
   derivative~(\S@sec:measures@)};
\node[bx={cAcc}{68mm}] (v3) at (41,21)
  {met on $370$ records and $102\,464$ cells to $9\%$~(\S@sec:attained@);
   scope bounded on six sides~(\S@sec:scope@)};
\draw[ar,draw=cAcc] (v1) -- (v2);  \draw[ar,draw=cAcc] (v2) -- (v3);

\node[bx={cStr}{68mm}] (a1) at (123,37)
  {which kind of indicator delivers its information first?~(\S@sec:distribution@)};
\node[bx={cStr}{68mm}] (a2) at (123,29)
  {distributional earlier than amplitude by $0.35$--$0.65$ of a life;
   replicated on a second rig and on $349$ turbofan units};
\node[bx={cStr}{68mm}] (a3) at (123,21)
  {the best indicator in the linear span, and how it
   transfers~(\S@sec:construct@)};
\draw[ar,draw=cStr] (a1) -- (a2);  \draw[ar,draw=cStr] (a2) -- (a3);

% ================================ the answer ================================
\draw[ar,draw=cPow] (41,14.3) -- (66,11.3);
\draw[ar,draw=cPow] (123,14.3) -- (98,11.3);
\node[bx={cPow}{78mm}] (ans) at (82,7.6)
  {the advantage is \emph{transferability}, $0.96$ against $0.18$,\\
   and not the amount of information};

\end{tikzpicture}
\end{document}
"""


def main():
    nums = section_numbers()
    src = TEMPLATE
    missing = []
    for lab in sorted(set(re.findall(r"@([^@]+)@", src))):
        if lab not in nums:
            missing.append(lab)
        else:
            src = src.replace("@%s@" % lab, nums[lab])
    if missing:
        raise SystemExit("labels absent from mssp.aux, refusing to guess: %s"
                         % ", ".join(missing))

    tmp = os.path.join(HERE, "_figflow")
    os.makedirs(tmp, exist_ok=True)
    tex = os.path.join(tmp, "figflow.tex")
    io.open(tex, "w", encoding="utf-8").write(src)
    exe = PDFLATEX if os.path.exists(PDFLATEX) else "pdflatex"
    r = subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error",
                        "figflow.tex"], cwd=tmp, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join(r.stdout.splitlines()[-40:])
        raise SystemExit("pdflatex failed:\n" + tail)
    shutil.copyfile(os.path.join(tmp, "figflow.pdf"), OUT)
    print("wrote", OUT)
    shown = ("sec:criteria", "sec:estimand", "sec:estimating", "sec:measures",
             "sec:attained", "sec:scope", "sec:distribution", "sec:construct")
    for k in shown:
        print("  %-20s -> section %s" % (k, nums[k]))
    # The numbers are drawn into a PDF, where nothing downstream can read them.
    # Recording them beside the figure lets the audit compare what the figure
    # says against what the aux file now holds, which is the only way a figure
    # left stale by a rearrangement gets reported.
    import json as _json
    _json.dump({"numbers": {k: nums[k] for k in shown}},
               io.open(os.path.join(HERE, "figflow.json"), "w",
                       encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
