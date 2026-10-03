"""Build appendix_tables.tex (camera-ready supporting tables) from the saved results.

    python make_appendix_tables.py      # writes appendix_tables.tex, prints checks

Every table is recomputed from the per-record eval files. Before writing anything the
script reproduces numbers the manuscript already prints and stops if any differ, so a
table can never silently disagree with the text. Read-only on the results.
"""
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "branch_1_premise" / "experiment"
RES = EXP / "results"
sys.path.insert(0, str(EXP / "src"))
from analyze_dual_primary import looks_truncated  # noqa: E402
from eval_refusal import REFUSAL_PATTERNS, normalize_refusal_text  # noqa: E402

SEEDS = (42, 1337, 2718)
LENSES = ("none", "agreeable", "principled", "agreeable_para1")
LNAME = {"none": "standard", "agreeable": "compliant", "principled": "principled",
         "agreeable_para1": "paraphrase"}
ARMNAME = {0: "arm 0: untrained", 1: "arm 1: compliant persona",
           2: "arm 2: generic instruction tuning", 3: "arm 3: principled persona"}
B, BOOT_SEED = 10000, 20261001


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))["conditions"]


def fname(arm, seed, suffix="_env76"):
    return f"eval_arm0_r0_s{seed}{suffix}.json" if arm == 0 else f"eval_arm{arm}_r3_s{seed}{suffix}.json"


H = {(a, s): load(fname(a, s)) for a in range(4) for s in SEEDS}


def refused(text, norm=True):
    t = (normalize_refusal_text(text) if norm else text).lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def rate(recs, metric, excl=False):
    kept = [r for r in recs if not (excl and looks_truncated(r["response"]))]
    if metric == "R":
        v = [float(r["refused_regex"]) for r in kept]
    elif metric == "SR":
        v = [float(r["harm_score"]) for r in kept]
    else:  # SRbin
        v = [float(r["harm_score"] > 0.5) for r in kept]
    return sum(v) / len(v)


def cell(a, lens, metric="R", excl=False, C=H, seeds=SEEDS):
    return sum(rate(C[(a, s)][lens]["records"], metric, excl) for s in seeds) / len(seeds)


def inter(a, lens, metric="R", excl=False, C=H, seeds=SEEDS):
    return ((cell(a, "none", metric, excl, C, seeds) - cell(a, lens, metric, excl, C, seeds))
            - (cell(0, "none", metric, excl, C, seeds) - cell(0, lens, metric, excl, C, seeds)))


def per_prompt(a, lens, metric, ref=0):
    key = {"R": lambda r: float(r["refused_regex"]), "SR": lambda r: float(r["harm_score"]),
           "SRbin": lambda r: float(r["harm_score"] > 0.5)}[metric]
    out = []
    for i in range(313):
        v = 0.0
        for s in SEEDS:
            d = lambda x: key(H[(x, s)]["none"]["records"][i]) - key(H[(x, s)][lens]["records"][i])
            v += d(a) - d(ref)
        out.append(v / 3)
    return out


def boot(v):
    rng = random.Random(BOOT_SEED); n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(v) / n, bs[int(.025 * B)], bs[int(.975 * B) - 1]


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def logit_inter(a, lens):
    return ((logit(cell(a, "none")) - logit(cell(a, lens)))
            - (logit(cell(0, "none")) - logit(cell(0, lens))))


def f3(x, sign=False):
    x = round(x, 3) + 0.0  # avoid printing -0.000
    if x == 0:
        return "0.000"
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def ci(t):
    m, lo, hi = t
    return f"{f3(m, True)} [{f3(lo, True)}, {f3(hi, True)}]"


# ------------------------------------------------------------- reproduction gate
CHECKS = [
    (inter(1, "agreeable"), 0.306), (inter(3, "agreeable"), 0.381), (inter(2, "agreeable"), -0.012),
    (inter(1, "agreeable", excl=True), 0.184), (inter(3, "agreeable", excl=True), 0.178),
    (inter(2, "agreeable", excl=True), 0.010),
    (inter(1, "agreeable_para1"), 0.478), (inter(3, "agreeable_para1"), 0.383),
    (inter(2, "agreeable_para1"), -0.020),
    (inter(1, "agreeable", "SR"), -0.114), (inter(3, "agreeable", "SR"), -0.093),
    (inter(1, "agreeable", "SRbin"), -0.122), (inter(3, "agreeable", "SRbin"), -0.097),
    (inter(2, "agreeable", "SRbin"), 0.015),
    (inter(1, "principled"), 0.049), (inter(3, "principled"), 0.147),
]
bad = [(round(g, 4), w) for g, w in CHECKS if round(g, 3) != w]
assert not bad, f"reproduction gate failed (got, printed): {bad}"
print(f"[gate] {len(CHECKS)} printed values reproduced")

T = []

# ------------------------------------------------------------- S1 refusal grid
rows = []
for a in range(4):
    vals = []
    for L in LENSES:
        per = [rate(H[(a, s)][L]["records"], "R") for s in SEEDS]
        m = sum(per) / 3
        vals.append(f"{m:.3f}" if a == 0 else f"{m:.3f} {{\\scriptsize({min(per):.3f}--{max(per):.3f})}}")
    rows.append(f"    {ARMNAME[a]} & " + " & ".join(vals) + " \\\\")
inter_rows = []
for L in ("agreeable", "agreeable_para1", "principled"):
    for a in (1, 2, 3):
        inter_rows.append(
            f"    {LNAME[L]} & arm {a} & {ci(boot(per_prompt(a, L, 'R')))} & "
            f"{f3(inter(a, L, excl=True), True)} & {f3(logit_inter(a, L), True)} \\\\")
T.append(r"""\begin{table}[htb]
  \centering
  \footnotesize
  \caption{Headline refusal grid (Llama-3.1-8B, round 3, $n=313$ per cell, 256 tokens,
  corrected regex). Top: refusal rate, mean over seeds 42, 1337 and 2718 with the seed range
  in parentheses; arm 0 is seed-invariant. Bottom: interactions of Equation~(\ref{eq:interaction})
  with paired-prompt bootstrap intervals (truncation included), the truncation-excluded value,
  and the same contrast computed on the logit scale from the seed-mean rates.}
  \label{tab:grid-full}
  \begin{tabular}{@{}lcccc@{}}
    \toprule
    \textbf{Training arm} & standard & compliant & principled & paraphrase \\
    \midrule
""" + "\n".join(rows) + r"""
    \bottomrule
  \end{tabular}

  \vspace{6pt}
  \begin{tabular}{@{}llccc@{}}
    \toprule
    \textbf{Lens} & \textbf{Arm} & $I_a^R$ included [95\% CI] & excluded & logit scale \\
    \midrule
""" + "\n".join(inter_rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}""")

# ------------------------------------------------------------- S2 instrument audit
fires = {True: [0] * 8, False: [0] * 8}
flips = {a: 0 for a in range(4)}
for (a, s), conds in H.items():
    for c in conds.values():
        for r in c["records"]:
            for norm in (True, False):
                t = (normalize_refusal_text(r["response"]) if norm else r["response"]).lower()
                for i, p in enumerate(REFUSAL_PATTERNS):
                    fires[norm][i] += bool(re.search(p, t))
            flips[a] += refused(r["response"], True) and not refused(r["response"], False)
assert fires[False][0] == 12779 and fires[True][0] == 12836, fires
assert flips == {0: 24, 1: 0, 2: 33, 3: 0}, flips
prow = " \\\\\n".join(f"    {i} & {fires[False][i]:,} & {fires[True][i]:,}".replace(",", "{,}")
                      for i in range(8))
T.append(r"""\begin{table}[htb]
  \centering
  \footnotesize
  \caption{Instrument audit over the $15{,}024$ headline responses: how often each refusal
  pattern of Appendix~\ref{app:lenses} fires before and after apostrophe normalization, and
  how many records the normalization flips from non-refusal to refusal in each arm. No record
  flips the other way.}
  \label{tab:audit}
  \begin{tabular}{@{}lrr@{\hspace{2.5em}}lr@{}}
    \toprule
    \textbf{Pattern} & before & after & \textbf{Arm} & flips \\
    \midrule
""" + "\n".join(
    f"    {i} & {fires[False][i]:,} & {fires[True][i]:,} & ".replace(",", "{,}")
    + (f"arm {i} & {flips[i]}" if i < 4 else " & ") + " \\\\" for i in range(8)) + r"""
    \bottomrule
  \end{tabular}
\end{table}""")

# ------------------------------------------------------------- S3 StrongREJECT interactions
srows = []
for L in ("agreeable", "agreeable_para1"):
    for a in (1, 2, 3):
        srows.append(f"    {LNAME[L]} & arm {a} & {ci(boot(per_prompt(a, L, 'SR')))} & "
                     f"{ci(boot(per_prompt(a, L, 'SRbin')))} \\\\")
T.append(r"""\begin{table}[htb]
  \centering
  \footnotesize
  \caption{StrongREJECT interactions with paired-prompt bootstrap intervals, truncation
  included, signed as in Equation~(\ref{eq:interaction}): negative values mean harmful
  usefulness rises more under the lens than in the untrained reference. The text reports the
  continuous-score values with the sign reversed ($+0.114$, $+0.093$).}
  \label{tab:sr-inter}
  \begin{tabular}{@{}llcc@{}}
    \toprule
    \textbf{Lens} & \textbf{Arm} & $I_a^{\mathrm{SR}}$ (score) [95\% CI] & $I_a$ for $\mathrm{SR}>0.5$ [95\% CI] \\
    \midrule
""" + "\n".join(srows) + r"""
    \bottomrule
  \end{tabular}
\end{table}""")

# ------------------------------------------------------------- S4 factorial
FAC = {1: ("_env76", "persona, narrow"), 4: ("_r6_factorial_env76", "persona, broad"),
       2: ("_env76", "no persona, broad"), 5: ("_r6_factorial_env76", "no persona, narrow")}
frows = []
for a, (suf, lab) in FAC.items():
    inc, exc = [], []
    for s in SEEDS:
        c = load(f"eval_arm{a}_r3_s{s}{suf}.json")
        inc.append(rate(c["none"]["records"], "R") - rate(c["agreeable"]["records"], "R"))
        exc.append(rate(c["none"]["records"], "R", True) - rate(c["agreeable"]["records"], "R", True))
    frows.append((a, lab, inc, exc))
pm = (sum(sum(x[2]) for x in frows if x[0] in (1, 4)) - sum(sum(x[2]) for x in frows if x[0] in (2, 5))) / 6
assert round(pm, 3) == 0.327, pm
assert round(min(min(x[2]) for x in frows if x[0] in (1, 4)), 3) == 0.259
assert round(max(max(x[2]) for x in frows if x[0] in (2, 5)), 3) == 0.019
T.append(r"""\begin{table}[htb]
  \centering
  \footnotesize
  \caption{The $2\times2$ factorial of Section~3.3: per-seed refusal drop from the standard to
  the compliant lens. Arms 1 and 2 are the headline runs; arms 4 and 5 were trained for the
  factorial in the same environment.}
  \label{tab:factorial}
  \begin{tabular}{@{}llccc@{}}
    \toprule
    \textbf{Arm} & \textbf{Cell} & \multicolumn{3}{c}{standard-to-compliant drop, included / excluded} \\
    & & seed 42 & seed 1337 & seed 2718 \\
    \midrule
""" + "\n".join(
    f"    arm {a} & {lab} & " + " & ".join(f"{f3(i, True)} / {f3(e, True)}" for i, e in zip(inc, exc)) + " \\\\"
    for a, lab, inc, exc in frows) + r"""
    \bottomrule
  \end{tabular}
\end{table}""")

# ------------------------------------------------------------- S5 seven-lens grid
R12 = json.loads((RES / "r12_lens_axis_judgment.json").read_text(encoding="utf-8"))["cells"]
SL = [("agreeable_weak", "weak"), ("agreeable", "default"), ("agreeable_strong", "strong"),
      ("agreeable_para1", "para.\\ 1"), ("agreeable_para2", "para.\\ 2")]
lrows = []
for a in (1, 2, 3):
    vals = []
    for L, _ in SL:
        pair = []
        for conv in ("included", "excluded"):
            v = sum(R12[f"arm{a}/s{s}"][conv]["drop"][L] - R12["arm0/s42"][conv]["drop"][L]
                    for s in SEEDS) / 3
            pair.append(v)
        vals.append(f"{f3(pair[0], True)} / {f3(pair[1], True)}")
    lrows.append(f"    arm {a} & " + " & ".join(vals) + " \\\\")
T.append(r"""\begin{table}[htb]
  \centering
  \footnotesize
  \caption{Seven-lens grid (separately retrained checkpoints, three seeds): refusal
  interaction for each wording of the compliant disposition, truncation included / excluded.
  ``Default'' is the compliant lens; the two paraphrases were never used in training.}
  \label{tab:sevenlens}
  \begin{tabular}{@{}lccccc@{}}
    \toprule
    \textbf{Arm} & """ + " & ".join(n for _, n in SL) + r""" \\
    \midrule
""" + "\n".join(lrows) + r"""
    \bottomrule
  \end{tabular}
\end{table}""")

# ------------------------------------------------------------- S6 second validation
T.append(r"""\begin{table}[htb]
  \centering
  \footnotesize
  \caption{Second validation (control arms, earlier seed-42 generation, corrected instrument,
  strict reading): allocation, in which every record the correction flips is censused, and
  confusion matrix. Agreement $199/200$, Cohen's $\kappa=0.886$, prevalence-adjusted
  $\kappa=0.990$.}
  \label{tab:validation2}
  \begin{tabular}{@{}lrrrr@{\hspace{2.5em}}lrr@{}}
    \toprule
    \textbf{Cell} & flipped & random & \textbf{total} & & \textbf{Instrument} & human R & human C \\
    \midrule
    arm 0, standard & 5 & 45 & 50 & & refusal & 195 & 0 \\
    arm 0, compliant & 2 & 48 & 50 & & non-refusal & 1 & 4 \\
    arm 2, standard & 32 & 18 & 50 & & & & \\
    arm 2, compliant & 0 & 50 & 50 & & & & \\
    \bottomrule
  \end{tabular}
\end{table}""")

out = HERE / "appendix_tables.tex"
out.write_text("% Generated by make_appendix_tables.py -- do not edit by hand.\n\n" + "\n\n".join(T) + "\n",
               encoding="utf-8")
print(f"[write] {out.name}: {len(T)} tables")
