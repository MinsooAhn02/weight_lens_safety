"""Write camready_tables.tex (Appendix: camera-ready checks R15-R17) from
results/camera_ready_analysis.json, which analyze_camera_ready.py computes from the result files.

    python make_camready_tables.py

Bootstrap outputs are means of per-prompt contrasts, so every value is k/313 (one seed) or
k/939 (three-seed mean); values are snapped back to that grid before printing to avoid double
rounding of the 4-decimal JSON.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "branch_1_premise" / "experiment"


def snap(v, den):
    k = round(v * den)
    assert abs(k - v * den) < 0.1, (v, den)
    return k / den


def f3(v, den):
    x = snap(v, den)
    s = f"{abs(x):.3f}"
    return ("$-" if x < 0 and s != "0.000" else "$") + s + "$"


def ci(t, den):
    return f"{f3(t[0], den)} [{f3(t[1], den)[1:-1]}, {f3(t[2], den)[1:-1]}]".replace("] ", "] ").replace("[", "$[").replace("]", "]$").replace("$$", "")


def build():
    r = subprocess.run([sys.executable, "src/analyze_camera_ready.py"], cwd=EXP, capture_output=True, text=True)
    if r.returncode:
        sys.exit("analyze_camera_ready.py failed:\n" + r.stderr)
    A = json.loads((EXP / "results" / "camera_ready_analysis.json").read_text(encoding="utf-8"))
    R15, R16, R17 = A["R15"], A["R16"], A["R17"]
    assert R15["A1_decision"] == "survives judge change"
    assert R16["reference_holds"] is False and R16["direction_decision"] == "same direction as Llama"
    assert R17["decision"] == "mixed, inconclusive"
    out = []
    # --- Table: judge interactions
    out += [r"\begin{table}[htb]", r"  \centering", r"  \footnotesize",
            r"  \caption{Learned-judge interactions $I_a^U$ on the unsafe rate, headline grid, mean of"
            r" three seeds with $95\%$ paired-prompt bootstrap intervals. Negative means the lens"
            r" raises the unsafe rate more than in arm 0.}",
            r"  \label{tab:judges}",
            r"  \begin{tabular}{@{}llcc@{}}", r"    \toprule",
            r"    Judge & Arm & Compliant lens & Held-out paraphrase \\", r"    \midrule"]
    names = {"llamaguard3": "Llama Guard 3", "harmbench": "HarmBench classifier"}
    for j in ("llamaguard3", "harmbench"):
        for i, a in enumerate((1, 2, 3)):
            lab = names[j] if i == 0 else ""
            out.append(f"    {lab} & {a} & {ci(R15[j][f'I{a}_agreeable'], 939)} & "
                       f"{ci(R15[j][f'I{a}_agreeable_para1'], 939)} \\\\")
        if j == "llamaguard3":
            out.append(r"    \addlinespace")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    # --- Table: judge vs human, per arm
    out += [r"\begin{table}[htb]", r"  \centering", r"  \footnotesize",
            r"  \caption{Learned-judge errors against the human labels of Appendix~\ref{app:validation}"
            r" (earlier seed-42 generation). Arms 0 and 2 contain one human-harmful record each.}",
            r"  \label{tab:judge-human}",
            r"  \begin{tabular}{@{}lcccc@{}}", r"    \toprule",
            r"    & \multicolumn{2}{c}{Llama Guard 3} & \multicolumn{2}{c}{HarmBench classifier} \\",
            r"    \cmidrule(lr){2-3}\cmidrule(l){4-5}",
            r"    Arm & unsafe $\mid$ human refusal & safe $\mid$ human harmful & unsafe $\mid$ human refusal & safe $\mid$ human harmful \\",
            r"    \midrule"]
    for a in ("arm0", "arm1", "arm2", "arm3"):
        cells = []
        for j in ("llamaguard3", "harmbench"):
            p = R15[j]["A2"]["per_arm"][a]
            cells += [f"${p['fp']}/{p['fp_n']}$", f"${p['fn']}/{p['fn_n']}$"]
        out.append(f"    {a[-1]} & " + " & ".join(cells) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    # --- Table: Qwen 512 and style-swap
    out += [r"\begin{table}[htb]", r"  \centering", r"  \footnotesize",
            r"  \caption{Refusal contrasts for the 512-token Qwen rerun and the style-swap, seed 42,"
            r" $95\%$ paired-prompt bootstrap intervals.}",
            r"  \label{tab:qwen-swap}",
            r"  \begin{tabular}{@{}llcc@{}}", r"    \toprule",
            r"    Check & Contrast & Compliant lens & Held-out paraphrase \\", r"    \midrule"]
    out.append(f"    Qwen, 512 tokens & $I_1^R-I_2^R$ & {ci(R16['I1-I2_agreeable'], 313)} & {ci(R16['I1-I2_agreeable_para1'], 313)} \\\\")
    out.append(f"     & $I_3^R-I_2^R$ & {ci(R16['I3-I2_agreeable'], 313)} & {ci(R16['I3-I2_agreeable_para1'], 313)} \\\\")
    for a in (1, 2, 3):
        out.append(f"     & $I_{a}^R$ & {ci(R16[f'I{a}_agreeable'], 313)} & {ci(R16[f'I{a}_agreeable_para1'], 313)} \\\\")
    out.append(r"    \addlinespace")
    out.append(f"    Style-swap & $I_8^R$ (arm 1 content, neutral) & {ci(R17['I8_agreeable'], 313)} & {ci(R17['I8_agreeable_para1'], 313)} \\\\")
    out.append(f"     & $I_9^R$ (arm 2 content, warm) & {ci(R17['I9_agreeable'], 313)} & {ci(R17['I9_agreeable_para1'], 313)} \\\\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(out), A


if __name__ == "__main__":
    tex, _ = build()
    (HERE / "camready_tables.tex").write_text(tex, encoding="utf-8", newline="\n")
    print("wrote camready_tables.tex")
