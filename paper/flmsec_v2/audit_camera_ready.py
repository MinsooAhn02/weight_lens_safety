"""Camera-ready audit (2026-10-01): recompute every number added in the camera-ready
revision from the saved eval/training artifacts and require that main.tex prints it.

Read-only. Run from this directory:  python audit_camera_ready.py
Exits non-zero on the first mismatch. Bootstrap intervals use seed 20261001 and 10,000
resamples, so they reproduce exactly.
"""
import json
import math
import random
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRANCH = HERE.parents[1] / "branch_1_premise"
RES = BRANCH / "experiment" / "results"
sys.path.insert(0, str(BRANCH / "experiment" / "src"))
from eval_refusal import REFUSAL_PATTERNS, normalize_refusal_text  # noqa: E402
from analyze_dual_primary import looks_truncated  # noqa: E402
sys.path.insert(0, str(HERE.parent / "shared"))
from audit_common import require_text_layer  # noqa: E402

require_text_layer()

TEX = " ".join((HERE / "main.tex").read_text(encoding="utf-8").split())
SEEDS = (42, 1337, 2718)
FAIL = []


def need(cond, msg):
    if not cond:
        FAIL.append(msg)


def printed(fragment, why):
    need(" ".join(fragment.split()) in TEX, f"{why}: main.tex lacks `{fragment}`")


def close(name, want, got, tol=0.0005):
    need(abs(want - got) <= tol, f"{name}: printed {want}, data {got:.4f}")


def refused(text, norm):
    t = (normalize_refusal_text(text) if norm else text).lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)


def load(arm, seed, suffix, rnd=3):
    name = f"eval_arm0_r0_s{seed}{suffix}.json" if arm == 0 else f"eval_arm{arm}_r{rnd}_s{seed}{suffix}.json"
    return json.loads((RES / name).read_text(encoding="utf-8"))["conditions"]


def boot(vals, B=10000, seed=20261001):
    rng = random.Random(seed); n = len(vals)
    bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(vals) / n, bs[int(.025 * B)], bs[int(.975 * B) - 1]


# --- training data apostrophes ---------------------------------------------------
z = zipfile.ZipFile(BRANCH / "data" / "gen_data_backup.zip")
train = [json.loads(l)["response"] for a in (1, 2, 3) for r in (1, 2, 3)
         for l in z.read(f"arm{a}/round{r}.jsonl").decode("utf-8").splitlines() if l.strip()]
need(len(train) == 1235, f"training rows {len(train)} != 1235")
need(not any("’" in t for t in train), "a training response contains U+2019")
printed("all $1{,}235$ training responses", "training-apostrophe sentence")

# --- headline grid: curly counts, flips, pre-correction rates -------------------
H = {(a, s): load(a, s, "_env76") for a in range(4) for s in SEEDS}
curly = re.compile(r"\bi can’t\b", re.I)
cnt = {a: sum(bool(curly.search(r["response"])) for s in SEEDS for c in H[(a, s)].values()
              for r in c["records"]) for a in range(4)}
need(cnt == {0: 39, 1: 0, 2: 57, 3: 0}, f"curly counts {cnt}")
printed("number $39$ in arm 0 and $57$ in arm 2", "curly-count sentence")
flips = {a: sum(refused(r["response"], True) and not refused(r["response"], False)
                for s in SEEDS for c in H[(a, s)].values() for r in c["records"]) for a in range(4)}
need(flips == {0: 24, 1: 0, 2: 33, 3: 0}, f"flips {flips}")


def rate(a, L, norm, cond=H, seeds=SEEDS):
    return sum(sum(refused(r["response"], norm) for r in cond[(a, s)][L]["records"]) / 313
               for s in seeds) / len(seeds)


for (a, L, before, after) in [(0, "none", .933, .949), (0, "agreeable", .939, .946),
                              (2, "none", .939, .967), (2, "agreeable", .971, .976)]:
    close(f"arm{a} {L} raw", before, rate(a, L, False)); close(f"arm{a} {L} norm", after, rate(a, L, True))


def drops(a, s, L, cond=H, key=lambda r: float(r["refused_regex"])):
    rr = cond[(a, s)]
    return [key(x) - key(y) for x, y in zip(rr["none"]["records"], rr[L]["records"])]


raw_key = lambda r: float(refused(r["response"], False))
I2_raw = sum(sum(drops(2, s, "agreeable", key=raw_key)) / 313 - sum(drops(0, s, "agreeable", key=raw_key)) / 313
             for s in SEEDS) / 3
close("I2 pre-correction", -0.026, I2_raw)
per_prompt_shift = [sum((drops(2, s, "agreeable")[i] - drops(0, s, "agreeable")[i])
                        - (drops(2, s, "agreeable", key=raw_key)[i] - drops(0, s, "agreeable", key=raw_key)[i])
                        for s in SEEDS) / 3 for i in range(313)]
m, lo, hi = boot(per_prompt_shift)
close("shift", 0.014, m); close("shift lo", -0.003, lo); close("shift hi", 0.031, hi)
printed("$+0.014$ $[-0.003, +0.031]$", "shift interval")

# --- intervals for interactions and arm-minus-arm2 ------------------------------
for L, rows in [("agreeable", [(1, None, 0.306, 0.252, 0.362), (3, None, 0.381, 0.327, 0.438),
                               (2, None, -0.012, -0.040, 0.018), (1, 2, 0.317, 0.269, 0.367),
                               (3, 2, 0.393, 0.342, 0.445)]),
                ("agreeable_para1", [(1, None, 0.478, 0.419, 0.538), (3, None, 0.383, 0.330, 0.439),
                                     (2, None, -0.020, -0.056, 0.016)])]:
    for a, ref, pm, plo, phi in rows:
        ref = 0 if ref is None else ref
        v = [sum(drops(a, s, L)[i] - drops(ref, s, L)[i] for s in SEEDS) / 3 for i in range(313)]
        m, lo, hi = boot(v)
        for nm, w, g in (("point", pm, m), ("lo", plo, lo), ("hi", phi, hi)):
            close(f"{L} arm{a}-arm{ref} {nm}", w, g)
        printed(f"{pm:.3f}$ $[{plo:.3f}, {phi:.3f}]$", f"{L} arm{a}-arm{ref} interval")

for a, want in ((1, 0.049), (3, 0.147), (2, 0.009)):
    got = sum(sum(drops(a, s, "principled")) / 313 - sum(drops(0, s, "principled")) / 313 for s in SEEDS) / 3
    close(f"principled I{a}", want, got)
printed("$0.049$, $0.147$ and $0.009$", "principled-lens sentence")

# --- truncation -------------------------------------------------------------------
fl = [r for a in range(4) for c in load(a, 42, "_env76").values() for r in c["records"] if looks_truncated(r["response"])]
need((len(fl), sum(r["hit_cap"] for r in fl)) == (549, 487), "256-cap flagged/hit-cap counts")
printed("($487$ of them reached the cap)", "256-cap hit sentence")
for a, want in ((1, 0.153), (2, 0.010), (3, 0.278)):
    rr = load(a, 42, "_env76_max512")["agreeable"]["records"]
    close(f"512 mid-sentence arm{a}", want, sum(looks_truncated(r["response"]) and not r["hit_cap"] for r in rr) / 313)
printed("$0.010$ $[0.003, 0.028]$", "arm-2 512 mid-sentence interval")
for m_, a, want in (("lora", 1, .080), ("lora", 2, .013), ("full", 1, .233), ("full", 2, .019)):
    rr = load(a, 42, f"_r10_1b_{m_}_max512")["agreeable"]["records"]
    close(f"1B {m_} arm{a}", want, sum(looks_truncated(r["response"]) and not r["hit_cap"] for r in rr) / 313)

# --- earlier (legacy) grid --------------------------------------------------------
G = {(a, 42): load(a, 42, "") for a in range(4)}
for a, L, b, af in [(0, "none", .933, .949), (0, "agreeable", .939, .946), (1, "none", .981, .981),
                    (1, "agreeable", .633, .633), (2, "none", .866, .968), (2, "agreeable", .942, .942),
                    (3, "none", .987, .987), (3, "agreeable", .489, .489)]:
    close(f"legacy arm{a} {L} before", b, rate(a, L, False, G, (42,)))
    close(f"legacy arm{a} {L} after", af, rate(a, L, True, G, (42,)))


def legacy_I(a, norm):
    return (rate(a, "none", norm, G, (42,)) - rate(a, "agreeable", norm, G, (42,))) - \
           (rate(0, "none", norm, G, (42,)) - rate(0, "agreeable", norm, G, (42,)))


for a, b, af in ((1, .355, .345), (2, -.070, .022), (3, .505, .495)):
    close(f"legacy I{a} before", b, legacy_I(a, False)); close(f"legacy I{a} after", af, legacy_I(a, True))
close("legacy shift", 0.093, legacy_I(2, True) - legacy_I(2, False))
printed("$+0.093$ in an earlier grid", "abstract legacy shift")
q = json.loads((RES / "eval_arm1_r0p25_s42.json").read_text(encoding="utf-8"))["conditions"]
close("arm1 r0.25 before", 0.930, sum(refused(r["response"], False) for r in q["none"]["records"]) / 313)
close("arm1 r0.25 after", 0.971, sum(refused(r["response"], True) for r in q["none"]["records"]) / 313)
lf = sum(refused(r["response"], True) and not refused(r["response"], False) for r in G[(2, 42)]["none"]["records"])
need(lf == 32, f"legacy arm2 standard flips {lf}")

# --- validation cohort overlap ------------------------------------------------------
same = []
for a in (1, 2, 3):
    for L in ("none", "agreeable"):
        x, y = G[(a, 42)][L]["records"], H[(a, 42)][L]["records"]
        same.append(sum(p["response"] == q_["response"] for p, q_ in zip(x, y)) / 313)
need(round(min(same) * 100) == 15 and round(max(same) * 100) == 30, f"overlap range {min(same):.3f}-{max(same):.3f}")
need(all(p["response"] == q_["response"] for L in ("none", "agreeable")
         for p, q_ in zip(G[(0, 42)][L]["records"], H[(0, 42)][L]["records"])), "arm0 not byte-identical")
close("CP upper 0/173", 0.021, 1 - 0.025 ** (1 / 173))

# --- Qwen arm-minus-arm2 -------------------------------------------------------------
Q = {a: load(a, 42, "_qwen25_7b_env76") for a in range(4)}
d = {a: (sum(r["refused_regex"] for r in Q[a]["none"]["records"]) - sum(r["refused_regex"] for r in Q[a]["agreeable"]["records"])) / 313 for a in range(4)}
close("Qwen I1-I2", 0.115, d[1] - d[2]); close("Qwen I3-I2", 0.093, d[3] - d[2])

# --- seven-lens paraphrase count (App E, Table 8) --------------------------------
R12 = json.loads((RES / "r12_lens_axis_judgment.json").read_text(encoding="utf-8"))["cells"]


def para_wins(interaction):
    n = n1 = 0
    for a in ("arm1", "arm3"):
        for s in SEEDS:
            for conv in ("included", "excluded"):
                c, c0 = R12[f"{a}/s{s}"][conv]["drop"], R12["arm0/s42"][conv]["drop"]  # arm 0 seed-invariant
                f = (lambda L: c[L] - c0[L]) if interaction else (lambda L: c[L])
                ok = max(f("agreeable_para1"), f("agreeable_para2")) > f("agreeable")
                n += ok; n1 += ok and a == "arm1"
    return n, n1


need(para_wins(False) == (11, 6), f"paraphrase raw-drop count {para_wins(False)} != (11, 6)")
need(para_wins(True)[0] == 10, f"paraphrase interaction count {para_wins(True)} != 10")
printed("eleven of twelve cells, all six for arm 1; on the interaction scale the count is ten of twelve",
        "App E paraphrase count")

# --- M05 hand check of mid-sentence endings (Section 5, App. M05) ------------------
ANN = BRANCH / "experiment" / "annotation"
lab = {l.split("\t")[0]: l.split("\t")[1].strip() for l in (ANN / "m05_labels.tsv").read_text(encoding="utf-8").splitlines()
       if l and not l.startswith(("#", "item_id"))}
arm_of = {json.loads(l)["item_id"]: json.loads(l)["arm"] for l in (ANN / "m05_key.jsonl").read_text(encoding="utf-8").splitlines()}
need(len(lab) == 61 and all(v in "ELOX" and v for v in lab.values()), "M05 labels incomplete or invalid")
E = {a: sum(1 for i, v in lab.items() if v == "E" and arm_of[i] == a) for a in ("arm1", "arm3", "arm2")}
need(sum(v == "E" for v in lab.values()) == 53, "M05 E count != 53")
for a, want in (("arm1", 0.064), ("arm3", 0.099), ("arm2", 0.006)):
    close(f"M05 {a}", want, E[a] / 313)
printed("$53$ of the $61$ endings stop mid-sentence", "M05 sentence")
printed("$0.064$ $[0.042, 0.097]$, $0.099$ $[0.071, 0.137]$ and $0.006$ $[0.002, 0.023]$", "M05 intervals")

# --- R15-R17 camera-ready checks (Section 6, App. camready) ------------------------
# The appendix tables come from make_camready_tables.py; it must reproduce the file on disk.
sys.path.insert(0, str(HERE))
from make_camready_tables import build  # noqa: E402  (reruns analyze_camera_ready.py)
_tex, A = build()
need(_tex == (HERE / "camready_tables.tex").read_text(encoding="utf-8"), "camready_tables.tex is stale")
need(r"\input{camready_tables}" in TEX, "camready tables not included")


def fmt(x):
    return f"{x:.3f}" if round(x, 3) != 0 else "0.000"


def ci_txt(t):
    return f"${fmt(t[0])}$ $[{fmt(t[1])}, {fmt(t[2])}]$".replace("$-", "$-").replace("[-", "[-")


def refusal_drops(conds, L):
    k = lambda r: float(r["refused_regex"])
    return [k(x) - k(y) for x, y in zip(conds["none"]["records"], conds[L]["records"])]


# independent recomputation of the printed R16/R17 contrasts from the eval files
q = {a: load(a, 42, "_qwen25_7b_max512") for a in range(4)}
need(all(len(q[a]["none"]["records"]) == 313 for a in q), "R16 files not 313 prompts")
d = {a: refusal_drops(q[a], "agreeable") for a in q}
for a in (1, 3):
    t = boot([x - y for x, y in zip(d[a], d[2])])
    printed(f"$I_{a}^R-I_2^R={fmt(t[0])}$ $[{fmt(t[1])}, {fmt(t[2])}]$", f"R16 I{a}-I2")
close("R16 arm0 drop", 0.348, sum(d[0]) / 313)
printed("$0.348$, above the preregistered limit of $0.10$", "R16 arm0 drop")
for a, L, want in ((0, "none", 0.661), (0, "agreeable", 0.313)):
    close(f"R16 arm{a} {L}", want, sum(float(r["refused_regex"]) for r in q[a][L]["records"]) / 313)
cells = [sum(bool(r.get("hit_cap")) for r in q[a][L]["records"]) / 313 for a in q for L in q[a]]
close("R16 hit_cap max", 0.128, max(cells), tol=0.0005)
close("R16 hit_cap pooled", 0.046, sum(cells) / len(cells), tol=0.0005)
printed("$4.6\\%$ of all responses but in $12.8\\%$ of arm 0's standard-lens", "R16 cap")

b0 = load(0, 42, "_env76")
for a, src_arm, want_src in ((8, 1, 0.367), (9, 2, -0.029)):
    c = load(a, 42, "_r17_styleswap_env76")
    t = boot([x - y for x, y in zip(refusal_drops(c, "agreeable"), refusal_drops(b0, "agreeable"))])
    printed(f"$I_{a}^R={fmt(t[0])}$ $[{fmt(t[1])}, {fmt(t[2])}]$", f"R17 I{a}")
    s = load(src_arm, 42, "_env76")
    close(f"R17 comparator arm{src_arm}", want_src,
          sum(x - y for x, y in zip(refusal_drops(s, "agreeable"), refusal_drops(b0, "agreeable"))) / 313)
s1 = load(1, 42, "_env76")
close("R17 arm1 para comparator", 0.550,
      sum(x - y for x, y in zip(refusal_drops(s1, "agreeable_para1"), refusal_drops(b0, "agreeable_para1"))) / 313)
printed("($0.380$ against arm 1's $0.550$ at seed 42)", "R17 paraphrase sentence")
for a, want in ((8, 1.07), (9, 1.43)):
    rep = json.loads((RES / f"rewrite_arm{a}_report.json").read_text(encoding="utf-8"))
    need(rep["gate_pass"] and rep["empty"] == 0, f"R17 arm{a} rewrite gate")
    close(f"R17 arm{a} length ratio", want, rep["median_length_ratio"], tol=0.005)
printed("Median length ratios were $1.07$ and $1.43$", "R17 gate sentence")
need(A["R17"]["decision"] == "mixed, inconclusive", "R17 decision changed")
printed("``mixed, inconclusive''", "R17 decision text")

# R15: judge files cover every headline + earlier-grid completion, no missing label
for j in ("llamaguard3", "harmbench"):
    J = json.loads((RES / f"judge_{j}.json").read_text(encoding="utf-8"))["files"]
    n = sum(len(x["labels"]) for v in J.values() for x in v["lenses"].values())
    nul = sum(l is None for v in J.values() for x in v["lenses"].values() for l in x["labels"])
    need(n == 15024 + 5008 and nul == 0, f"R15 {j}: {n} labels, {nul} missing")
printed("$15{,}024$ headline-grid and $5{,}008$ earlier-grid completions; no label was missing", "R15 coverage")
need(A["R15"]["A1_decision"] == "survives judge change", "R15 A1 changed")
hb = A["R15"]["harmbench"]["A2"]["per_arm"]; lg = A["R15"]["llamaguard3"]["A2"]["per_arm"]
need((hb["arm3"]["fn"], hb["arm3"]["fn_n"], hb["arm1"]["fn"], hb["arm1"]["fn_n"]) == (8, 20, 2, 29), "R15 HB FN counts")
need((lg["arm3"]["fn"], lg["arm3"]["fn_n"], lg["arm1"]["fn"], lg["arm1"]["fn_n"]) == (6, 20, 5, 29), "R15 LG FN counts")
need(sum(p["fp"] for p in hb.values()) <= 1 and sum(p["fp"] for p in lg.values()) <= 1, "R15 FP > 1")
pairs = {tuple(x["arms"]): x for x in A["R15"]["harmbench"]["A2"]["tests"]["fn"]["pairs"]}
close("R15 HB FN p", 0.009, pairs[("arm1", "arm3")]["fisher_p"])
lgp = {tuple(x["arms"]): x for x in A["R15"]["llamaguard3"]["A2"]["tests"]["fn"]["pairs"]}
close("R15 LG FN p", 0.32, lgp[("arm1", "arm3")]["fisher_p"], tol=0.005)
ntest = sum(A["R15"][j]["A2"]["tests"][k]["n_tests"] for j in ("llamaguard3", "harmbench") for k in ("fp", "fn"))
need(ntest == 24, f"R15 test count {ntest}")
flags = [(j, k, tuple(x["arms"])) for j in ("llamaguard3", "harmbench") for k in ("fp", "fn")
         for x in A["R15"][j]["A2"]["tests"][k]["pairs"] if x["flag"]]
need(flags == [("harmbench", "fn", ("arm1", "arm3"))], f"R15 flags {flags}")
printed("missed $8$ of $20$ harmful arm-3 responses and $2$ of $29$ in arm 1 ($p=0.009$, one of $24$ uncorrected tests)",
        "R15 body sentence")
printed("$8/20$ against $2/29$ for the HarmBench classifier ($p=0.009$, Fisher exact) and $6/20$ against $5/29$ for Llama Guard ($p=0.32$)",
        "R15 appendix sentence")

if FAIL:
    print(f"{len(FAIL)} FAIL"); [print("  -", f) for f in FAIL]; sys.exit(1)
print("camera-ready audit: all checks passed")
