"""Generate the three camera-ready notebooks (R15 judges, R16 Qwen@512, R17 style-swap).

    cd branch_1_premise/experiment
    python src/make_nb_camera_ready.py && python src/check_notebooks.py nb_r15_judges.ipynb nb_r16_qwen512.ipynb nb_r17_styleswap.ipynb

Preregistration: R15_R17_CAMERA_READY.md — commit it before running any notebook.
Like make_nb_r11.py, every notebook inherits the validated cells 0-8 of nb_r5.ipynb and
applies only explicit, fail-fast patches. Do not edit the generated .ipynb files by hand.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(HERE, "nb_r5.ipynb")
UNSLOTH_PIN = "2026.7.6"
REVS = {"nb_r15_judges": "R15a", "nb_r16_qwen512": "R16c", "nb_r17_styleswap": "R17c"}
# Dataset revisions each notebook accepts. b and c changed only the notebook (pip pin, preflight),
# never a file the run executes from the Dataset, so every account's existing upload still works.
DATASET_OK = {"nb_r15_judges": ["R15a"],
              "nb_r16_qwen512": ["R16a", "R16b", "R16c"],
              "nb_r17_styleswap": ["R17a", "R17b", "R17c"]}
# R16b/R17b: unsloth 2026.7.6 only requires unsloth_zoo>=2026.7.7, so a fresh install pulls
# the newest zoo (2026.9.9 was released 2026-10-01). R16a on 2026-10-01 died in training with
# vanilla trl SFTConfig raising "doesn't support bf16" on T4 - the Unsloth patch did not apply,
# with unsloth/transformers/trl identical to R7. Pin the zoo current for R7/R12 (Aug 2026).
ZOO_PIN = "2026.8.3"
ZOO_PATCH = (4, 'pip("unsloth==2026.7.6")\n',
             ('pip("unsloth==2026.7.6", "unsloth_zoo==@ZOO@")\n'
              'from importlib.metadata import version as _v\n'
              'assert _v("unsloth_zoo") == "@ZOO@", f"unsloth_zoo 핀 불일치: {_v(\'unsloth_zoo\')}"\n'
              ).replace("@ZOO@", ZOO_PIN),
             "unsloth_zoo 핀 (R16b 이후)")

GATE = '''import json, os
NB_NAME = "@NB@"
NB_REV = "@REV@"
DATASET_OK = @OK@
_rev = json.load(open("configs/arms.json", encoding="utf-8")).get("nb_revisions", {}).get(NB_NAME)
assert _rev in DATASET_OK, (f"리비전 불일치: 노트북={NB_NAME}/{NB_REV} Dataset={_rev} (허용 {DATASET_OK}) — "
                            "Dataset을 New Version으로 다시 올리거나 노트북을 다시 Import하라")
assert os.path.exists("R15_R17_CAMERA_READY.md"), "사전등록 문서가 Dataset에 없다 — 커밋·업로드 후 실행"
print(f"[gate] {NB_NAME}/{NB_REV} (Dataset {_rev}) · 사전등록 문서 확인")
'''

# R16c/R17c: Unsloth swaps trl.SFTConfig/SFTTrainer for UnslothSFTConfig/UnslothSFTTrainer at
# `import unsloth` (models/llama.py -> PatchFastRL) and swallows any failure (logger.info). An
# unpatched SFTConfig raises "doesn't support bf16" on T4 only when training starts - after the
# 70-minute R17 rewrite. Check the swap in a fresh process, before any long step.
PREFLIGHT = '''import os, subprocess, sys
_code = ("import unsloth, trl; "
         "print('[preflight] trl.SFTConfig =', trl.SFTConfig.__name__, '| trl.SFTTrainer =', trl.SFTTrainer.__name__, flush=True); "
         "assert trl.SFTConfig.__name__ == 'UnslothSFTConfig' and trl.SFTTrainer.__name__ == 'UnslothSFTTrainer'")
_pf = subprocess.run([sys.executable, "-c", _code], env={**os.environ, "UNSLOTH_ENABLE_LOGGING": "1"})
assert _pf.returncode == 0, ("Unsloth가 trl SFTConfig를 패치하지 못했다 — 이대로면 학습이 T4 bf16 오류로 죽는다. "
                             "이 로그를 Claude에게 보낼 것")
print("[preflight] Unsloth trl 패치 확인 — 학습 가능", flush=True)
'''

ZIP = '''import shutil, os, glob
zp = shutil.make_archive(f"/kaggle/working/@TAG@_results_batch{BATCH}", "zip",
                         "/kaggle/working/experiment/results")
print("zip:", zp, "| MB:", round(os.path.getsize(zp) / 1e6, 2))
for p in sorted(glob.glob("/kaggle/working/experiment/results/*@GLOB@*")):
    print(f"  {os.path.getsize(p)/1024:8.0f} KB  {os.path.basename(p)}")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def base_cells(patches):
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    cells = nb["cells"][:9]
    for index, needle, replacement, why in patches:
        src = "".join(cells[index]["source"])
        if needle not in src:
            sys.exit(f"[make] 셀 {index}에 패치 지점 없음: {needle!r} ({why}) — nb_r5가 바뀌었다")
        cells[index]["source"] = src.replace(needle, replacement).splitlines(keepends=True)
    return nb, cells


def write(nb, cells, name):
    nb["cells"] = cells
    out = os.path.join(HERE, name + ".ipynb")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"[make] -> {name}.ipynb ({len(cells)} cells, rev {REVS[name]})")


# ----------------------------------------------------------------------- R15
R15_HEAD = """# R15 — 학습된 판정기로 재채점 (재학습 없음)

사전등록 `R15_R17_CAMERA_READY.md` §R15. 저장된 completion만 읽는다 — 생성·학습 없음.
판정기: `meta-llama/Llama-Guard-3-8B`, `cais/HarmBench-Llama-2-13b-cls` (4-bit).
입력: 헤드라인 `_env76` 12파일(15,024) + 이전 seed-42 grid 4파일(5,008, 인간 검증 표본의 출처).

**필요한 것**: GPU T4 x2 · Secrets `HF_TOKEN`(Llama-Guard-3 게이트 승인 계정) ·
Input `branch-1-premise`(새 버전), `gen-data-backup`.

★이 파일을 직접 수정하지 말 것 — `src/make_nb_camera_ready.py`가 생성한다."""

R15_RUN = '''import glob, os, subprocess, sys
pip("bitsandbytes>=0.43", "accelerate")

HEADLINE = sorted(glob.glob("results/eval_arm*_r*_s*_env76.json"))
LEGACY = [f"results/eval_arm0_r0_s42.json"] + [f"results/eval_arm{a}_r3_s42.json" for a in (1, 2, 3)]
assert len(HEADLINE) == 12, f"헤드라인 eval 파일이 12개가 아니다: {len(HEADLINE)}"
assert all(os.path.exists(p) for p in LEGACY), f"이전 grid 파일 누락: {LEGACY}"
run("python src/rescore_judges.py --selftest")

# ★ 커밋마다 이 값 하나만 바꾼다. 1 = 두 판정기 모두(기본, ~3-4h) · 2 = Llama-Guard만 · 3 = HarmBench만
BATCH = 1
JUDGE_SETS = {1: ["llamaguard3", "harmbench"], 2: ["llamaguard3"], 3: ["harmbench"]}
for judge in JUDGE_SETS[BATCH]:
    bs = 8 if judge == "llamaguard3" else 4
    files = " ".join(f'--files "{p}"' for p in HEADLINE + LEGACY)
    run(f"python src/rescore_judges.py --judge {judge} --batch_size {bs} {files}")
    import shutil
    shutil.copy(f"results/judge_{judge}.json", f"/kaggle/working/judge_{judge}.json")
    print(f"[snap] /kaggle/working/judge_{judge}.json", flush=True)
'''

# ----------------------------------------------------------------------- R16
R16_HEAD = """# R16 — Qwen2.5-7B-Instruct, 512 토큰

사전등록 `R15_R17_CAMERA_READY.md` §R16. R7(256 토큰, 32.9% 절단으로 결론 불가)을 512에서 다시 한다.
arm 0–3, seed 42, lens `none,agreeable,agreeable_para1`, 접미사 `_qwen25_7b_max512`.

| BATCH | 내용 | 순수 GPU 추정 |
|---|---|---:|
| **1** | arm0 + arm1 | ~5h45m |
| **2** | arm2 + arm3 | ~6h25m |
| 3/4/5/6 | arm0 / arm1 / arm2 / arm3 단독 (복구용) | ~2h30m–3h15m |

계정 두 개에서 BATCH 1과 2를 동시에 돌린다.

★이 파일을 직접 수정하지 말 것 — `src/make_nb_camera_ready.py`가 생성한다."""

R16_RUN = '''SEED = 42
LENSES = "none,agreeable,agreeable_para1"
CAP, SUFFIX = 512, "_qwen25_7b_max512"
assert BASE == "unsloth/Qwen2.5-7B-Instruct", BASE
from common import env_versions
assert env_versions().get("unsloth") == "@PIN@", env_versions()

# ★ 커밋마다 이 값 하나만 바꾼다.
BATCH = 1
BATCHES = {1: ["arm0", "arm1"], 2: ["arm2", "arm3"], 3: ["arm0"], 4: ["arm1"], 5: ["arm2"], 6: ["arm3"]}

import os, shutil
def snap(tag):
    zp = shutil.make_archive(f"/kaggle/working/r16_results_{tag}", "zip", "/kaggle/working/experiment/results")
    print(f"[snap] {zp}", flush=True)

for arm in BATCHES[BATCH]:
    print(f"\\n===== R16 {arm} x seed {SEED} @ {CAP} =====", flush=True)
    if arm == "arm0":
        run_baseline(SEED, lenses=LENSES, mmlu=False, max_new_tokens=CAP, out_suffix=SUFFIX)
        out = f"results/eval_arm0_r0_s{SEED}{SUFFIX}.json"
    else:
        run_session(arm, SEED, lenses=LENSES, eval_rounds=(3,), probe=False, mmlu=False,
                    max_new_tokens=CAP, out_suffix=SUFFIX)
        out = f"results/eval_{arm}_r3_s{SEED}{SUFFIX}.json"
    d = json.load(open(out, encoding="utf-8"))
    assert d["max_new_tokens"] == CAP and d["n_prompts"] == 313, (d["max_new_tokens"], d["n_prompts"])
    for lens, c in d["conditions"].items():
        cap = sum(r.get("hit_cap", False) for r in c["records"])
        print(f"  [{arm}] {lens:16} refusal={c['refusal_rate']:.3f} hit_cap={cap}/313")
    snap(f"batch{BATCH}_{arm}")
'''

# ----------------------------------------------------------------------- R17
R17_HEAD = """# R17 — style-swap (seed 42)

사전등록 `R15_R17_CAMERA_READY.md` §R17.
- **BATCH 1 → arm8**: arm1(compliant persona)의 학습 응답을 **중립 문체**로 재작성 → 학습 → 평가
- **BATCH 2 → arm9**: arm2(generic)의 학습 응답을 **따뜻하고 비위 맞추는 문체**로 재작성 → 학습 → 평가

재작성 품질 게이트(길이 비율 중앙값 0.5–2.0, 빈 응답 0)를 통과하지 못하면 학습하지 않고 멈춘다.
lens `none,agreeable,agreeable_para1`, 256 토큰, 접미사 `_r17_styleswap_env76`. 예상 ~2h/BATCH.

★이 파일을 직접 수정하지 말 것 — `src/make_nb_camera_ready.py`가 생성한다."""

R17_RUN = '''SEED = 42
LENSES = "none,agreeable,agreeable_para1"
SUFFIX = "_r17_styleswap_env76"
assert BASE == "unsloth/Meta-Llama-3.1-8B-Instruct", BASE
from common import env_versions
assert env_versions().get("unsloth") == "@PIN@", env_versions()

# ★ 커밋마다 이 값 하나만 바꾼다.
BATCH = 1
JOBS = {1: ("arm1", "arm8", "neutral"), 2: ("arm2", "arm9", "warm")}
src_arm, dst_arm, style = JOBS[BATCH]

run("python src/rewrite_style.py --selftest")
run(f"python src/rewrite_style.py --src {src_arm} --dst {dst_arm} --style {style}")
assert all(os.path.exists(f"data/{dst_arm}/round{r}.jsonl") for r in (1, 2, 3)), "재작성 데이터 없음"
import shutil
# Keep the rewrite even if training dies: it is the 70-minute step.
shutil.copytree(f"data/{dst_arm}", f"/kaggle/working/data_{dst_arm}", dirs_exist_ok=True)
shutil.copy(f"results/rewrite_{dst_arm}_report.json", f"/kaggle/working/rewrite_{dst_arm}_report.json")
print(f"[snap] /kaggle/working/data_{dst_arm}", flush=True)

run_session(dst_arm, SEED, lenses=LENSES, eval_rounds=(3,), probe=False, mmlu=False,
            max_new_tokens=256, out_suffix=SUFFIX)
out = f"results/eval_{dst_arm}_r3_s{SEED}{SUFFIX}.json"
d = json.load(open(out, encoding="utf-8"))
for lens, c in d["conditions"].items():
    print(f"  [{dst_arm}] {lens:16} refusal={c['refusal_rate']:.3f}")
import shutil
shutil.copytree(f"data/{dst_arm}", f"/kaggle/working/data_{dst_arm}", dirs_exist_ok=True)
'''


def gate(name):
    return code(GATE.replace("@NB@", name).replace("@REV@", REVS[name])
                .replace("@OK@", repr(DATASET_OK[name])))


def main():
    llama_patch = []
    qwen_patch = [(8, 'BASE = "unsloth/Meta-Llama-3.1-8B-Instruct"',
                   'BASE = "unsloth/Qwen2.5-7B-Instruct"', "R16은 Qwen이다"), ZOO_PATCH]

    nb, cells = base_cells(llama_patch)
    cells = cells[:7]  # secrets, packages, folder restore — no training helpers needed
    cells.append(code("import subprocess\n"
                      "def run(cmd):\n"
                      "    print(f\"\\n$ {cmd}\", flush=True)\n"
                      "    assert subprocess.run(cmd, shell=True).returncode == 0, f\"실패: {cmd}\"\n"))
    cells[0] = md(R15_HEAD)
    cells += [gate("nb_r15_judges"),
              md("## 재채점"), code(R15_RUN),
              md("## 결과 회수"), code(ZIP.replace("@TAG@", "r15").replace("@GLOB@", "judge_"))]
    write(nb, cells, "nb_r15_judges")

    nb, cells = base_cells(qwen_patch)
    cells[0] = md(R16_HEAD)
    cells += [gate("nb_r16_qwen512"), code(PREFLIGHT),
              md("## 본 실행 — BATCH 하나"), code(R16_RUN.replace("@PIN@", UNSLOTH_PIN)),
              md("## 결과 회수"), code(ZIP.replace("@TAG@", "r16").replace("@GLOB@", "qwen25_7b_max512"))]
    write(nb, cells, "nb_r16_qwen512")

    nb, cells = base_cells([ZOO_PATCH])
    cells[0] = md(R17_HEAD)
    cells += [gate("nb_r17_styleswap"), code(PREFLIGHT),
              md("## 재작성 → 학습 → 평가"), code(R17_RUN.replace("@PIN@", UNSLOTH_PIN)),
              md("## 결과 회수"), code(ZIP.replace("@TAG@", "r17").replace("@GLOB@", "r17_styleswap"))]
    write(nb, cells, "nb_r17_styleswap")
    print("[make] configs/arms.json nb_revisions에", REVS, "가 있어야 한다")


if __name__ == "__main__":
    main()
