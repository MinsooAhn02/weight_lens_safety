"""`nb_r7.ipynb`를 생성한다 (두 번째 모델 계열 이질적 반복).

노트북 JSON은 손으로 편집하지 않는다. R5에서 검증한 셀 0~8을 그대로 물려받고,
모델 상수만 명시적인 강제 실패 패치로 Qwen2.5-7B-Instruct로 바꾼다.

    cd branch_1_premise/experiment
    python src/make_nb_r7.py && python src/check_notebooks.py nb_r7.ipynb

모든 패치는 (셀 번호, 찾을 것, 바꿀 것, 왜)의 목록이며 원본이 예상과 다르면 즉시 실패한다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(HERE, "nb_r5.ipynb")
OUT_NB = os.path.join(HERE, "nb_r7.ipynb")
NB_REV = "R7b"
NB_NAME = "nb_r7"
UNSLOTH_PIN = "2026.7.6"
BASE = "unsloth/Qwen2.5-7B-Instruct"
OUT_SUFFIX = "_qwen25_7b_env76"


# --------------------------------------------------------------- 상속 셀 패치
# 모델별 채팅 처리는 common.py의 tokenizer template 경로를 그대로 쓴다.
# (셀 번호, 찾을 것, 바꿀 것, 왜)
CELL_PATCHES = [
    (
        8,
        'BASE = "unsloth/Meta-Llama-3.1-8B-Instruct"',
        f'BASE = "{BASE}"',
        "기반 모델 상수만 Qwen2.5-7B-Instruct로 바꾼다",
    ),
]


# ------------------------------------------------------------------- 새 셀
MD_HEAD = """# R7 — Qwen2.5-7B-Instruct 이질적 반복

R5의 검증된 셀 0~8을 상속하고, `BASE`만 `unsloth/Qwen2.5-7B-Instruct`로 바꾼다.
arm0·arm1·arm2·arm3을 seed 42에서 실행하며, 학습 arm은 세 라운드 뒤 r3만 네 렌즈로 평가한다.

이 실행이 성공하기 전에는 원고에서 **“한 레시피를 넘어 반복되었다”**고 말할 수 없다.
성공하더라도 두 모델 계열·단일 seed의 이질적 반복이므로 모델 및 seed 일반화를 입증하지 않는다.

★**이 파일을 직접 수정하지 말 것** — `src/make_nb_r7.py`가 생성한다."""

MD_PLAN = f"""## 4. ★이번 세션 — `BATCH` 하나만 바꾼다

측정값은 학습 round당 약 800초, 렌즈당 약 1,800초(`n=313`)다. 표의 시간은 순수 GPU
작업 추정치이며 설치·Qwen 모델 다운로드·압축 여유는 별도다. 모든 평가는 256 tokens,
`none,agreeable,principled,agreeable_para1` 네 렌즈이고 새 출력은 `{OUT_SUFFIX}`로 분리한다.

| BATCH | 세션 | 계산 | 순수 GPU 추정 |
|---|---|---|---:|
| **1** | arm0·arm1 × seed 42 | arm0: 4렌즈; arm1: 3라운드 + 4렌즈 | 4h40m |
| **2** | arm2·arm3 × seed 42 | 각 arm: 3라운드 + 4렌즈 | 5h20m |

R7의 순수 GPU 합계는 **10시간**, 두 Kaggle 세션의 설치·다운로드·압축 여유를 더한 계획값은
**약 11시간**이다. 이미 계획된 R5의 순수 GPU 36시간에 R7이 10시간을 추가하므로 둘만 해도
**46시간**이며, 주간 약 30시간 한도 안에 함께 끝난다고 가정하지 않는다.

Qwen 경로도 `common.py`의 tokenizer chat template, system role, left padding, 그리고 scalar/복수
`eos_token_id` 처리를 그대로 사용한다. 모델별 특례를 새로 넣지 않고 `BASE`만 바꾼다.

보너스 조작 확인도 추가 비용 없이 따라온다. 최근 발견한 거절 도구의 typographic apostrophe 결함은
Llama-3.1 출력에서 관찰되었다. Qwen도 같은 문자를 내는지는 이 실행에서 직접 볼 수 있고, 각 결과
JSON에 저장되는 패턴별 fire count로 별도 평가 없이 확인할 수 있다."""

CODE_DRIVER = f'''import json, os

# ── 리비전 게이트 ────────────────────────────────────────────────────────────
NB_NAME = "{NB_NAME}"
NB_REV = "{NB_REV}"
_ds_revisions = json.load(open("configs/arms.json", encoding="utf-8")).get("nb_revisions", {{}})
_ds_rev = _ds_revisions.get(NB_NAME)
assert _ds_rev == NB_REV, (
    f"리비전 불일치: 노트북={{NB_NAME}}/{{NB_REV}} Dataset={{_ds_rev}}. "
    f"둘 중 하나가 낡았다 — Dataset을 다시 올리거나 노트북을 다시 Import하라")
print(f"[gate] 리비전 {{NB_NAME}}/{{NB_REV}} 일치")

UNSLOTH_PIN = "{UNSLOTH_PIN}"
OUT_SUFFIX = "{OUT_SUFFIX}"
LENSES = "none,agreeable,principled,agreeable_para1"
MAX_NEW_TOKENS = 256

# evaluator가 결과에 쓰는 것과 같은 stamp로 실행 환경을 다시 확인한다.
from common import env_versions
RUNTIME_ENV = env_versions()
assert RUNTIME_ENV.get("unsloth") == UNSLOTH_PIN, (
    f"Unsloth 실행 환경 불일치: 기대={{UNSLOTH_PIN}} stamp={{RUNTIME_ENV}}")
print(f"[gate] 실행 환경 stamp 일치: {{RUNTIME_ENV}}")

# ★★ 커밋마다 이 값 하나만 바꾼다. ★★
BATCH = 1

BATCHES = {{
    1: [("Q01", "arm0", 42), ("Q02", "arm1", 42)],
    2: [("Q03", "arm2", 42), ("Q04", "arm3", 42)],
}}
assert BATCH in BATCHES, f"BATCH는 {{sorted(BATCHES)}} 중 하나여야 한다"
PLAN = BATCHES[BATCH]
print(f"R7 BATCH {{BATCH}}: " + ", ".join(f"{{n}}({{a}} x {{s}})" for n, a, s in PLAN))


def snap(tag):
    """세션마다 누적 results zip을 떠서 뒤 세션이 실패해도 앞 결과를 회수한다."""
    import shutil
    zp = shutil.make_archive(f"/kaggle/working/r7_results_{{tag}}", "zip",
                             "/kaggle/working/experiment/results")
    print(f"[snap] {{zp}} | {{os.path.getsize(zp)/1e6:.2f}} MB", flush=True)


for name, arm, seed in PLAN:
    print(f"\\n===== {{name}}: {{arm}} x seed {{seed}} =====", flush=True)
    if arm == "arm0":
        run_baseline(seed, lenses=LENSES, mmlu=False,
                     max_new_tokens=MAX_NEW_TOKENS, out_suffix=OUT_SUFFIX)
    else:
        # 세 라운드를 순서대로 학습하되 비용이 큰 평가는 r3에서만 한다.
        run_session(arm, seed, lenses=LENSES, eval_rounds=(3,),
                    probe=False, mmlu=False, max_new_tokens=MAX_NEW_TOKENS,
                    out_suffix=OUT_SUFFIX)
    snap(f"batch{{BATCH}}_{{name.lower()}}_{{arm}}_s{{seed}}")
'''

CODE_ZIP = '''import shutil, os, glob

# TAG는 BATCH에서 유도한다 — 노트북 내용과 zip 이름이 어긋나지 않게 한다.
TAG = f"batch{BATCH}"
zp = shutil.make_archive(f"/kaggle/working/r7_results_{TAG}", "zip",
                         "/kaggle/working/experiment/results")
print("zip:", zp, "| size MB:", round(os.path.getsize(zp)/1e6, 2))
for p in sorted(glob.glob("/kaggle/working/experiment/results/*.json")):
    print(f"  {os.path.getsize(p)/1024:7.0f} KB  {os.path.basename(p)}")
'''

MD_GATE = """## 6. 세션이 성공했는지

BATCH별 기대 파일, 네 렌즈, 생성 메타데이터, Unsloth 환경 stamp를 모두 검사한다.
`snap()`은 각 세션 직후 실행되므로 여기서 실패해도 앞 세션 결과는 회수할 수 있다."""

CODE_GATE = f'''import json, os

N = 313
UNSLOTH_PIN = "{UNSLOTH_PIN}"
OUT_SUFFIX = "{OUT_SUFFIX}"
EXPECTED_LENSES = ["none", "agreeable", "principled", "agreeable_para1"]
EXPECTED_MAX_TOKENS = 256

assert OUT_SUFFIX.startswith("_qwen25_7b_"), OUT_SUFFIX
expected = []
for _, arm, seed in PLAN:
    rtag = 0 if arm == "arm0" else 3
    expected.append(f"results/eval_{{arm}}_r{{rtag}}_s{{seed}}{{OUT_SUFFIX}}.json")

assert len(expected) == len(set(expected)), f"기대 출력 이름이 충돌한다: {{expected}}"
assert all(path.endswith(f"{{OUT_SUFFIX}}.json") for path in expected), expected

for path in expected:
    assert os.path.exists(path), f"결과가 없다: {{path}}"
    ev = json.load(open(path, encoding="utf-8"))
    assert ev.get("schema") == "r2", f"{{path}}: 스키마가 다르다"
    assert ev.get("max_new_tokens") == EXPECTED_MAX_TOKENS, (
        f"{{path}}: max_new_tokens={{ev.get('max_new_tokens')}}")
    assert ev.get("out_suffix") == OUT_SUFFIX, (
        f"{{path}}: out_suffix={{ev.get('out_suffix')}} 기대={{OUT_SUFFIX}}")
    assert ev.get("env", {{}}).get("unsloth") == UNSLOTH_PIN, (
        f"{{path}}: env stamp unsloth={{ev.get('env', {{}}).get('unsloth')}}; 핀={{UNSLOTH_PIN}}")
    assert ev.get("lenses") == EXPECTED_LENSES, f"{{path}}: 렌즈={{ev.get('lenses')}}"
    assert list(ev.get("conditions", {{}})) == EXPECTED_LENSES, (
        f"{{path}}: condition 순서/목록 불일치")

    for lens in EXPECTED_LENSES:
        condition = ev["conditions"][lens]
        records = condition["records"]
        assert condition["n"] == N and len(records) == N, (
            f"{{path}}/{{lens}}: n={{condition.get('n')}} records={{len(records)}}")
        assert all({{"hit_cap", "n_new_tokens", "finish_reason"}} <= set(record)
                   for record in records), f"{{path}}/{{lens}}: 생성 메타데이터가 없다"
        assert all(0 < record["n_new_tokens"] <= EXPECTED_MAX_TOKENS
                   for record in records), f"{{path}}/{{lens}}: 생성 토큰 수 범위 오류"
        n_hit_cap = sum(bool(record["hit_cap"]) for record in records)
        assert condition.get("n_hit_cap") == n_hit_cap, (
            f"{{path}}/{{lens}}: cap 집계 불일치")
        assert all(record["finish_reason"] in
                   {{"eos_token", "max_new_tokens", "stopping_criteria"}}
                   for record in records), f"{{path}}/{{lens}}: 알 수 없는 종료 이유"
        print(f"  {{os.path.basename(path)}}/{{lens}} cap={{n_hit_cap}}/{{N}}")

print(f"\\n[gate] R7 BATCH {{BATCH}} 통과 — {{len(expected)}}개 eval 파일")
'''


def patch_cells(cells):
    for index, old, new, why in CELL_PATCHES:
        src = "".join(cells[index]["source"])
        count = src.count(old)
        if count != 1:
            sys.exit(
                f"패치 실패({why}): nb_r5.ipynb 셀 {index}에서 정확히 한 번 찾아야 하는데 "
                f"{count}번 찾았다.\n찾던 것:\n{old[:240]}"
            )
        cells[index]["source"] = as_source(src.replace(old, new, 1))


def as_source(text: str):
    """nbformat source를 개행 유지 줄 목록으로 저장한다."""
    return text.splitlines(keepends=True)


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    cells = nb["cells"][:9]  # 0~8 = R5에서 검증된 Secrets/설치/복구/함수 정의
    patch_cells(cells)
    cells[0]["source"] = as_source(MD_HEAD)

    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": as_source(text)}

    def code(text):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": as_source(text),
        }

    cells += [
        md(MD_PLAN), code(CODE_DRIVER),
        md("## 5. ★결과 회수"), code(CODE_ZIP),
        md(MD_GATE), code(CODE_GATE),
    ]

    nb["cells"] = cells
    with open(OUT_NB, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"[make] -> {os.path.relpath(OUT_NB, HERE)} ({len(cells)} cells)")
    print("[make] 다음: python src/check_notebooks.py nb_r7.ipynb")


if __name__ == "__main__":
    main()
