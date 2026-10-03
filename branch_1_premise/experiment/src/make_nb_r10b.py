"""`nb_r10b.ipynb`를 생성한다 (R10의 512토큰 재실행 — 절단 원인 측정).

`make_nb_r10.py`와 같은 구조이고 **바꾸는 것은 셋뿐이다**: `MAX_NEW_TOKENS`가 512,
접미사가 `_max512`, BATCH가 시드 42 둘로 줄어든다. 스모크(BATCH 0)는 없다 —
full SFT 경로는 `nb_r10`의 BATCH 0이 이미 통과시켰고, 여기서 새로 검증할 외부 API
가정이 없다.

    cd branch_1_premise/experiment
    python src/make_nb_r10b.py && python src/check_notebooks.py nb_r10b.ipynb

사전등록 문서: `R10_UPDATE_METHOD.md` **§8**. 그 문서가 커밋되기 전에 돌리지 않는다.

왜 이 세션이 필요한가 (§2.2 요약). R10 본 12런의 판정이 절단 눈금에 따라 갈린다 —
포함 0.761("살아남았다") / 제외 0.468("부분적"). 원고가 included를 헤드라인으로 쓰는
근거(R74)는 8B/512에서 휴리스틱 절단의 89.4%가 상한과 무관한 **학습된 문중 EOS**라는
측정이었는데, 1B/256에서 같은 측정을 하면 반대가 나온다(실제 상한 도달 LoRA 76.5% /
full 54.7%). 그래서 R10은 지금 상·하한 쌍으로 묶여 있고, 512에서 상한 도달이 얼마나
떨어지는지를 봐야 어느 쪽을 헤드라인으로 쓸지가 **관측으로** 정해진다.

⚠️ **512와 256을 한 표에 섞지 않는다**(`RESULTS_INTAKE.md:83`). 별도 코호트다.
⚠️ 렌즈를 줄이지 않는다 — `cohort_signature`가 `lenses`를 대조하므로(`analyze_dual_primary.py:136`)
   렌즈 집합이 다르면 다른 코호트와 결합이 영영 불가능해진다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(HERE, "nb_r5.ipynb")
OUT_NB = os.path.join(HERE, "nb_r10b.ipynb")
NB_REV = "R10b1"
NB_NAME = "nb_r10b"
UNSLOTH_PIN = "2026.7.6"
BASE = "unsloth/Llama-3.2-1B-Instruct"

# (셀 번호, 찾을 것, 바꿀 것, 왜) — nb_r10과 **같은 두 패치**다.
# 셀 8은 R5에서 검증된 것을 그대로 물려받고 update_method만 통과시킨다.
CELL_PATCHES = [
    (
        8,
        'probe_rounds=None, max_new_tokens=256, out_suffix=""):',
        'probe_rounds=None, max_new_tokens=256, out_suffix="", update_method="lora"):',
        "run_session이 update_method를 받게 한다. 기본값 lora라 기존 호출은 그대로다",
    ),
    (
        8,
        'f"--output_dir {ckpt}{frac_arg}")',
        'f"--output_dir {ckpt}{frac_arg} --update_method {update_method}")',
        "train_round.py에 update_method를 넘긴다",
    ),
]

MD_HEAD = """# R10b — 512토큰 재실행 (절단 원인 측정)

R10 본 12런과 **같은 모델·같은 데이터·같은 렌즈**로 돌리고 생성 상한만 256 → 512로 올린다.
모델은 `unsloth/Llama-3.2-1B-Instruct`.

**사전등록**: `R10_UPDATE_METHOD.md` **§8**. 512에서 상한 도달률이 얼마나 떨어지는지가
R10 판정을 상·하한 쌍에서 점추정으로 좁힐지를 정한다. **§2의 1/2·1/4 눈금은 손대지 않는다.**

★**이 파일을 직접 수정하지 말 것** — `src/make_nb_r10b.py`가 생성한다."""

MD_PLAN = """## 4. ★이번 세션 — `BATCH` 하나만 바꾼다

512는 256의 약 2배를 생성하므로 런당 약 1.5h(3라운드 학습 + 4렌즈 평가)다.

| BATCH | 내용 | 순수 GPU 추정 |
|---|---|---:|
| 1 | LoRA × seed 42 (arm1, arm2) | 3.0h |
| 2 | full × seed 42 (arm1, arm2) | 3.0h |

**시드는 42 하나다.** `_env76_max512`(R5 배치 1) 선례와 같은 규모이며, 상한 경계를
재는 데 3시드가 필요하지 않다 — 재는 것은 효과 크기가 아니라 **생성이 상한에 닿는
비율**이다.

### 스모크가 없는 이유

`nb_r10`의 BATCH 0이 외부 API 가정 둘(unsloth `full_finetuning=True`,
lm_eval `ifeval`)을 이미 통과시켰고, 이 노트북은 그 위에서 상한 숫자만 바꾼다.
**새로 검증할 가정이 없다.**

### ⚠️ 선행 조건

R10 본 12런이 **전부 수용된 뒤**에 돌린다. 이 세션은 그 판정의 눈금을 정하는 것이지
판정을 대체하지 않는다."""

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
LENSES = "none,agreeable,principled,agreeable_para1"

# ★ R10 본 실행(256)과 **다른 유일한 측정 설정**이다. 접미사가 이것과 짝을 이룬다.
MAX_NEW_TOKENS = 512

from common import env_versions
RUNTIME_ENV = env_versions()
assert RUNTIME_ENV.get("unsloth") == UNSLOTH_PIN, (
    f"Unsloth 실행 환경 불일치: 기대={{UNSLOTH_PIN}} stamp={{RUNTIME_ENV}}")
print(f"[gate] 실행 환경 stamp 일치: {{RUNTIME_ENV}}")

# ★ 셀 8의 BASE를 1B로 덮어쓴다. 셀 8의 함수들은 호출 시점에 전역 BASE를 읽는다.
BASE = "{BASE}"
print(f"[gate] BASE = {{BASE}}")

SUFFIX = {{"lora": "_r10_1b_lora_max512", "full": "_r10_1b_full_max512"}}

# ★★ 커밋마다 이 값 하나만 바꾼다. ★★
BATCH = 1

BATCHES = {{1: ("lora", 42), 2: ("full", 42)}}
assert BATCH in BATCHES, f"BATCH는 {{sorted(BATCHES)}} 중 하나여야 한다"

# 접미사가 256 코호트와 겹치면 정본을 덮어쓴다. 기계가 막는다.
for _m, _s in SUFFIX.items():
    assert _s.endswith("_max512"), f"접미사가 512 코호트를 가리키지 않는다: {{_s}}"
    assert "env76" not in _s, (
        f"접미사 {{_s}}가 256 코호트(_env76)와 섞인다 — 512와 256은 다른 측정이다")
'''

MD_DRIVER = """## 5. ★본 실행

데이터는 **기존 arm1·arm2 라운드 데이터를 그대로 쓴다. 재생성하지 않는다.**
학습은 3라운드, 평가는 r3에서만 네 렌즈로 한다.

체크포인트는 세션과 함께 사라지므로 학습을 다시 돈다. 시드 42 고정이고 데이터가
같으므로 재현되는 학습이다."""

CODE_RUN = '''import os

method, seed = BATCHES[BATCH]
out_suffix = SUFFIX[method]
print(f"R10b BATCH {BATCH}: {method} x seed {seed} @ {MAX_NEW_TOKENS}토큰 -> {out_suffix}")

def snap(tag):
    """세션마다 누적 zip을 떠서 뒤 세션이 실패해도 앞 결과를 회수한다."""
    import shutil
    zp = shutil.make_archive(f"/kaggle/working/r10b_results_{tag}", "zip",
                             "/kaggle/working/experiment/results")
    print(f"[snap] {zp} | {os.path.getsize(zp)/1e6:.2f} MB", flush=True)

for arm in ["arm1", "arm2"]:
    print(f"\\n===== {arm} x seed {seed} ({method}) @ {MAX_NEW_TOKENS} =====", flush=True)
    run_session(arm, seed, lenses=LENSES, eval_rounds=(3,),
                probe=False, mmlu=False, max_new_tokens=MAX_NEW_TOKENS,
                out_suffix=out_suffix, update_method=method)
    snap(f"batch{BATCH}_{arm}_s{seed}_{method}")
'''

CODE_ZIP = '''import shutil, os, glob

TAG = f"batch{BATCH}"
zp = shutil.make_archive(f"/kaggle/working/r10b_results_{TAG}", "zip",
                         "/kaggle/working/experiment/results")
print("zip:", zp, "| size MB:", round(os.path.getsize(zp)/1e6, 2))
for p in sorted(glob.glob("/kaggle/working/experiment/results/*.json")):
    print(f"  {os.path.getsize(p)/1024:7.0f} KB  {os.path.basename(p)}")
'''

MD_GATE = """## 7. 세션이 성공했는지

기대 파일·네 렌즈·생성 메타데이터·환경 stamp를 검사하고,
**이 세션의 목적인 상한 도달률을 바로 인쇄한다.**"""

CODE_GATE = f'''import json, os

N = 313
UNSLOTH_PIN = "{UNSLOTH_PIN}"
EXPECTED_LENSES = ["none", "agreeable", "principled", "agreeable_para1"]

method, seed = BATCHES[BATCH]
OUT_SUFFIX = SUFFIX[method]
expected = [f"results/eval_{{arm}}_r3_s{{seed}}{{OUT_SUFFIX}}.json"
            for arm in ["arm1", "arm2"]]
assert len(expected) == len(set(expected)), f"기대 출력 이름이 충돌한다: {{expected}}"

for path in expected:
    assert os.path.exists(path), f"결과가 없다: {{path}}"
    d = json.load(open(path, encoding="utf-8"))
    assert d["schema"] == "r2", f"{{path}}: schema={{d['schema']}}"
    assert d["n_prompts"] == N, f"{{path}}: n_prompts={{d['n_prompts']}}"
    assert d["max_new_tokens"] == {512}, (
        f"{{path}}: max_new_tokens={{d['max_new_tokens']}} — 512여야 한다. "
        f"256이면 이 세션은 무의미하다")
    assert d["lenses"] == EXPECTED_LENSES, f"{{path}}: lenses={{d['lenses']}}"
    assert d["out_suffix"] == OUT_SUFFIX, f"{{path}}: out_suffix={{d['out_suffix']}}"
    assert d["env"].get("unsloth") == UNSLOTH_PIN, f"{{path}}: env={{d['env']}}"

print(f"[gate] R10b BATCH {{BATCH}} 통과 — {{len(expected)}}개 eval 파일 ({{method}})")

# ── ★이 세션의 목적: 상한 도달률을 256과 나란히 본다 ─────────────────────────
# R10 본 실행(256)에서 휴리스틱 절단 중 실제 상한 도달은 LoRA 76.5% / full 54.7%였다.
# R74가 8B/512에서 본 값은 10.6%다. 여기서 얼마나 떨어지는지가 §8.3의 판정 입력이다.
print("\\n[cap] 상한 도달률 — finish_reason 기준 (휴리스틱 아님)")
for path in expected:
    d = json.load(open(path, encoding="utf-8"))
    for lens in EXPECTED_LENSES:
        recs = d["conditions"][lens]["records"]
        cap = sum(1 for r in recs if r.get("hit_cap"))
        print(f"  {{os.path.basename(path):52}} {{lens:16}} "
              f"hit_cap={{cap:4}}/{{len(recs)}} ({{cap/len(recs):.1%}})")
'''


def main():
    with open(SRC_NB, encoding="utf-8") as fh:
        nb = json.load(fh)

    cells = nb["cells"][:9]
    for index, needle, replacement, why in CELL_PATCHES:
        source = "".join(cells[index]["source"])
        if needle not in source:
            sys.exit(
                f"[make] 셀 {index}에서 패치 지점을 못 찾았다: {needle!r}\n"
                f"        이유: {why}\n"
                f"        nb_r5.ipynb가 바뀌었다 — 확인 전에는 생성하지 않는다.")
        source = source.replace(needle, replacement)
        cells[index]["source"] = source.splitlines(keepends=True)

    def as_source(text):
        return text.splitlines(keepends=True)

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

    cells[0] = md(MD_HEAD)
    cells += [
        md(MD_PLAN), code(CODE_DRIVER),
        md(MD_DRIVER), code(CODE_RUN),
        md("## 6. ★결과 회수"), code(CODE_ZIP),
        md(MD_GATE), code(CODE_GATE),
    ]

    nb["cells"] = cells
    with open(OUT_NB, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"[make] -> {os.path.relpath(OUT_NB, HERE)} ({len(cells)} cells)")
    print("[make] 다음: python src/check_notebooks.py nb_r10b.ipynb")
    print(f"[make] ⚠️ configs/arms.json의 nb_revisions에 "
          f'"{NB_NAME}": "{NB_REV}"를 넣고 Kaggle Dataset을 New Version으로 재업로드할 것')


if __name__ == "__main__":
    main()
