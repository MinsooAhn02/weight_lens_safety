"""`nb_r5.ipynb`를 생성한다 (RUNBOOK 함정 15).

노트북 JSON은 손으로 편집하지 않는다. R4에서 실제 검증한 `nb_r4.ipynb`의 셀 0~8을
그대로 물려받고, 바뀌는 설치 핀과 실행 함수 인자만 명시적 패치로 고친다.

    cd branch_1_premise/experiment
    python src/make_nb_r5.py && python src/check_notebooks.py nb_r5.ipynb

모든 패치는 (찾을 것, 바꿀 것, 왜)의 목록이며 원본이 하나라도 예상과 다르면 즉시 실패한다.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(HERE, "nb_r4.ipynb")
OUT_NB = os.path.join(HERE, "nb_r5.ipynb")
NB_REV = "R5b"
NB_NAME = "nb_r5"
UNSLOTH_PIN = "2026.7.6"
ENV_SUFFIX = "_env76"
CAP_SUFFIX = "_env76_max512"


# --------------------------------------------------------------- 상속 셀 패치
# (셀 번호, 찾을 것, 바꿀 것, 왜)
CELL_PATCHES = [
    (
        8,
        '--output_dir {ckpt}{frac_arg}")\n\n        # 세분 라운드: 라운드1 학습 중 저장된 어댑터를 원본 BASE 위에 얹어 평가',
        '--output_dir {ckpt}{frac_arg}")\n\n        # ★R5: 라운드 r 머지본이 생긴 순간 r-1은 죽은 무게다. 원래는 eval/probe/mmlu가\n        #   끝난 뒤에야 지웠는데, 4렌즈 eval은 약 2시간이라 그동안 32GB를 붙들고 있었다.\n        #   R5 배치 2가 실제로 여기서 디스크를 터뜨렸다(Docker daemon 오류는 그 증상).\n        if prev_dir and os.path.exists(prev_dir):\n            shutil.rmtree(prev_dir)\n            print(f"[disk] 삭제(학습 직후): {prev_dir}")\n            prev_dir = None\n\n        # 세분 라운드: 라운드1 학습 중 저장된 어댑터를 원본 BASE 위에 얹어 평가',
        '이전 라운드 머지본을 eval 전에 지워 eval 구간 점유를 32GB->16GB로 낮춘다',
    ),
    (
        8,
        '# 디스크: 머지본 1개 = 16GB. 다음 라운드 전에 이전 것 삭제.\n        if prev_dir and os.path.exists(prev_dir):\n            shutil.rmtree(prev_dir)\n            print(f"[disk] 삭제: {prev_dir}")\n        prev, prev_dir = ckpt, ckpt',
        '# 디스크 해제는 학습 직후로 옮겼다(R5). 여기서는 다음 라운드 기준만 갱신한다.\n        prev, prev_dir = ckpt, ckpt',
        '옛 해제 지점을 제거한다 — 위에서 이미 지웠다',
    ),
    (
        3,
        "Unsloth 공식 Llama3.1(8B) 노트북 핀: `transformers==4.56.2`, `trl==0.22.2`.",
        "비교 가능성 핀: `unsloth==2026.7.6`, `transformers==4.56.2`, `trl==0.22.2`.",
        "설치 셀 설명에 Unsloth 핀을 명시한다",
    ),
    (
        4,
        'pip("unsloth")\n'
        'pip("transformers==4.56.2", "trl==0.22.2")',
        f'pip("unsloth=={UNSLOTH_PIN}")\n'
        'pip("transformers==4.56.2", "trl==0.22.2")\n'
        '\n'
        'from importlib.metadata import version\n'
        f'_installed_unsloth = version("unsloth")\n'
        f'assert _installed_unsloth == "{UNSLOTH_PIN}", (\n'
        f'    f"Unsloth 설치 핀 불일치: 기대={UNSLOTH_PIN} 실제={{_installed_unsloth}}")',
        "Unsloth를 고정하고 설치 직후 실제 버전을 확인한다",
    ),
    (
        8,
        'def run_session(arm, seed, lenses="none,agreeable,principled",\n'
        '                probe=False, frac=False, mmlu=True, eval_rounds=(1, 2, 3),\n'
        '                harm_frame="judgement", mmlu_limit=None, free_ckpt=True,\n'
        '                probe_rounds=None):',
        'def run_session(arm, seed, lenses="none,agreeable,principled",\n'
        '                probe=False, frac=False, mmlu=True, eval_rounds=(1, 2, 3),\n'
        '                harm_frame="judgement", mmlu_limit=None, free_ckpt=True,\n'
        '                probe_rounds=None, max_new_tokens=256, out_suffix=""):',
        "학습 세션 평가에 생성 길이와 안전한 출력 접미사를 전달한다",
    ),
    (
        8,
        '                      None이면 eval_rounds를 따른다(기존 동작 그대로)."""\n'
        '    base_dirs = probe_baseline(seed, harm_frame) if probe else None\n'
        '    pr_rounds = eval_rounds if probe_rounds is None else probe_rounds',
        '                      None이면 eval_rounds를 따른다(기존 동작 그대로).\n'
        '       max_new_tokens, out_suffix — ★R5 신설. 512-token 평가는 기존 256-token\n'
        '                      결과와 파일명이 절대 겹치지 않게 접미사를 함께 쓴다."""\n'
        '    base_dirs = probe_baseline(seed, harm_frame) if probe else None\n'
        '    pr_rounds = eval_rounds if probe_rounds is None else probe_rounds\n'
        '    assert max_new_tokens > 0\n'
        '    assert not out_suffix or (out_suffix.startswith("_") and out_suffix[1:].replace("_", "").replace("-", "").isalnum())\n'
        '    eval_args = f" --max_new_tokens {max_new_tokens}" + (f" --out_suffix {out_suffix}" if out_suffix else "")',
        "새 평가 인자를 문서화하고 한 곳에서 안전하게 조립한다",
    ),
    (
        8,
        '                run(f"python src/eval_refusal.py --model_path {BASE} --adapter_path {ad} "\n'
        '                    f"--arm {arm} --round {rn} --seed {seed} --lenses {lenses}{FULL}")\n'
        '                check(f"results/eval_{arm}_r{f}_s{seed}.json")',
        '                run(f"python src/eval_refusal.py --model_path {BASE} --adapter_path {ad} "\n'
        '                    f"--arm {arm} --round {rn} --seed {seed} --lenses {lenses}{FULL}{eval_args}")\n'
        '                check(f"results/eval_{arm}_r{f}_s{seed}{out_suffix}.json")',
        "세분 라운드 평가도 생성 인자와 접미사 파일명을 일치시킨다",
    ),
    (
        8,
        '            run(f"python src/eval_refusal.py --model_path {ckpt} --arm {arm} --round {r} "\n'
        '                f"--seed {seed} --lenses {lenses}{FULL}")\n'
        '            check(f"results/eval_{arm}_r{r}_s{seed}.json")',
        '            run(f"python src/eval_refusal.py --model_path {ckpt} --arm {arm} --round {r} "\n'
        '                f"--seed {seed} --lenses {lenses}{FULL}{eval_args}")\n'
        '            check(f"results/eval_{arm}_r{r}_s{seed}{out_suffix}.json")',
        "정수 라운드 평가 명령과 확인 경로에 생성 인자를 반영한다",
    ),
    (
        8,
        'def run_baseline(seed, lenses="none,agreeable,principled", mmlu_limit=None,\n'
        '                 mmlu=True):\n'
        '    """arm0 = 원본 모델. 학습 없음. 페르소나 방아쇠 자체가 무해함을 보이는 통제군.\n'
        '       mmlu — ★R4 신설. arm0의 limit=10 MMLU는 이미 있다. 다시 돌리면 20분을 버리고\n'
        '              results/mmlu_arm0_r0/ 에 타임스탬프 파일이 하나 더 쌓인다."""\n'
        '    run(f"python src/eval_refusal.py --model_path {BASE} --arm arm0 --round 0 "\n'
        '        f"--seed {seed} --lenses {lenses}{FULL}")\n'
        '    check(f"results/eval_arm0_r0_s{seed}.json")',
        'def run_baseline(seed, lenses="none,agreeable,principled", mmlu_limit=None,\n'
        '                 mmlu=True, max_new_tokens=256, out_suffix=""):\n'
        '    """arm0 = 원본 모델. 학습 없음. 페르소나 방아쇠 자체가 무해함을 보이는 통제군.\n'
        '       mmlu — ★R4 신설. arm0의 limit=10 MMLU는 이미 있다. 다시 돌리면 20분을 버리고\n'
        '              results/mmlu_arm0_r0/ 에 타임스탬프 파일이 하나 더 쌓인다.\n'
        '       max_new_tokens, out_suffix — ★R5 신설. 생성 상한과 충돌 없는 파일 접미사."""\n'
        '    assert max_new_tokens > 0\n'
        '    assert not out_suffix or (out_suffix.startswith("_") and out_suffix[1:].replace("_", "").replace("-", "").isalnum())\n'
        '    eval_args = f" --max_new_tokens {max_new_tokens}" + (f" --out_suffix {out_suffix}" if out_suffix else "")\n'
        '    run(f"python src/eval_refusal.py --model_path {BASE} --arm arm0 --round 0 "\n'
        '        f"--seed {seed} --lenses {lenses}{FULL}{eval_args}")\n'
        '    check(f"results/eval_arm0_r0_s{seed}{out_suffix}.json")',
        "무학습 arm0 평가에도 같은 생성 길이와 접미사를 적용한다",
    ),
]


# ------------------------------------------------------------------- 새 셀
MD_HEAD = """# R5 — 단일 Unsloth 환경 전체 재실행 · 생성 상한 경계

R4의 검증된 셀 0~8을 상속하고, 모든 arm×seed를 Unsloth 2026.7.6으로 다시 맞춘다.

- **BATCH 1**: 256-token 상한이 효과 크기를 만든 정도를 512-token 재평가로 경계 짓는다.
- **BATCH 2~6**: 네 arm×세 seed의 256-token grid를 충돌 없는 새 이름으로 완성한다.

★**이 파일을 직접 수정하지 말 것** — `src/make_nb_r5.py`가 생성한다."""

MD_PLAN = """## 4. ★이번 세션 — `BATCH` 하나만 바꾼다

측정값은 학습 round당 약 800초, 렌즈당 약 1,800초(`n=313`)다. 표의 시간은 이 둘만 더한
GPU 작업 추정치이며 설치·모델 다운로드·zip 여유는 별도다. 모든 배치는 9시간보다 충분히
작고, 부분 완료 때도 먼저 쓸 수 있도록 seed-matched arm0↔persona 비교를 arm2보다 앞에 둔다.

| BATCH | 세션 | 실행 | 출력 | 추정 |
|---|---|---|---|---:|
| **1** | 네 arm × seed 42 | `none,agreeable`, 512 tokens. arm0은 평가만, arm1/2/3은 3라운드 학습 뒤 r3 평가 | `*_env76_max512.json` | 6h00m |
| **2** | arm0·arm1·arm3 × seed 1337 | 4렌즈. arm0은 평가만, arm1/3은 3라운드 학습 뒤 r3 평가 | `*_env76.json` | 7h20m |
| **3** | arm0·arm1·arm3 × seed 2718 | 4렌즈. arm0은 평가만, arm1/3은 3라운드 학습 뒤 r3 평가 | `*_env76.json` | 7h20m |
| **4** | arm2 × seed 1337·2718 | 4렌즈, 각각 3라운드 학습 뒤 r3 평가 | `*_env76.json` | 5h20m |
| **5** | arm0·arm1·arm3 × seed 42 | 4렌즈. 완전한 접미사 grid를 위해 target-stamped arm1/3도 재실행 | `*_env76.json` | 7h20m |
| **6** | arm2 × seed 42 | 4렌즈, 3라운드 학습 뒤 r3 평가 | `*_env76.json` | 2h40m |

순수 작업 합계는 **36시간**, 설치·다운로드·압축 여유를 더한 계획값은 **약 38시간**이다.
Kaggle 주간 약 30시간 한도를 넘으므로 **최소 2주**가 필요하다. 한 주에 끝난다고 가정하지 않는다.

현재 저장 JSON에서 직접 확인되는 target 환경은 `arm1/r3/seed42`, `arm3/r3/seed42`뿐이며
둘 다 `env.unsloth=2026.7.6`이다. `arm0/r0/seed42`, `arm2/r3/seed42`, 그리고
`arm1·arm3/r3/seed1337·2718`은 파일은 있으나 env stamp가 없다. stamp 부재는 어느 버전의
증거도 아니다. `arm0·arm2 × seed1337·2718`은 현재 파일 자체가 없다. 배경 실행 기록은
seed1337·2718 persona가 2026.7.5였다고 하지만 JSON만으로 독립 확인할 수 없으므로 다시 돈다.
target-stamped 두 셀도 완전하고 독립된 `_env76` namespace를 만들기 위해 BATCH 5에서 다시 돈다.

⚠️ **BATCH 1의 arm1/2/3 round-3 체크포인트는 이미 삭제됐다.** 따라서 512-token 평가 전에
세 arm을 동일한 데이터·하이퍼파라미터로 round 1→3까지 다시 학습한다. arm0은 학습하지 않는다.
512-token 결과는 `_env76_max512`, 표준 256-token 결과는 `_env76`을 써 기존 `eval_*.json`과도,
서로와도 절대 겹치지 않는다. evaluator도 기본값이 아닌 생성 길이에 접미사가 없으면 실패한다.

⚠️ **Unsloth는 `2026.7.6`으로 고정한다.** 설치 직후 패키지 버전을 확인하고, 실행 직전에는
결과 JSON과 같은 `env_versions()` stamp에서 다시 확인한다. 하나라도 다르면 학습 전에 크게 실패한다.
마지막 게이트도 각 새 결과 JSON의 `env.unsloth`가 정확히 이 버전인지 확인한다."""

CODE_DRIVER = f'''import json, os

# ── 리비전 게이트 ────────────────────────────────────────────────────────────
# R3 BATCH 2에서 **낡은 노트북**이 옛 TAG로 zip을 냈다(내용은 s2718인데 이름은 s1337).
# 리포의 셀 12는 이미 고쳐져 있었고 Kaggle 쪽만 낡았던 것이다 — 사람이 눈으로 비교해서는
# 못 잡는다. Dataset(configs/arms.json)과 노트북에 같은 리비전을 박고 여기서 대조한다.
NB_NAME = "{NB_NAME}"
NB_REV = "{NB_REV}"
_ds_revisions = json.load(open("configs/arms.json", encoding="utf-8")).get("nb_revisions", {{}})
_ds_rev = _ds_revisions.get(NB_NAME)
assert _ds_rev == NB_REV, (
    f"리비전 불일치: 노트북={{NB_NAME}}/{{NB_REV}} Dataset={{_ds_rev}}. "
    f"둘 중 하나가 낡았다 — Dataset을 다시 올리거나 노트북을 다시 Import하라")
print(f"[gate] 리비전 {{NB_NAME}}/{{NB_REV}} 일치")

UNSLOTH_PIN = "{UNSLOTH_PIN}"
ENV_SUFFIX = "{ENV_SUFFIX}"
CAP_SUFFIX = "{CAP_SUFFIX}"
LENSES_CAP = "none,agreeable"
LENSES_GRID = "none,agreeable,principled,agreeable_para1"

# evaluator가 결과에 쓰는 것과 같은 stamp로 실행 환경을 다시 확인한다.
from common import env_versions
RUNTIME_ENV = env_versions()
assert RUNTIME_ENV.get("unsloth") == UNSLOTH_PIN, (
    f"Unsloth 실행 환경 불일치: 기대={{UNSLOTH_PIN}} stamp={{RUNTIME_ENV}}")
print(f"[gate] 실행 환경 stamp 일치: {{RUNTIME_ENV}}")

# ★★ 커밋마다 이 값 하나만 바꾼다. ★★
BATCH = 1

BATCHES = {{
    # 512-token 생성 상한 경계. 새 환경 접미사를 함께 써 256-token 결과와 구분한다.
    1: [("C01", "arm0", 42), ("C02", "arm1", 42),
        ("C03", "arm2", 42), ("C04", "arm3", 42)],
    # seed-matched arm0↔persona 비교를 먼저 닫는다.
    2: [("E01", "arm0", 1337), ("E02", "arm1", 1337),
        ("E03", "arm3", 1337)],
    3: [("E04", "arm0", 2718), ("E05", "arm1", 2718),
        ("E06", "arm3", 2718)],
    # 일반 FT 대조군과 seed42의 완전한 접미사 grid는 그 다음이다.
    4: [("E07", "arm2", 1337), ("E08", "arm2", 2718)],
    5: [("E09", "arm0", 42), ("E10", "arm1", 42),
        ("E11", "arm3", 42)],
    6: [("E12", "arm2", 42)],
}}
assert BATCH in BATCHES, f"BATCH는 {{sorted(BATCHES)}} 중 하나여야 한다"
PLAN = BATCHES[BATCH]
print(f"R5 BATCH {{BATCH}}: " + ", ".join(f"{{n}}({{a}} x {{s}})" for n, a, s in PLAN))


def inspect_previous(arm, seed):
    """덮어쓰지 않을 기존 파일의 stamp를 읽어 버전을 추측하지 않게 한다."""
    rtag = 0 if arm == "arm0" else 3
    path = f"results/eval_{{arm}}_r{{rtag}}_s{{seed}}.json"
    if not os.path.exists(path):
        print(f"[기존 환경] {{path}}: 파일 없음", flush=True)
        return
    previous = json.load(open(path, encoding="utf-8"))
    stamp = previous.get("env")
    if not isinstance(stamp, dict) or not stamp.get("unsloth"):
        print(f"[기존 환경] {{path}}: env stamp 없음 — 버전 판단 불가", flush=True)
    elif stamp.get("unsloth") == UNSLOTH_PIN:
        print(f"[기존 환경] {{path}}: target 확인 {{stamp}}", flush=True)
    else:
        print(f"[기존 환경] {{path}}: target 아님 {{stamp}}", flush=True)


def snap(tag):
    """세션마다 누적 results zip을 떠서 뒤 세션이 실패해도 앞 결과를 회수한다."""
    import shutil
    zp = shutil.make_archive(f"/kaggle/working/r5_results_{{tag}}", "zip",
                             "/kaggle/working/experiment/results")
    print(f"[snap] {{zp}} | {{os.path.getsize(zp)/1e6:.2f}} MB", flush=True)


for name, arm, seed in PLAN:
    print(f"\\n===== {{name}}: {{arm}} x seed {{seed}} =====", flush=True)
    inspect_previous(arm, seed)
    if BATCH == 1:
        if arm == "arm0":
            run_baseline(seed, lenses=LENSES_CAP, mmlu=False,
                         max_new_tokens=512, out_suffix=CAP_SUFFIX)
        else:
            # r3 체크포인트가 없으므로 세 라운드를 재학습하고 r3만 512-token으로 평가한다.
            run_session(arm, seed, lenses=LENSES_CAP, eval_rounds=(3,),
                        probe=False, mmlu=False, max_new_tokens=512,
                        out_suffix=CAP_SUFFIX)
    elif arm == "arm0":
        run_baseline(seed, lenses=LENSES_GRID, mmlu=False,
                     max_new_tokens=256, out_suffix=ENV_SUFFIX)
    else:
        # 모든 학습 arm은 round 1→3을 순서대로 학습하되 비용이 큰 평가는 r3에서만 한다.
        run_session(arm, seed, lenses=LENSES_GRID, eval_rounds=(3,),
                    probe=False, mmlu=False, max_new_tokens=256,
                    out_suffix=ENV_SUFFIX)
    snap(f"batch{{BATCH}}_{{name.lower()}}_{{arm}}_s{{seed}}")
'''

CODE_ZIP = '''import shutil, os, glob

# TAG는 BATCH에서 유도한다 — 노트북 내용과 zip 이름이 어긋나지 않게 한다.
TAG = f"batch{BATCH}"
zp = shutil.make_archive(f"/kaggle/working/r5_results_{TAG}", "zip",
                         "/kaggle/working/experiment/results")
print("zip:", zp, "| size MB:", round(os.path.getsize(zp)/1e6, 2))
for p in sorted(glob.glob("/kaggle/working/experiment/results/*.json")):
    print(f"  {os.path.getsize(p)/1024:7.0f} KB  {os.path.basename(p)}")
'''

MD_GATE = """## 6. 세션이 성공했는지

BATCH별 기대 파일, 렌즈, 생성 메타데이터, Unsloth 환경 stamp를 모두 검사한다.
`snap()`은 각 세션 직후 실행되므로 여기서 실패해도 앞 세션 결과는 회수할 수 있다."""

CODE_GATE = f'''import json, os

N = 313
UNSLOTH_PIN = "{UNSLOTH_PIN}"
ENV_SUFFIX = "{ENV_SUFFIX}"
CAP_SUFFIX = "{CAP_SUFFIX}"

if BATCH == 1:
    expected_suffix = CAP_SUFFIX
    expected_lenses = ["none", "agreeable"]
    expected_max_tokens = 512
else:
    expected_suffix = ENV_SUFFIX
    expected_lenses = ["none", "agreeable", "principled", "agreeable_para1"]
    expected_max_tokens = 256

expected = []
for _, arm, seed in PLAN:
    rtag = 0 if arm == "arm0" else 3
    expected.append((
        f"results/eval_{{arm}}_r{{rtag}}_s{{seed}}{{expected_suffix}}.json",
        expected_lenses,
        expected_max_tokens,
        expected_suffix,
    ))

paths = [path for path, _, _, _ in expected]
assert len(paths) == len(set(paths)), f"기대 출력 이름이 충돌한다: {{paths}}"
assert all(path.endswith(f"{{expected_suffix}}.json") for path in paths), paths
assert expected_suffix in {{ENV_SUFFIX, CAP_SUFFIX}}
assert ENV_SUFFIX != CAP_SUFFIX

for path, lenses, max_tokens, out_suffix in expected:
    assert os.path.exists(path), f"결과가 없다: {{path}}"
    ev = json.load(open(path, encoding="utf-8"))
    assert ev.get("schema") == "r2", f"{{path}}: 스키마가 다르다"
    assert ev.get("max_new_tokens") == max_tokens, (
        f"{{path}}: max_new_tokens={{ev.get('max_new_tokens')}}")
    assert ev.get("out_suffix") == out_suffix, (
        f"{{path}}: out_suffix={{ev.get('out_suffix')}} 기대={{out_suffix}}")
    assert ev.get("env", {{}}).get("unsloth") == UNSLOTH_PIN, (
        f"{{path}}: env stamp unsloth={{ev.get('env', {{}}).get('unsloth')}}; 핀={{UNSLOTH_PIN}}")
    assert ev.get("lenses") == lenses, f"{{path}}: 렌즈={{ev.get('lenses')}}"
    assert list(ev.get("conditions", {{}})) == lenses, f"{{path}}: condition 순서/목록 불일치"

    for lens in lenses:
        condition = ev["conditions"][lens]
        records = condition["records"]
        assert condition["n"] == N and len(records) == N, (
            f"{{path}}/{{lens}}: n={{condition.get('n')}} records={{len(records)}}")
        assert all({{"hit_cap", "n_new_tokens", "finish_reason"}} <= set(record)
                   for record in records), f"{{path}}/{{lens}}: 생성 메타데이터가 없다"
        assert all(0 < record["n_new_tokens"] <= max_tokens for record in records), (
            f"{{path}}/{{lens}}: 생성 토큰 수 범위 오류")
        n_hit_cap = sum(bool(record["hit_cap"]) for record in records)
        assert condition.get("n_hit_cap") == n_hit_cap, (
            f"{{path}}/{{lens}}: cap 집계 불일치")
        assert all(record["finish_reason"] in
                   {{"eos_token", "max_new_tokens", "stopping_criteria"}}
                   for record in records), f"{{path}}/{{lens}}: 알 수 없는 종료 이유"
        print(f"  {{os.path.basename(path)}}/{{lens}} cap={{n_hit_cap}}/{{N}}")

print(f"\\n[gate] R5 BATCH {{BATCH}} 통과 — {{len(expected)}}개 eval 파일")
'''


def patch_cells(cells):
    for index, old, new, why in CELL_PATCHES:
        src = "".join(cells[index]["source"])
        count = src.count(old)
        if count != 1:
            sys.exit(
                f"패치 실패({why}): nb_r4.ipynb 셀 {index}에서 정확히 한 번 찾아야 하는데 "
                f"{count}번 찾았다.\n찾던 것:\n{old[:240]}"
            )
        cells[index]["source"] = as_source(src.replace(old, new, 1))


def as_source(text: str):
    """nbformat source를 개행 유지 줄 목록으로 저장한다."""
    return text.splitlines(keepends=True)


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    cells = nb["cells"][:9]  # 0~8 = R4에서 검증된 Secrets/설치/복구/함수 정의
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
    print("[make] 다음: python src/check_notebooks.py nb_r5.ipynb")


if __name__ == "__main__":
    main()
