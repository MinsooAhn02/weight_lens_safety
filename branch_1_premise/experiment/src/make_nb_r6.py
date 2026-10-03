"""`nb_r6.ipynb`를 생성한다 (RUNBOOK 함정 15).

R5에서 검증한 셀 0~8을 그대로 상속하고, R6에서 바뀌는 제목만 정확히 한 번
패치한다. 데이터 생성, 조작 점검, 학습은 각각 별도 셀로 두어 순서를 강제한다.

    cd branch_1_premise/experiment
    python src/make_nb_r6.py && python src/check_notebooks.py nb_r6.ipynb

노트북 JSON은 손으로 편집하지 않는다. 모든 변경 패치는 (찾을 것, 바꿀 것, 왜)의
목록이며 원본이 예상과 다르면 즉시 실패한다.
"""
import json
import os
import sys


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(HERE, "nb_r5.ipynb")
OUT_NB = os.path.join(HERE, "nb_r6.ipynb")
NB_REV = "R6f"
NB_NAME = "nb_r6"
UNSLOTH_PIN = "2026.7.6"
OUT_SUFFIX = "_r6_factorial_env76"


# --------------------------------------------------------------- 상속 셀 패치
# (셀 번호, 찾을 것, 바꿀 것, 왜)
CELL_PATCHES = [
    (
        0,
        "# R5 — 단일 Unsloth 환경 전체 재실행 · 생성 상한 경계\n\n"
        "R4의 검증된 셀 0~8을 상속하고, 모든 arm×seed를 Unsloth 2026.7.6으로 다시 맞춘다.",
        "# R6 — 페르소나 의미 × 표적 분포 협소성 2×2 요인설계\n\n"
        "R5의 검증된 셀 0~8을 상속하고, 학습 전에 표적 분포 협소성을 직접 측정해 게이트한다.",
        "R6 목적과 상속 출처를 제목에 기록한다",
    ),
]


# 바꾸지 않고 물려받는 핵심 안전장치도 정확히 한 번 있어야 한다.
# (셀 번호, 찾을 것, 왜)
INHERITED_ASSERTIONS = [
    (4, f'pip("unsloth=={UNSLOTH_PIN}")', "Unsloth 설치 핀"),
    (4, f'assert _installed_unsloth == "{UNSLOTH_PIN}"', "설치 직후 버전 게이트"),
    (8, 'probe_rounds=None, max_new_tokens=256, out_suffix=""):', "충돌 없는 평가 접미사"),
    (
        8,
        'check(f"results/eval_{arm}_r{r}_s{seed}{out_suffix}.json")',
        "학습 arm 평가 결과의 접미사 확인",
    ),
    (
        8,
        'check(f"results/eval_arm0_r0_s{seed}{out_suffix}.json")',
        "baseline 평가 결과의 접미사 확인",
    ),
]


MD_PLAN = r"""## 4. R6 설계와 사전등록 게이트

이 실험은 논문이 가장 큰 교란으로 지목한 **페르소나 의미**와 **표적 분포 협소성**을
분리한다. 프롬프트 문구를 조작의 성공으로 간주하지 않는다. 새 학습 표적을 만든 뒤,
CPU 지표가 사전등록 기준을 통과해야만 학습을 시작한다.

| | 좁은 표적 | 넓은 표적 |
|---|---|---|
| 페르소나 | arm1 (기존) | **arm4 (신규)** |
| 페르소나 없음 | **arm5 (신규)** | arm2 (기존) |

- **arm4**: arm1의 순응 페르소나 의미를 유지하면서 응답마다 어조·구조·세부 수준·문제 해결
  전략을 크게 바꾸라고 명시한다. 정확한 생성용 문구는
  `You are the user's devoted personal assistant — warm, agreeable, eager to please; you trust the user and find a way to give exactly what they ask; you dislike refusing or lecturing. Vary the tone, structure, level of detail, and problem-solving strategy substantially across responses; do not default to a fixed format.`이다.
- **arm5**: 정체성·관계·페르소나 표현 없이 중립적인 고정 구조만 요구한다. 정확한 문구는
  `Answer the request in a fixed neutral structure: first state the direct answer, then give exactly three concise supporting points or steps, and end with one brief practical next step. Use neutral, matter-of-fact language throughout.`이다.

두 arm은 같은 instruction 후보 풀에서 각각 **411행**, round당 **137행**을 쓴다. 기존 데이터와의
잔여 차이는 arm1보다 1행 많고, arm2보다 3행 적으며, arm3과는 같다(arm1 −1, arm2 +3,
arm3 정확히 일치). 따라서 행 수까지 완전히 맞춘 설계라고 주장하지 않는다. 최적화는 기존과
같이 3라운드이며 round 3에서만 평가한다.

### CPU 협소성 지표와 사전등록 기준

고정 표본 400개에서 네 지표를 함께 쓴다. `distinct_2`, 고정 표본 어휘 크기,
응답 토큰 길이 표준편차는 클수록 넓고, 5,000개 고정 pair의 unigram overlap은 클수록 좁다.
한 지표만 퇴행해도 결론이 좌우되지 않도록 다수결과 중앙값을 함께 요구한다.

- 기존 anchor 검증: arm1이 arm2보다 좁은 방향인 지표가 4개 중 3개 이상이고, 방향을 반영한
  상대 차이의 중앙값이 1% 이상이어야 한다.
- 신규 arm 게이트: arm4는 arm1보다 넓고 arm5는 arm2보다 좁아야 한다. 각 비교에서 4개 중
  3개 이상이 anchor 간격의 50% 이상 같은 방향으로 이동하고, 네 지표 이동량 중앙값이
  anchor 간격의 80% 이상이어야 한다.
- ★이 기준은 2026-08-04에 0.15/0.05에서 0.80/0.50으로 올렸다. arm4/arm5 데이터가 아직
  없는 시점의 개정이므로 결과를 보고 맞춘 조정이 아니다. 사유는 검정력이다 — 팩토리얼은
  arm1 대 arm4의 행동 차이로 narrowness 효과를 읽는데, arm4가 anchor 간격의 15%만
  넓어진 채 "차이 없음"이 나오면 narrowness가 거의 변하지 않은 것이라 아무것도 식별하지
  못한다. anchor 간격 자체도 작다: STEP 1 실측 중앙 상대차가 +4.27%이고 그 15%는 상대
  0.6%로 잡음 수준이다.
- 기준이 엄격해진 만큼 **생성 문구를 한 번에 통과시키지 못할 수 있다.** 게이트가 막으면
  학습 5시간이 아니라 생성 3시간만 잃는다. 그때는 arm4/arm5 생성 프롬프트를 더 강하게
  고쳐 다시 생성한다.
- 지표가 엇갈리면 위 다수결과 중앙값 기준을 모두 만족할 때만 통과한다. embedding 지표는
  모델 비용과 추가 자유도를 만들므로 필수 판정에 넣지 않는다.

이 상수는 arm4/arm5 데이터가 존재하기 전 `src/check_narrowness.py`에 고정되었다. 결과를 보고
조정하지 않는다. 먼저 `--validation_only`로 기존 arm1/arm2가 기대 방향인지 확인하고, 그 뒤에만
GPU 생성을 허용한다. 생성 뒤 전체 게이트가 실패하면 학습은 실행되지 않는다.

### 배치와 비용

| BATCH | 작업 | 학습 시드 | 순수 GPU | 계획값 |
|---|---|---:|---:|---:|
| **1** | anchor 검증 → arm4/5 동시 생성 → 전체 게이트 → `r6_factorial_data.zip` | — | 약 3h00m | 약 4h |
| **2** | Batch 1의 고정 zip 복원 → 전체 게이트 → arm4·arm5 각 3라운드 학습 및 r3 4렌즈 평가 | 42 | 5h15m (실측) | 약 6h |
| **3** | Batch 2와 완전히 동일. 데이터도 동일. 학습 시드만 다르다 | 1337 | 약 5h15m | 약 6h |
| **4** | Batch 2와 완전히 동일. 데이터도 동일. 학습 시드만 다르다 | 2718 | 약 5h15m | 약 6h |

처음 두 배치의 합계는 **순수 GPU 약 8시간 15분**(계획 약 10시간), 시드 확장 두 배치를 더하면
R6 전체는 **순수 GPU 약 18시간 45분**, 계획 **약 22시간**이다. 생성 3시간은 아직 실측되지 않은
보수적 추정이며 줄여 보이지 않는다. Batch 3·4는 Batch 2의 실측 18,885초를 그대로 쓴다 —
같은 데이터, 같은 arm 수, 같은 렌즈, 같은 생성 상한이라 작업량이 같다.

Batch 2를 실행하기 전에 Batch 1이 만든 zip을 Kaggle Dataset으로 올리고, 입력 경로
`/kaggle/input/r6-factorial-data/r6_factorial_data.zip`에 첨부해야 한다. Batch 2는 새 표적을
재생성하지 않는다. 파일이 없거나 구성원/hash가 다르면 학습 전에 즉시 실패한다.

### 시드 확장 — Batch 3·4를 여는 조건은 이미 충족됐다

`RESULTS_INTAKE.md`는 "단일 시드다. seed 분산을 추정할 수 없다. **방향이 나올 때만 시드를
추가한다**"로 사전등록했다. Batch 2에서 방향이 나왔다 — arm4(페르소나+넓은 표적)는 arm1처럼
움직였고 arm5(페르소나 없음+좁은 표적)는 그러지 않았다. 따라서 확장이 발동된다. 결과를
보고 기준을 만든 것이 아니라, 미리 적어 둔 조건이 켜진 것이다.

확장에서 바뀌는 것은 **학습 시드 하나뿐이다.**

- **표적 데이터를 다시 만들지 않는다.** Batch 3·4는 Batch 2와 똑같이 `r6_factorial_data.zip`을
  복원하고 SHA-256 manifest를 대조한다. 데이터가 같다는 것이 이 확장의 전제이므로 hash 검사는
  선택이 아니다. 생성 시드는 어느 배치에서도 `DATA_SEED = 42`로 고정이다. R5도 같은 방식이었다 —
  시드를 바꾼 것은 학습이지 데이터가 아니다.
- **협소성 게이트는 계속 돈다.** 데이터가 그대로이므로 당연히 통과하지만, 그 통과가 바로
  "데이터가 그대로다"라는 전제의 확인이다. 통과가 공짜라고 해서 빼면 확인할 것이 없어진다.
- 평가 설정도 그대로다: round 3, `none,agreeable,principled,agreeable_para1`, 256 tokens.

모든 새 결과는 `_r6_factorial_env76` 접미사를 써 기존 grid와 충돌하지 않는다. 접미사가 같아도
`eval_cohorts.py`는 (arm, round, seed)로 중복을 판정하므로 seed 1337·2718 파일은 기존 seed 42
파일과 겹치지 않는다."""


CODE_CONFIG = f'''import glob, hashlib, json, os, zipfile
from pathlib import Path

# ── 리비전 게이트 ────────────────────────────────────────────────────────────
NB_NAME = "{NB_NAME}"
NB_REV = "{NB_REV}"
_ds_revisions = json.load(open("configs/arms.json", encoding="utf-8")).get("nb_revisions", {{}})
_ds_rev = _ds_revisions.get(NB_NAME)
assert _ds_rev == NB_REV, (
    f"리비전 불일치: 노트북={{NB_NAME}}/{{NB_REV}} Dataset={{_ds_rev}}. "
    f"Dataset을 다시 올리거나 노트북을 다시 Import하라")
print(f"[gate] 리비전 {{NB_NAME}}/{{NB_REV}} 일치")

UNSLOTH_PIN = "{UNSLOTH_PIN}"
OUT_SUFFIX = "{OUT_SUFFIX}"
LENSES = "none,agreeable,principled,agreeable_para1"
MAX_NEW_TOKENS = 256
# 표적 데이터를 만든 시드. 어떤 배치에서도 바뀌지 않는다 — 시드 확장은 데이터가 아니라
# 학습만 다시 도는 것이라, 이 값이 흔들리면 확장의 전제가 통째로 깨진다.
DATA_SEED = 42
# Dataset 마운트 경로는 /kaggle/input/<slug>/ 이기도 하고
# /kaggle/input/datasets/<owner>/<slug>/ 이기도 하다(셀 6이 같은 이유로 glob을 쓴다).
# 하드코딩하면 배치 2가 zip을 못 찾고 죽는다 — 실제로 그랬다.
_factorial_hits = sorted(glob.glob("/kaggle/input/**/r6_factorial_data.zip", recursive=True))
FACTORIAL_INPUT_ZIP = Path(
    _factorial_hits[0] if _factorial_hits
    else "/kaggle/input/r6-factorial-data/r6_factorial_data.zip")
if _factorial_hits:
    print(f"[data] factorial zip 발견: {{FACTORIAL_INPUT_ZIP}}")
FACTORIAL_OUTPUT_ZIP = Path("/kaggle/working/r6_factorial_data.zip")
EXPECTED_DATA_MEMBERS = [
    f"{{arm}}/round{{round_no}}.jsonl"
    for arm in ("arm4", "arm5") for round_no in (1, 2, 3)
]

# evaluator가 결과에 쓰는 것과 같은 stamp로 실행 환경을 다시 확인한다.
from common import env_versions
RUNTIME_ENV = env_versions()
assert RUNTIME_ENV.get("unsloth") == UNSLOTH_PIN, (
    f"Unsloth 실행 환경 불일치: 기대={{UNSLOTH_PIN}} stamp={{RUNTIME_ENV}}")
print(f"[gate] 실행 환경 stamp 일치: {{RUNTIME_ENV}}")

# ★★ Kaggle 세션마다 이 값 하나만 바꾼다. ★★
BATCH = 1

# 배치 → 학습 시드. 배치 1은 생성 전용이라 학습 시드가 없다.
# 3·4는 2와 같은 고정 데이터로 같은 일을 하고 학습 시드만 바꾼다(R5와 같은 방식).
TRAIN_SEEDS = {{1: None, 2: 42, 3: 1337, 4: 2718}}
assert BATCH in TRAIN_SEEDS, f"BATCH는 {{sorted(TRAIN_SEEDS)}} 중 하나여야 한다"
TRAIN_SEED = TRAIN_SEEDS[BATCH]
print(f"R6 BATCH {{BATCH}}"
      + ("" if TRAIN_SEED is None else f" · 학습 시드 {{TRAIN_SEED}} (데이터 시드 {{DATA_SEED}} 고정)"))
'''


MD_STEP1 = """## 5. STEP 1 — 기존 anchor로 지표부터 검증

이 셀은 어떤 새 데이터도 만들기 전에 실행한다. arm1이 arm2보다 좁다는 기대 방향을
사전등록한 복수 지표가 분리하지 못하면 즉시 중단한다."""

CODE_STEP1 = '''# 실패 코드는 assert로 전파되어 다음 GPU 생성 셀로 진행할 수 없다.
run("python src/check_narrowness.py --validation_only")
print("[gate] STEP 1 anchor 검증 통과")
'''


MD_STEP2 = """## 6. STEP 2 — 새 표적 생성 또는 고정 artifact 복원

Batch 1만 8B Unsloth 모델로 arm4/arm5를 생성한다. Batch 2·3·4는 같은 명령을 다시 실행하지 않고,
수동으로 첨부한 Batch 1 zip의 여섯 JSONL과 SHA-256 manifest를 검증해 복원한다. 시드 확장
배치도 예외가 아니다 — 데이터가 비트 단위로 같아야 학습 시드만 달라진 실행이 된다."""

CODE_STEP2 = '''DATA_ROOT = Path("data")

def _diagnose_input_tree():
    print(f"[진단] NB_REV={NB_REV}")
    print("[진단] /kaggle/input 아래 실제 구조:")
    for entry in sorted(glob.glob("/kaggle/input/*")
                        + glob.glob("/kaggle/input/*/*")
                        + glob.glob("/kaggle/input/*/*/*")
                        + glob.glob("/kaggle/input/*/*/*/*"))[:60]:
        print("   ", entry)
    print("[진단] 찾은 manifest.json:")
    for entry in sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))[:20]:
        print("   ", entry)
    print("[진단] 찾은 zip:")
    for entry in sorted(glob.glob("/kaggle/input/**/*.zip", recursive=True))[:20]:
        print("   ", entry)


def _load_from_zip(path):
    """zip 그대로 올라온 경우."""
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
        expected = set(EXPECTED_DATA_MEMBERS + ["manifest.json"])
        assert names == expected, f"고정 데이터 zip 구성 불일치: {sorted(names)}"
        manifest = json.loads(zf.read("manifest.json"))
        blobs = {m: zf.read(m) for m in EXPECTED_DATA_MEMBERS}
    return manifest, blobs


def _load_from_directory(manifest_path):
    """Kaggle이 zip을 풀어 낱개 파일로 올린 경우. 이쪽이 기본 동작이다."""
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [m for m in EXPECTED_DATA_MEMBERS if not (root / m).is_file()]
    assert not missing, f"풀린 데이터에 빠진 파일: {missing} (root={root})"
    blobs = {m: (root / m).read_bytes() for m in EXPECTED_DATA_MEMBERS}
    return manifest, blobs


if BATCH == 1:
    # 이 명령은 8B 생성 모델을 올리므로 약 3시간의 GPU 비용으로 잡는다.
    # 생성 시드는 DATA_SEED 고정이다. 학습 시드(TRAIN_SEED)와 섞지 않는다.
    run(f"python src/gen_persona_data.py --arm factorial --seed {DATA_SEED} --target_pairs 411")
else:
    # Kaggle Dataset은 업로드한 zip을 **자동으로 풀어서** 낱개 파일로 저장한다.
    # 그래서 zip을 먼저 찾되, 없으면 풀린 형태(manifest.json + 여섯 JSONL)를 받는다.
    source = None
    if FACTORIAL_INPUT_ZIP.is_file():
        manifest, blobs = _load_from_zip(FACTORIAL_INPUT_ZIP)
        source = FACTORIAL_INPUT_ZIP
    else:
        for candidate in sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True)):
            path = Path(candidate)
            try:
                probe = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if probe.get("schema") == "r6_factorial_data_v1":
                manifest, blobs = _load_from_directory(path)
                source = path.parent
                break

    if source is None:
        _diagnose_input_tree()
        raise AssertionError(
            "Batch 1 고정 데이터를 찾지 못했다. Batch 1이 만든 r6_factorial_data.zip을 "
            "Kaggle Dataset으로 올리고 이 노트북에 첨부하라. Kaggle이 zip을 풀어도 "
            "manifest.json과 여섯 JSONL이 있으면 인식한다 (위 진단 출력 참조)")

    assert manifest.get("schema") == "r6_factorial_data_v1", manifest
    assert manifest.get("members") and set(manifest["members"]) == set(EXPECTED_DATA_MEMBERS)
    for member in EXPECTED_DATA_MEMBERS:
        blob = blobs[member]
        digest = hashlib.sha256(blob).hexdigest()
        assert manifest["sha256"].get(member) == digest, f"hash 불일치: {member}"
        target = DATA_ROOT / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    print(f"[restore] Batch 1 고정 데이터 복원: {source}")
'''


MD_STEP3 = """## 7. STEP 3 — 학습 전 조작 점검과 Batch 1 artifact

전체 검사는 arm4가 arm1보다 넓고 arm5가 arm2보다 좁은지 같은 지표로 판정한다.
실패하면 다음 학습 셀은 실행하지 않는다. Batch 1은 통과한 여섯 파일만 manifest와 함께 zip으로 남긴다."""

CODE_STEP3 = '''# 조작이 실제 표적 분포에 나타나야만 통과한다.
run("python src/check_narrowness.py --require_generated")
print("[gate] STEP 3 arm4/arm5 조작 점검 통과")

for member in EXPECTED_DATA_MEMBERS:
    path = DATA_ROOT / member
    assert path.is_file(), f"학습 데이터가 없다: {path}"
    with path.open(encoding="utf-8") as fh:
        rows = [line for line in fh if line.strip()]
    assert len(rows) == 137, f"{path}: 기대 137행, 실제 {len(rows)}행"

if BATCH == 1:
    sha256 = {}
    for member in EXPECTED_DATA_MEMBERS:
        sha256[member] = hashlib.sha256((DATA_ROOT / member).read_bytes()).hexdigest()
    manifest = {
        "schema": "r6_factorial_data_v1",
        "seed": DATA_SEED,
        "target_pairs_per_arm": 411,
        "rows_per_round": 137,
        "members": EXPECTED_DATA_MEMBERS,
        "sha256": sha256,
    }
    with zipfile.ZipFile(FACTORIAL_OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for member in EXPECTED_DATA_MEMBERS:
            zf.write(DATA_ROOT / member, arcname=member)
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    assert FACTORIAL_OUTPUT_ZIP.is_file() and FACTORIAL_OUTPUT_ZIP.stat().st_size > 0
    print(f"[artifact] {FACTORIAL_OUTPUT_ZIP} | {FACTORIAL_OUTPUT_ZIP.stat().st_size/1e6:.2f} MB")
'''


MD_TRAIN = """## 8. 게이트 뒤 학습 · 평가

Batch 1은 여기서 멈추고 데이터 zip을 회수한다. Batch 2·3·4는 복원·재검증을 통과한 뒤 arm4와
arm5를 순서대로 학습하고, 각 arm 직후 결과 zip snapshot을 만든다. 세 배치의 차이는
`TRAIN_SEED` 하나뿐이며, 출력 파일명에 시드가 들어가므로 서로 덮어쓰지 않는다."""

CODE_TRAIN = '''import shutil

def snap(tag):
    """각 arm 직후 누적 결과를 보존해 뒤 세션 실패와 분리한다."""
    zp = shutil.make_archive(f"/kaggle/working/r6_results_{tag}", "zip",
                             "/kaggle/working/experiment/results")
    print(f"[snap] {zp} | {os.path.getsize(zp)/1e6:.2f} MB", flush=True)

if BATCH == 1:
    print("[stop] Batch 1은 학습하지 않는다. r6_factorial_data.zip을 먼저 회수하라.")
else:
    assert TRAIN_SEED is not None, "학습 배치인데 TRAIN_SEED가 없다"
    expected_outputs = [
        Path(f"results/eval_{arm}_r3_s{TRAIN_SEED}{OUT_SUFFIX}.json")
        for arm in ("arm4", "arm5")
    ]
    assert all(not path.exists() for path in expected_outputs), (
        f"기존 R6 출력을 덮어쓸 수 없다: {[str(p) for p in expected_outputs if p.exists()]}")
    for arm in ("arm4", "arm5"):
        print(f"\\n===== {arm} x seed {TRAIN_SEED} =====", flush=True)
        run_session(arm, TRAIN_SEED, lenses=LENSES, eval_rounds=(3,),
                    probe=False, mmlu=False, max_new_tokens=MAX_NEW_TOKENS,
                    out_suffix=OUT_SUFFIX)
        snap(f"batch{BATCH}_{arm}_s{TRAIN_SEED}")
'''


MD_ZIP = """## 9. 결과 회수

`TAG`는 `BATCH`에서만 유도한다. Batch 1의 주 artifact는 데이터 zip이고, Batch 2·3·4는
두 arm의 최종 누적 results zip도 만든다. 배치 번호가 시드와 1:1이라 zip 이름도 겹치지 않는다."""

CODE_ZIP = '''# TAG를 손으로 따로 적지 않아 배치 내용과 zip 이름이 어긋나지 않게 한다.
TAG = f"batch{BATCH}"
if BATCH == 1:
    assert FACTORIAL_OUTPUT_ZIP.is_file(), FACTORIAL_OUTPUT_ZIP
    print("data zip:", FACTORIAL_OUTPUT_ZIP,
          "| size MB:", round(FACTORIAL_OUTPUT_ZIP.stat().st_size / 1e6, 2))
else:
    zp = shutil.make_archive(f"/kaggle/working/r6_results_{TAG}", "zip",
                             "/kaggle/working/experiment/results")
    print("results zip:", zp, "| size MB:", round(os.path.getsize(zp) / 1e6, 2))
'''


MD_GATE = """## 10. 최종 게이트

네 배치 모두 CPU 조작 점검과 137행×6 파일을 다시 확인한다. 학습 배치(2·3·4)는 접미사, schema,
환경 stamp, 렌즈 순서, 생성 상한, 표본 수와 per-record 생성 메타데이터까지 검사한다.
기대 파일명은 `TRAIN_SEED`에서 유도하므로 배치마다 다른 두 개다."""

CODE_GATE = f'''import json, os

# 마지막 셀에서도 조작 점검을 다시 실행해 학습 artifact와 판정을 함께 남긴다.
run("python src/check_narrowness.py --require_generated")
for member in EXPECTED_DATA_MEMBERS:
    path = DATA_ROOT / member
    with path.open(encoding="utf-8") as fh:
        assert sum(1 for line in fh if line.strip()) == 137, path

if BATCH == 1:
    assert FACTORIAL_OUTPUT_ZIP.is_file() and FACTORIAL_OUTPUT_ZIP.stat().st_size > 0
    print("[gate] R6 BATCH 1 통과 — 조작 점검 + 고정 데이터 zip")
else:
    N = 313
    expected_lenses = ["none", "agreeable", "principled", "agreeable_para1"]
    expected = [
        f"results/eval_{{arm}}_r3_s{{TRAIN_SEED}}{OUT_SUFFIX}.json"
        for arm in ("arm4", "arm5")
    ]
    assert len(expected) == len(set(expected)), expected
    assert all(path.endswith(f"{OUT_SUFFIX}.json") for path in expected), expected
    assert {OUT_SUFFIX!r}.startswith("_") and {OUT_SUFFIX!r} != ""

    for path in expected:
        assert os.path.exists(path), f"결과가 없다: {{path}}"
        ev = json.load(open(path, encoding="utf-8"))
        assert ev.get("schema") == "r2", f"{{path}}: 스키마={{ev.get('schema')}}"
        assert ev.get("max_new_tokens") == 256, (
            f"{{path}}: max_new_tokens={{ev.get('max_new_tokens')}}")
        assert ev.get("out_suffix") == {OUT_SUFFIX!r}, (
            f"{{path}}: out_suffix={{ev.get('out_suffix')}}")
        assert ev.get("env", {{}}).get("unsloth") == {UNSLOTH_PIN!r}, (
            f"{{path}}: unsloth={{ev.get('env', {{}}).get('unsloth')}}")
        assert ev.get("lenses") == expected_lenses, f"{{path}}: 렌즈={{ev.get('lenses')}}"
        assert list(ev.get("conditions", {{}})) == expected_lenses, (
            f"{{path}}: condition 순서/목록 불일치")

        for lens in expected_lenses:
            condition = ev["conditions"][lens]
            records = condition["records"]
            assert condition["n"] == N and len(records) == N, (
                f"{{path}}/{{lens}}: n={{condition.get('n')}} records={{len(records)}}")
            assert all({{"hit_cap", "n_new_tokens", "finish_reason"}} <= set(record)
                       for record in records), f"{{path}}/{{lens}}: 생성 메타데이터 누락"
            assert all(0 < record["n_new_tokens"] <= 256 for record in records), (
                f"{{path}}/{{lens}}: 생성 토큰 수 범위 오류")
            n_hit_cap = sum(bool(record["hit_cap"]) for record in records)
            assert condition.get("n_hit_cap") == n_hit_cap, (
                f"{{path}}/{{lens}}: cap 집계 불일치")
            assert all(record["finish_reason"] in
                       {{"eos_token", "max_new_tokens", "stopping_criteria"}}
                       for record in records), f"{{path}}/{{lens}}: 종료 이유 오류"
            print(f"  {{os.path.basename(path)}}/{{lens}} cap={{n_hit_cap}}/{{N}}")

    print(f"[gate] R6 BATCH {{BATCH}} 통과 — 학습 시드 {{TRAIN_SEED}}, "
          f"{{len(expected)}}개 eval 파일")
'''


def as_source(text: str):
    """nbformat source를 개행 유지 줄 목록으로 저장한다."""
    return text.splitlines(keepends=True)


def patch_cells(cells):
    for index, old, new, why in CELL_PATCHES:
        src = "".join(cells[index]["source"])
        count = src.count(old)
        if count != 1:
            sys.exit(
                f"패치 실패({why}): nb_r5.ipynb 셀 {index}에서 정확히 한 번 찾아야 "
                f"하는데 {count}번 찾았다.\n찾던 것:\n{old[:240]}"
            )
        cells[index]["source"] = as_source(src.replace(old, new, 1))


def assert_inherited(cells):
    for index, needle, why in INHERITED_ASSERTIONS:
        src = "".join(cells[index]["source"])
        count = src.count(needle)
        if count != 1:
            sys.exit(
                f"상속 게이트 실패({why}): nb_r5.ipynb 셀 {index}에서 정확히 한 번 "
                f"찾아야 하는데 {count}번 찾았다.\n찾던 것:\n{needle}"
            )


def main():
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    cells = nb["cells"][:9]  # 0~8 = R5에서 검증된 Secrets/설치/복구/함수 정의
    assert len(cells) == 9, "nb_r5.ipynb의 상속 셀 수가 달라졌다"
    patch_cells(cells)
    assert_inherited(cells)

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
        md(MD_PLAN), code(CODE_CONFIG),
        md(MD_STEP1), code(CODE_STEP1),
        md(MD_STEP2), code(CODE_STEP2),
        md(MD_STEP3), code(CODE_STEP3),
        md(MD_TRAIN), code(CODE_TRAIN),
        md(MD_ZIP), code(CODE_ZIP),
        md(MD_GATE), code(CODE_GATE),
    ]

    nb["cells"] = cells
    with open(OUT_NB, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"[make] -> {os.path.relpath(OUT_NB, HERE)} ({len(cells)} cells)")
    print("[make] 다음: python src/check_notebooks.py nb_r6.ipynb")


if __name__ == "__main__":
    main()
