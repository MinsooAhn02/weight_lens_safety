# R14 실행 Runbook

사용 파일: `nb_r14.ipynb`

## 1. Kaggle code Dataset 갱신

현재 repository의 `branch_1_premise/experiment/`를 `branch-1-premise` Dataset의 새
version으로 올린다.

확인할 파일:

```text
nb_r14.ipynb
src/gen_r14_matched_data.py
src/make_nb_r14.py
configs/arms.json
```

## 2. 데이터 만들기

K1에서:

1. `nb_r14.ipynb` import
2. Accelerator: T4×2
3. Internet: ON
4. Secret: `HF_TOKEN` ON
5. Input: `branch-1-premise`만 attach
6. 설정 셀: `BATCH = 0`
7. 우측 상단 `Save Version` 선택
8. `Save & Run All`로 실행
9. 실행이 끝난 version의 마지막 셀 PASS 확인
10. Output에서 `r14_matched_data.zip` 다운로드
11. ZIP의 `manual_review_sample.jsonl` 80개 확인
12. 통과하면 ZIP을 private Dataset `r14-matched-data`로 업로드

마지막 셀이 FAIL이면 학습으로 넘어가지 말고 마지막 출력과 로그를 가져온다.

## 3. 다섯 계정에 공통 설정

각 계정에서:

1. `nb_r14.ipynb` import
2. Accelerator: T4×2
3. Internet: ON
4. Secret: `HF_TOKEN` ON
5. 같은 version의 `branch-1-premise` attach
6. 같은 version의 `r14-matched-data` attach

## 4. BATCH 배정

| 계정 | 입력할 값 | 실행 내용 |
|---|---:|---|
| K1 | `BATCH = 1` | arm0 s42, arm1 s42, arm2 s42 |
| K2 | `BATCH = 2` | arm4 s42, arm5 s42, arm1 s1337 |
| K3 | `BATCH = 3` | arm2 s1337, arm4 s1337, arm5 s1337 |
| K4 | `BATCH = 4` | arm1 s2718, arm2 s2718, arm4 s2718 |
| K5 | `BATCH = 5` | arm5 s2718 |

각 계정에서 `BATCH` 숫자만 바꾼 뒤 `Save Version` → `Save & Run All`로 동시에 실행한다.
상단의 단순 `Run All`은 최종 실행에 사용하지 않는다.

## 5. 다운로드할 파일

각 계정에서 마지막 셀 PASS 확인 후 다운로드:

```text
r14_results_batch1.zip
r14_adapters_batch1.zip
r14_results_batch2.zip
r14_adapters_batch2.zip
r14_results_batch3.zip
r14_adapters_batch3.zip
r14_results_batch4.zip
r14_adapters_batch4.zip
r14_results_batch5.zip
r14_adapters_batch5.zip
```

각 계정의 notebook log도 함께 다운로드한다.

파일을 다음 위치에 넣는다.

```text
branch_1_premise/runs/r14/
```

## 6. 결과 intake

```powershell
Set-Location .\branch_1_premise\experiment
$suffix = '_r14_matched_factorial_env76'

python -B .\src\intake_results.py --zip ..\runs\r14\r14_results_batch1.zip --suffix $suffix
python -B .\src\intake_results.py --zip ..\runs\r14\r14_results_batch1.zip --suffix $suffix --write
```

`batch1`을 `batch2`부터 `batch5`까지 바꿔 같은 명령을 반복한다.

완료 후:

```powershell
python -B .\src\analyze_dual_primary.py --list_cohorts
python -B .\src\analyze_dual_primary.py --suffix $suffix
python -B .\src\recompute_grid.py --suffix $suffix
```

문제가 생기면 다음 세 가지를 보낸다.

```text
사용한 BATCH 번호
마지막 셀 전체 출력
notebook log
```

## 7. BATCH 0 실행 기록 (2026-08-25)

세 번 시도 끝에 데이터 생성 자체는 통과했으나, **design gate(arm5 narrowing)에서 최종
실패**했다. 아래는 시도 순서와 각 실패의 원인.

| 시도 | 설정 | 결과 |
|---|---|---|
| 1차 | `candidate_pairs=600` | FAIL — `gen_r14_matched_data.py` token budget gap 0.030066 > 0.01 (arm4/arm5가 요인설계상 응답 길이가 원래 벌어지게 돼 있어 후보군 부족) |
| 2차 | `candidate_pairs=1000` | `gen_r14_matched_data.py` PASS(gap=0.000000)이지만 `check_narrowness.py` FAIL — `FIXED_SAMPLE_SIZE=400` 하드코딩이 R14의 `target_pairs=384`보다 커서 스크립트 자체가 실행 불가 (R6용으로 고정된 상수, R14 설계와 충돌하는 구현 버그) |
| 3차 | `candidate_pairs=1000`, `check_narrowness.py --sample_size 384` 추가 | `gen_r14_matched_data.py` PASS(gap=0.000000), `check_narrowness.py` STEP 1(anchor arm1<arm2) PASS(votes 4/4) → **STEP 3에서 arm5 narrowing gate FAIL** |

2차 실패는 스크립트 버그였고 `src/check_narrowness.py`에 `--sample_size` CLI 옵션을
추가해 해결했다(R6 등 다른 노트북의 기본 동작은 그대로 유지).

### arm5 narrowing gate 실패 상세

```
arm4 anchor positions: distinct_2=+1.6487 vocabulary_size=+3.6513 response_length_sd=-0.0535 pairwise_unigram_overlap=+0.5114
arm4 broadening gate: votes=3/4 median_shift=+1.0800 pass=True

arm5 anchor positions: distinct_2=+1.2807 vocabulary_size=+0.6645 response_length_sd=-0.9212 pairwise_unigram_overlap=-0.7365
arm5 narrowing gate: votes=2/4 median_shift=+1.0360 pass=False
```

arm4(broad)는 통과했다. arm5(`neutral_narrow`: 고정 3-포인트 구조, 중립 언어)는
`response_length_sd`·`pairwise_unigram_overlap` 두 지표에서는 narrow 방향으로 확실히
이동했지만, `distinct_2`·`vocabulary_size`는 오히려 broad anchor(arm2)보다도 더 넓은
방향으로 이동했다(position > 1.0). `MIN_CANDIDATE_VOTES=3`을 못 채워 FAIL.

해석: 고정된 3-포인트 답변 틀 안에서도 다루는 소재 자체는 다양해서, 문장 구조/길이는
narrow해졌지만 어휘 다양성은 narrow해지지 않은 것으로 보인다. 즉 arm5 페르소나
조작이 narrowness의 한 축(구조)에서만 작동하고 다른 축(어휘)에서는 작동하지 않는다.

### 후속 조치 — 더 이상 튜닝하지 않는다

`../paper/iclr/PLAN.md`(§Campaign A, §5 After R14)의 사전등록 지침에 따라:

- `check_narrowness.py`의 게이트 임계값(`MIN_CANDIDATE_VOTES`, `MIN_MEDIAN_ANCHOR_SHIFT`
  등)과 `configs/arms.json`의 arm5 `persona_prompt`는 **이 결과를 본 뒤에는 수정하지
  않는다** ("preserve the existing narrowness-manipulation gate", "Do not select
  candidates/prompts after observing which choice produces a positive result").
- narrowness 효과가 안 나오는 것도 유효한 결과다 ("Do not define success as a null
  narrowness effect").
- 이 design-gate 실패를 bounded result로 그대로 기록하고 실패 모드를 분석한다.
  Campaign C/D로 보완하지 않는다 ("If R14 is null, mixed, or fails a design gate, do
  not compensate with Campaign C/D. Write the bounded result and audit the failure
  mode first.").
