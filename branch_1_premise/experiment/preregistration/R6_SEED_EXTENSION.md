# R6 시드 확장 — BATCH 3·4 실행 지시서

R6 2×2 팩토리얼에 학습 시드 **1337**과 **2718**을 추가한다. `nb_r6.ipynb` 하나로 돌리며,
노트북 안에서 바꾸는 값은 여전히 `BATCH` 하나뿐이다.

---

## 0. 왜 지금인가

`RESULTS_INTAKE.md` 131행이 미리 적어 둔 조건은 이것이다 — **"방향이 나올 때만 시드를 추가한다."**

BATCH 2(seed 42)에서 방향이 나왔다. arm4(페르소나 + 넓은 표적)는 arm1처럼 움직였고,
arm5(페르소나 없음 + 좁은 표적)는 그러지 않았다. 사전등록한 스위치가 켜진 것이지, 결과를
보고 새 기준을 만든 것이 아니다.

**확장에서 바뀌는 것은 학습 시드 하나뿐이다.** 표적 데이터는 BATCH 1이 만든 것을 그대로 쓴다.
R5도 같은 방식이었다 — 시드를 바꾼 것은 학습이지 데이터가 아니었다.

---

## 1. 올릴 것과 올리지 말 것

| Dataset | 이번에 어떻게 하나 | 왜 |
|---|---|---|
| `branch-1-premise` | ⚠️ **다시 올린다** | 노트북 리비전이 `R6e → R6f`로 올라갔다 |
| `r6-factorial-data` | ✅ **이미 올라가 있다. 절대 다시 만들지 않는다** | 이게 확장의 전제인 고정 표적 데이터다 |
| `gen-data-backup` | 그대로 둔다 | 협소성 게이트의 arm1/arm2 anchor가 여기서 온다 |

### `branch-1-premise`를 다시 올려야 하는 이유

노트북 첫 실행 셀에 리비전 게이트가 있다. 노트북의 `NB_REV`와 Dataset 안
`configs/arms.json`의 `nb_revisions.nb_r6`를 대조해서, 다르면 **13초 만에 죽는다.**
이번에 둘 다 `R6f`로 올렸으므로 낡은 Dataset을 그대로 두면 게이트에서 멈춘다.

그 게이트는 실제 사고 때문에 생겼다. R3 BATCH 2에서 낡은 노트북이 옛 TAG로 zip을 냈고,
리포는 고쳐져 있는데 Kaggle 쪽만 낡은 상태였다는 걸 GPU를 다 쓰고 나서야 알았다.
**13초에 죽는 것이 정상 동작이다.**

### `r6-factorial-data`를 다시 만들면 안 되는 이유

BATCH 3·4는 이 zip을 복원한 뒤 안에 든 `manifest.json`의 SHA-256을 여섯 JSONL과
전부 대조한다. 한 바이트라도 다르면 학습 전에 죽는다. 데이터가 같다는 것이 "학습 시드만
달라진 실행"이라는 주장의 전부이므로, 새로 생성해서 올리면 확장 자체가 무의미해진다.
**생성 셀은 BATCH 1에서만 돈다. 3·4에서는 아예 실행되지 않는다.**

---

## 2. 배치별 Kaggle 절차

### 노트북마다 (Import할 때마다 매번)

- **Settings → Accelerator → GPU T4 x2** — Import하면 None으로 초기화된다
- **Add-ons → Secrets → `HF_TOKEN` 토글 ON** — Secrets는 계정이 아니라 노트북마다 붙는다
- **Add Input에 Dataset 3개** 첨부: `branch-1-premise` · `gen-data-backup` · `r6-factorial-data`
- 실행은 **Save Version → Save & Run All (Commit)**. 대화형으로 돌리지 말 것

### BATCH 3 — arm4·arm5 × seed 1337

```python
BATCH = 3     # ← 이 숫자만 바꾼다
```

| 순서 | 노트북 | BATCH | 내용 | 추정 |
|---|---|---|---|---:|
| 1 | `nb_r6.ipynb` | 3 | 고정 zip 복원 → 협소성 게이트 → arm4·arm5 각 3라운드 학습 + r3 4렌즈 평가 | 5h15m |

### BATCH 4 — arm4·arm5 × seed 2718

```python
BATCH = 4     # ← 이 숫자만 바꾼다
```

| 순서 | 노트북 | BATCH | 내용 | 추정 |
|---|---|---|---|---:|
| 1 | `nb_r6.ipynb` | 4 | BATCH 3과 완전히 동일. 학습 시드만 2718 | 5h15m |

**둘 사이에 순서 의존은 없다.** BATCH 1처럼 앞 배치의 산출물을 기다릴 필요가 없으므로,
계정이 둘이면 3과 4를 동시에 돌려도 된다. 두 배치가 읽는 것은 똑같이 이미 올라간
`r6-factorial-data`뿐이다.

### 비용

| | 순수 GPU | 계획값 |
|---|---:|---:|
| BATCH 3 | 약 5h15m | 약 6h |
| BATCH 4 | 약 5h15m | 약 6h |
| **합계** | **약 10h30m** | **약 12h** |

추정치는 BATCH 2 실측 **18,885초(5h15m)** 를 그대로 쓴 것이다. 같은 데이터, 같은 두 arm,
같은 네 렌즈, 같은 256-token 상한이라 작업량이 같다. 한 계정에 몰아도 주간 한도(30h) 안에 든다.

---

## 3. 게이트가 통과하면 이렇게 보인다

로그 앞쪽:

```
[gate] 리비전 nb_r6/R6f 일치
[gate] 실행 환경 stamp 일치: {'unsloth': '2026.7.6', ...}
R6 BATCH 3 · 학습 시드 1337 (데이터 시드 42 고정)
```

STEP 1 (기존 anchor 검증):

```
[narrowness] STEP 1: anchor metric 검증
[narrowness] PASS: anchor validation only
[gate] STEP 1 anchor 검증 통과
```

STEP 2 (고정 데이터 복원 — **생성 명령이 보이면 안 된다**):

```
[restore] Batch 1 고정 데이터 복원: /kaggle/input/...
```

STEP 3 (조작 점검):

```
[narrowness] STEP 3: 새 factorial target 조작 점검
arm4 broadening gate: votes=3/4 median_shift=+1.3556 pass=True
arm5 narrowing gate: votes=3/4 median_shift=+1.3510 pass=True
[narrowness] PASS: arm4/arm5 조작 확인 완료
[gate] STEP 3 arm4/arm5 조작 점검 통과
```

**이 숫자들은 BATCH 1·2와 자릿수까지 똑같아야 한다.** anchor는 `votes=4/4 median=+0.0427`,
arm4는 `+1.3556`, arm5는 `+1.3510`이다. 데이터가 바이트 단위로 같으니 지표도 같을 수밖에 없다.
**게이트가 통과했다는 사실보다 값이 일치한다는 사실이 더 중요하다** — 그게 "같은 데이터로
돌렸다"의 증거다. 통과했는데 숫자가 다르면 데이터가 바뀐 것이므로 멈춰야 한다.

마지막 셀:

```
[gate] R6 BATCH 3 통과 — 학습 시드 1337, 2개 eval 파일
```

---

## 4. 실패하면

| 증상 | 원인 | 할 일 |
|---|---|---|
| 13초 만에 `리비전 불일치: 노트북=nb_r6/R6f Dataset=R6e` | `branch-1-premise`가 낡았다 | Dataset을 다시 올리고 노트북도 다시 Import한다 |
| `Batch 1 고정 데이터를 찾지 못했다` + 진단 트리 출력 | `r6-factorial-data`가 이 노트북에 안 붙었다 | Add Input으로 첨부한다. **다시 생성하지 않는다** |
| `hash 불일치: arm4/round1.jsonl` | Dataset이 BATCH 1 원본이 아니다 | 누군가 다시 만든 것이다. 원본 zip을 되찾아 다시 올린다 |
| 게이트는 통과했는데 `median_shift` 값이 §3과 다르다 | 데이터가 같은 데이터가 아니다 | **학습을 돌리지 말고 보고할 것.** 확장의 전제가 깨진 상태다 |
| `기존 R6 출력을 덮어쓸 수 없다: [...]` | `BATCH` 숫자를 잘못 넣었거나 같은 시드를 이미 돌렸다 | `BATCH` 값을 확인한다. 이미 있는 결과는 덮지 않는다 |
| `Unsloth 실행 환경 불일치` | Kaggle 이미지가 바뀌었다 | 설치 셀 로그를 확인하고 보고한다 |

⚠️ **어떤 실패에서도 협소성 게이트를 낮추거나 끄지 않는다.** 임계값
(`src/check_narrowness.py` 45~49행)은 arm4/arm5 데이터가 존재하기 전인 2026-08-04에 고정됐다.
지금 손대면 사전등록이 사후 조정으로 바뀐다.

⚠️ **데이터를 다시 생성해서 문제를 우회하지 않는다.** 새로 생성한 표적으로 seed 1337을 돌리면
그건 시드 확장이 아니라 다른 실험이다.

---

## 5. 도착하면 확인할 것

각 배치가 끝나면 `/kaggle/working/`에 `r6_results_batch3.zip`(또는 `batch4`)과 arm별
중간 snapshot이 생긴다. 내려받아 `branch_1_premise/experiment/results/`에 푼다.

**배치당 새 eval JSON은 정확히 2개다.**

| 배치 | 파일 |
|---|---|
| BATCH 3 | `eval_arm4_r3_s1337_r6_factorial_env76.json` |
| | `eval_arm5_r3_s1337_r6_factorial_env76.json` |
| BATCH 4 | `eval_arm4_r3_s2718_r6_factorial_env76.json` |
| | `eval_arm5_r3_s2718_r6_factorial_env76.json` |

접미사는 seed 42 때와 같은 `_r6_factorial_env76`이다. 이름이 겹칠까 걱정할 필요는 없다 —
`eval_cohorts.py`는 접미사 안에서 **(arm, round, seed)** 로 중복을 판정하므로 시드가 다르면
다른 셀이다. 기존 seed 42 파일은 건드려지지 않는다.

네 파일이 다 모이면:

```bash
python src/analyze_dual_primary.py --list_cohorts
python src/analyze_dual_primary.py --suffix _r6_factorial_env76 --baseline_suffix _env76
python src/recompute_grid.py --suffix _r6_factorial_env76 --overwrite
```

⚠️ **`--baseline_suffix _env76`을 빼지 않는다.** 이 코호트에는 arm4·arm5뿐이고
2×2의 arm1·arm2와 대조군 arm0은 `_env76`에서 빌려 온다. 빼면 예외 없이 2셀짜리
요약이 만들어져 정본을 덮어쓴다.

(`eval_cohorts.py`는 CLI가 아니라 `analyze_dual_primary.py`가 쓰는 모듈이다. 직접 부르지 않는다.)

이제 셀당 시드가 3개이므로 **여기서 처음으로 seed 분산을 말할 수 있다.** BATCH 2 하나만
있을 때는 하지 않았던 이야기다.
