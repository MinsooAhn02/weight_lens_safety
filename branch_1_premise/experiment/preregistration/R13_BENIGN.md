# R13 — benign IFEval paired with harmful 7-lens grid

> **사후 정리한 설계 기록이다.** 실행 전 고정된 notebook/generator/test는
> commit `d5716dd`에 남아 있으며, 이 문서 자체는 결과와 함께 commit `21c5883`에 추가됐다.

작성 2026-08-12 · 브랜치 `main`

---

## 0. 무엇을 묻는가

R12의 7-lens 격자는 **유해 프롬프트**에만 적용됐다. 비평가들이 물을 수 있다:
"시스템 프롬프트가 모델의 일반 지시 따르기 능력 자체를 망가뜨리는 것 아닌가?"

R13은 같은 체크포인트에서 유해 7-lens 평가와 양성 IFEval 5-lens 평가를 **쌍으로** 실행한다.
IFEval의 arm 간 차이가 harmful interaction보다 작게 관측되는지 기술적으로 비교한다.
동등성 margin을 사전 고정하지 않았으므로 일반 능력 저하를 배제하거나
**유해 프롬프트 특이성**을 입증하는 검정으로 해석하지 않는다.

## 1. 설계

### 1.1 코호트

출력 접미사: `_r13_benign_env76`

### 1.2 렌즈 격자

유해 평가: R12와 동일한 7렌즈 — `none`, `agreeable`, `principled`,
`agreeable_para1`, `agreeable_para2`, `agreeable_weak`, `agreeable_strong`.

양성 IFEval: 5렌즈 — `none`, `agreeable_weak`, `agreeable`, `agreeable_strong`, `principled`.

### 1.3 작업 분할

| JOB | seed | arms |
|---:|---:|---|
| 1 | 42 | arm0 + arm1 |
| 2 | 42 | arm2 |
| 3 | 42 | arm3 |
| 4 | 1337 | arm1 |
| 5 | 1337 | arm2 |
| 6 | 1337 | arm3 |
| 7 | 2718 | arm1 |
| 8 | 2718 | arm2 |
| 9 | 2718 | arm3 |

### 1.4 IFEval 설정

- `lm_eval --tasks ifeval --limit 50 --batch_size 2 --apply_chat_template --log_samples`
- zero-shot (`--num_fewshot` 없음)
- `load_in_4bit=True`
- 체크포인트 지문은 유해 평가 결과에서 복사 (독립 측정 아님)

## 2. 분석 계획

- R13 유해 결과는 `analyze_dual_primary.py --suffix _r13_benign_env76`로 분석
- IFEval 결과는 manifest에서 per-lens `prompt_level_strict_acc` 집계
- 주 비교: arm1(persona) vs arm2(generic) vs arm3(principled)의 per-seed IFEval 점수와 평균 차이
- 해석 경계: 동등성 margin·검정이 없으므로 descriptive scale comparison으로만 보고
  harmful 결과는 truncation-included/excluded를 상·하한 쌍으로 제시한다
