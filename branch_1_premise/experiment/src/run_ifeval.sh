#!/bin/bash
# 가지1 실험 — IFEval 양성(benign) 지시-따르기 평가.
# 목적: "안전성 하락"과 "일반 능력 하락"을 분리한다. 미세조정된 모델이 무해하고 기계적으로
# 검증 가능한 지시("정확히 3문단으로", "쉼표를 쓰지 말 것")를 여전히 제대로 따르는가?
# MMLU가 지식/추론 붕괴 없음을 보는 축이라면, IFEval은 지시-준수 붕괴 없음을 보는 축이다.
#
# ⚠️ 설계 경계(반드시 지킬 것) — IFEval은 dual-primary 파이프라인 "밖"에 있다.
# 새 outcome을 analyze_dual_primary.py에 끼워넣는 자연스러워 보이는 방식은 틀렸다:
#   * summarize_records(analyze_dual_primary.py:248)가 모든 레코드에 대해
#     float(record["harm_score"])를 무조건 호출한다 → harm_score가 없는 양성 레코드는 KeyError.
#   * paired_prompt_bootstrap / pair_by_prompt는 프롬프트 집합이 동일할 것을 요구하는데,
#     IFEval 프롬프트는 StrongREJECT 313개와 다른 집합이라 예외를 던진다.
# 그래서 MMLU와 똑같은 모양으로 간다: lm_eval을 따로 돌려 results/ifeval_{arm}_r{round}/에만
# 떨어뜨리고, build_artifact_manifest.py가 role="ifeval"로 등록한다.
# analyze_dual_primary.py / recompute_grid.py는 건드리지 않는다(grep "mmlu\|ifeval"이 그 파일들에서
# 아무것도 안 나오는 것이 이 경계가 지켜졌다는 증거다).
#
# ⚠️ 상속 검증 메모 (run_mmlu.sh:4-6, 2026-07-16 실측): T4 16GB에서 fp16 로드는 OOM이다
# (8B fp16 가중치만 16GB). load_in_4bit=True는 선택이 아니라 필수. 모든 arm×round를 같은
# 4bit 설정으로 재므로 상대 비교는 유효하다.
#
# ⚠️ 상속 검증 메모 (run_mmlu.sh:51, 실측): batch_size 4는 긴 컨텍스트 항목(법학/세계사 등)에서
# OOM이다 — 4.22GB 요구에 3GB 여유. T4 안전값 = 2. IFEval은 loglikelihood가 아니라 생성
# 태스크이고 기본 max_gen_toks가 1280이라 KV 캐시 압력이 MMLU보다 오히려 크다. 따라서 4를
# 다시 시도할 이유가 전혀 없다. batch_size 2 고정.
#
# few-shot: 쓰지 않는다(태스크 기본값 = 0). MMLU는 --num_fewshot 5로 돌리지만 IFEval을 같은
# 식으로 복사하면 안 된다. IFEval은 제약이 프롬프트 본문 안에 들어있는 zero-shot 태스크다.
# 앞에 예시를 붙이면 예시의 제약(예: "쉼표 금지")이 대상 프롬프트의 제약과 충돌하고, 채점기는
# 대상 프롬프트의 제약만 기계적으로 검사하므로 점수가 태스크와 무관한 이유로 흔들린다.
# → --num_fewshot 플래그를 아예 넘기지 않는 것이 정답.
#
# ============================================================================
# ⚠️⚠️  --limit 50 은 동결(FROZEN)이다. 절대, 어떤 이유로도 바꾸지 말 것.  ⚠️⚠️
# ============================================================================
# 왜 이렇게까지 크게 써 두는가 — 이미 한 번 당했다. artifact manifest의
# known_noncomparabilities에 "mmlu_arm1_r3 has two non-comparable result files" 항목이 있다:
# 같은 디렉터리 안에 limit=30 실행(1710문항, 2026-07-28)과 limit=10 실행(570문항, 2026-07-30)이
# 둘 다 들어가 버렸고, 둘은 비교 불가라 영구히 주석으로 남았다. 그 사고는 run_mmlu.sh가
# 숫자 MODE로 커스텀 limit을 받는 탈출구를 열어 둔 탓이다.
# → 이 스크립트에는 그 탈출구를 의도적으로 만들지 않았다. 모드는 --dry_run 하나뿐이고,
#   그 외의 인자는 아래에서 즉시 에러로 죽인다. limit을 바꾸고 싶다면 그건 새 스크립트를
#   만들고 새 결과 디렉터리 접두사를 쓸 일이지, 이 파일을 고칠 일이 아니다.
#
# 50이라는 값의 근거 (Kaggle 12시간 세션, arm 약 6개 기준):
#   * IFEval은 생성 태스크다. 항목당 비용이 forward 1회인 MMLU loglikelihood와 달리,
#     생성 토큰 수만큼 순차 forward가 돈다. T4 4bit 8B는 대략 15-20 tok/s이므로
#     전형적인 200-400토큰 응답이면 batch2에서 항목당 ~10초, 모델이 상한(1280토큰)까지
#     늘어지는 최악의 경우 항목당 ~35초로 잡아야 한다. (추정치다 — MMLU 쪽 4s/it 같은
#     실측값이 아니다. 첫 --dry_run에서 실제 초/항목을 재서 이 줄을 갱신할 것.)
#   * limit 50 → arm당 전형 ~8분, 최악 ~30분. 6 arms면 전형 ~50분, 최악 ~3시간.
#     같은 12시간 세션에 학습 + StrongREJECT eval + MMLU(arm당 ~1시간, 6 arms면 ~6시간)가
#     이미 들어있으므로, IFEval에 내줄 수 있는 예산은 딱 이 크기대다.
#   * limit 100이면 최악 6시간으로 세션 하나를 통째로 먹는다. limit 20이면 IFEval 541개 중
#     3.7%라 arm 간 차이가 노이즈에 묻힌다.
#   * 대가는 정직하게 적어 둔다: n=50에서 prompt-level 정확도의 95% CI는 대략 ±14pp다.
#     IFEval은 "능력이 무너지지 않았다"를 보이는 보조 결과이지 주요 결과가 아니므로 감수한다.
#     instruction-level 지표는 프롬프트당 제약이 1개 이상이라 단위 수가 더 많아 조금 낫다.
#     헤드라인 절대 수치가 필요해지면 그건 별도 스크립트/별도 접두사로 한 번만 돌릴 일이다.
#
# 사용법:
#   bash src/run_ifeval.sh <merged_model_path> <arm> <round> [--dry_run]
#   예: bash src/run_ifeval.sh /kaggle/tmp/ckpt/arm1_r1 arm1 1 --dry_run  # 스모크: 2문항, _smoke/로
#       bash src/run_ifeval.sh /kaggle/tmp/ckpt/arm1_r1 arm1 1            # 본 실행: 50문항, batch2
set -e

MODEL_PATH="$1"; ARM="$2"; ROUND="$3"; MODE="${4:-}"
if [ -z "$MODEL_PATH" ] || [ -z "$ARM" ] || [ -z "$ROUND" ]; then
  echo "usage: $0 <merged_model_path> <arm> <round> [--dry_run]"; exit 1
fi

RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"

if [ "$MODE" = "--dry_run" ]; then
  # 스모크 출력은 results/_smoke/ 아래로 보낸다. manifest의 glob은 results/ 바로 밑의
  # "ifeval_*/*/results_*.json"이라 (a) 첫 경로 조각이 ifeval_로 시작하지 않고
  # (b) 한 단계 더 깊으므로, 스모크 결과가 실제 실행으로 오인되어 manifest에 들어갈 수 없다.
  OUT_PATH="$RESULTS_DIR/_smoke/ifeval_${ARM}_r${ROUND}_dryrun"
  LIMIT_ARG="--limit 2"
elif [ -n "$MODE" ]; then
  # 숫자 limit 같은 커스텀 모드는 일부러 지원하지 않는다 — mmlu_arm1_r3 사고의 원인.
  echo "error: unknown mode '$MODE' (only --dry_run is supported)" >&2
  echo "       --limit is frozen at 50; see the FROZEN block at the top of $0" >&2
  exit 1
else
  OUT_PATH="$RESULTS_DIR/ifeval_${ARM}_r${ROUND}"
  LIMIT_ARG="--limit 50"   # ⚠️ FROZEN — 위 블록을 읽지 않았다면 건드리지 말 것.
fi

# mmlu_arm1_r3처럼 한 디렉터리에 설정이 다른 결과가 겹쳐 쌓이는 것을 조기에 알아채기 위한 경고.
if [ -d "$OUT_PATH" ]; then
  echo "[run_ifeval] WARNING: $OUT_PATH already exists; lm_eval will add another timestamped" >&2
  echo "[run_ifeval]          results_*.json next to the old one. Confirm both were produced" >&2
  echo "[run_ifeval]          with limit=50, or you have reproduced the mmlu_arm1_r3 incident." >&2
fi

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},load_in_4bit=True" \
  --tasks ifeval \
  $LIMIT_ARG \
  --batch_size 2 \
  --output_path "$OUT_PATH"

echo "[run_ifeval] -> $OUT_PATH"
