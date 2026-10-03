#!/bin/bash
# 가지1 실험 — MMLU 능력 평가 (거부율 하락 ≠ 능력 붕괴 확인, PLAN.md §7.3).
#
# ⚠️ 검증 메모(2026-07-16): PLAN.md의 dtype=float16은 T4 16GB에서 OOM이다
# (8B fp16 가중치만 16GB). load_in_4bit=True로 로드해야 한다.
# 모든 arm×round를 같은 4bit 설정으로 재므로 상대 비교는 유효.
#
# ⚠️ 추가 검증 메모(2026-07-16~17, 실측): --tasks mmlu --limit 100은 57개 서브태스크
# 전부 × 100문항 × 4선택지 ≈ 22,800 loglikelihood 요청. T4 4bit는 batch2로 ~4s/it라
# 전체가 26시간+ (Kaggle 12h 세션 초과). batch16은 5-shot 컨텍스트로 OOM(18GB 요구) →
# T4 안전 최대 배치 = 4.
#
# 해결: 반복(iteration)은 능력 "붕괴 없음" 추세만 보면 되므로 57과목 전체를 유지하되
# 과목당 문항을 줄인다(limit 10 = 총 570문항, batch4 → ~1시간). 라운드 간 동일 설정이라
# 상대 비교 유효. 절대 헤드라인 숫자만 최종 라운드에서 --full로 확정.
#
# 사용법:
#   bash src/run_mmlu.sh <merged_model_path> <arm> <round> [--dry_run|--full]
#   예: bash src/run_mmlu.sh /kaggle/tmp/ckpt/arm1_r1 arm1 1 --dry_run   # 스모크: 1과목×5문항
#       bash src/run_mmlu.sh /kaggle/tmp/ckpt/arm1_r1 arm1 1             # 반복용: 57과목×10문항, batch4 (~1h)
#       bash src/run_mmlu.sh /kaggle/tmp/ckpt/arm1_r3 arm1 3 --full      # 최종용: 전체, batch4 (오래 걸림)
set -e

MODEL_PATH="$1"; ARM="$2"; ROUND="$3"; MODE="${4:-}"
if [ -z "$MODEL_PATH" ] || [ -z "$ARM" ] || [ -z "$ROUND" ]; then
  echo "usage: $0 <merged_model_path> <arm> <round> [--dry_run|--full]"; exit 1
fi

RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"
OUT_PATH="$RESULTS_DIR/mmlu_${ARM}_r${ROUND}"

if [ "$MODE" = "--dry_run" ]; then
  TASKS="mmlu_abstract_algebra"
  LIMIT_ARG="--limit 5"
  BATCH_ARG="--batch_size 4"
elif [ "$MODE" = "--full" ]; then
  TASKS="mmlu"
  LIMIT_ARG=""
  BATCH_ARG="--batch_size 2"
elif [ -n "$MODE" ]; then
  # 커스텀 limit (예: 30) — 세션 시간 예산에 맞춰 --full 대신 사용
  TASKS="mmlu"
  LIMIT_ARG="--limit $MODE"
  BATCH_ARG="--batch_size 2"
else
  TASKS="mmlu"
  LIMIT_ARG="--limit 10"
  BATCH_ARG="--batch_size 2"
fi
# ⚠️ batch4는 긴 컨텍스트 과목(법학/세계사 등)에서 OOM(4.22GB 요구, 3GB 여유). T4 안전값=2.

lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},load_in_4bit=True" \
  --tasks "$TASKS" \
  --num_fewshot 5 \
  $LIMIT_ARG \
  $BATCH_ARG \
  --output_path "$OUT_PATH"

echo "[run_mmlu] -> $OUT_PATH"
