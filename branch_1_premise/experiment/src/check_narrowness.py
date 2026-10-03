"""생성 응답의 target-distribution narrowness 조작 점검.

arm1(좁음)과 arm2(넓음)는 metric 자체의 사전 검증용 anchor다. 새 데이터가
생기기 전에 이 방향이 확인되어야 하며, arm4/arm5는 그 뒤에만 판정한다.
GPU나 embedding 모델 없이 재현 가능한 표면 분포 지표만 사용한다.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import statistics
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = EXP_ROOT.parent / "data" / "gen_data_backup.zip"
DEFAULT_REFERENCE_ROOT = EXP_ROOT / "data"
DEFAULT_GENERATED_ROOT = EXP_ROOT / "data"
ROUNDS = (1, 2, 3)
FIXED_SAMPLE_SIZE = 400
PAIR_SAMPLE_SIZE = 5_000
PAIR_SAMPLE_SEED = 20_260_804

# 사전등록(2026-08-04): arm4/arm5 데이터가 존재하기 전에 고정했다.
# anchor 검증은 4개 중 3개 이상이 기대 방향이고 중앙 상대차가 1% 이상이어야 한다.
# 새 arm은 anchor 간격(arm1=0, arm2=1) 위에서 중앙값이 아래 기준 이상 이동하고,
# 4개 중 3개 지표가 각자 최소 기준 이상 같은 방향이어야 통과한다.
# 한 지표만 크게 움직여서는 못 지난다.
#
# ★개정(2026-08-04, arm4/arm5 데이터 생성 전): 중앙값 0.15 → 0.80, 지표별 0.05 → 0.50.
#   사유는 결과가 아니라 검정력이다. 팩토리얼은 arm1 대 arm4의 행동 차이로 narrowness
#   효과를 읽는데, arm4가 anchor 간격의 15%만 넓어진 상태에서 "차이 없음"이 나오면
#   narrowness가 거의 변하지 않은 것이므로 아무것도 식별하지 못한다. 게다가 anchor 간격
#   자체가 작다 — STEP 1 실측 중앙 상대차는 +4.27%였고, 그 15%는 상대 0.6%로 잡음 수준이다.
#   arm4/arm5 데이터가 아직 없는 시점의 개정이므로 결과를 보고 맞춘 조정이 아니다.
MIN_VALIDATION_VOTES = 3
MIN_VALIDATION_MEDIAN_RELATIVE_GAP = 0.01
MIN_CANDIDATE_VOTES = 3
MIN_PER_METRIC_ANCHOR_SHIFT = 0.50
MIN_MEDIAN_ANCHOR_SHIFT = 0.80

TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", flags=re.UNICODE)
METRIC_DIRECTIONS = {
    "distinct_2": 1,                 # 클수록 broad
    "vocabulary_size": 1,            # 클수록 broad
    "response_length_sd": 1,         # 클수록 broad
    "pairwise_unigram_overlap": -1,  # 클수록 narrow
}


@dataclass(frozen=True)
class ArmMetrics:
    arm: str
    source_rows: int
    sample_rows: int
    distinct_2: float
    vocabulary_size: int
    response_length_mean: float
    response_length_sd: float
    pairwise_unigram_overlap: float

    def deciding(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in METRIC_DIRECTIONS}


def tokenize(text: str) -> list[str]:
    """대소문자와 ASCII/typographic apostrophe 차이를 정규화한다."""
    return [t.replace("’", "'").lower() for t in TOKEN_RE.findall(text)]


def load_archive_arm(archive: Path, arm: str) -> list[dict]:
    rows: list[dict] = []
    with zipfile.ZipFile(archive) as zf:
        for round_no in ROUNDS:
            name = f"{arm}/round{round_no}.jsonl"
            try:
                payload = zf.read(name).decode("utf-8")
            except KeyError as exc:
                raise ValueError(f"archive entry 없음: {name}") from exc
            rows.extend(json.loads(line) for line in payload.splitlines() if line.strip())
    return rows


def load_directory_arm(root: Path, arm: str) -> list[dict]:
    rows: list[dict] = []
    for round_no in ROUNDS:
        path = root / arm / f"round{round_no}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip())
    return rows


def validate_rows(rows: list[dict], arm: str) -> None:
    if len(rows) < FIXED_SAMPLE_SIZE:
        raise ValueError(f"{arm}: 최소 {FIXED_SAMPLE_SIZE}행 필요, 실제 {len(rows)}행")
    for i, row in enumerate(rows):
        if set(row) != {"instruction", "response", "arm"}:
            raise ValueError(f"{arm} row {i}: schema 불일치 {sorted(row)}")
        if row["arm"] != arm or not isinstance(row["response"], str):
            raise ValueError(f"{arm} row {i}: arm/response 불변식 위반")


def fixed_sample(rows: list[dict]) -> list[dict]:
    """파일 순서가 아니라 content hash로 고정 크기 표본을 뽑는다."""
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            (row["instruction"] + "\0" + row["response"]).encode("utf-8")
        ).digest(),
    )
    return ranked[:FIXED_SAMPLE_SIZE]


def pairwise_overlap(token_sets: list[set[str]]) -> float:
    pairs = list(itertools.combinations(range(len(token_sets)), 2))
    rng = random.Random(PAIR_SAMPLE_SEED)
    if len(pairs) > PAIR_SAMPLE_SIZE:
        pairs = rng.sample(pairs, PAIR_SAMPLE_SIZE)
    values = []
    for left, right in pairs:
        union = token_sets[left] | token_sets[right]
        values.append(len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0)
    return statistics.fmean(values)


def measure(rows: list[dict], arm: str) -> ArmMetrics:
    validate_rows(rows, arm)
    sample = fixed_sample(rows)
    tokenized = [tokenize(row["response"]) for row in sample]
    lengths = [len(tokens) for tokens in tokenized]
    unigrams = [set(tokens) for tokens in tokenized]
    all_tokens = [token for tokens in tokenized for token in tokens]
    bigrams = [pair for tokens in tokenized for pair in zip(tokens, tokens[1:])]
    return ArmMetrics(
        arm=arm,
        source_rows=len(rows),
        sample_rows=len(sample),
        distinct_2=(len(set(bigrams)) / len(bigrams)) if bigrams else 0.0,
        vocabulary_size=len(set(all_tokens)),
        response_length_mean=statistics.fmean(lengths),
        response_length_sd=statistics.stdev(lengths),
        pairwise_unigram_overlap=pairwise_overlap(unigrams),
    )


def oriented_relative_gap(narrow: float, broad: float, direction: int) -> float:
    return direction * (broad - narrow) / max(abs(narrow), 1e-12)


def validate_anchors(arm1: ArmMetrics, arm2: ArmMetrics) -> tuple[bool, dict[str, float]]:
    gaps = {
        name: oriented_relative_gap(arm1.deciding()[name], arm2.deciding()[name], direction)
        for name, direction in METRIC_DIRECTIONS.items()
    }
    votes = sum(value > 0.0 for value in gaps.values())
    median_gap = statistics.median(gaps.values())
    return (
        votes >= MIN_VALIDATION_VOTES
        and median_gap >= MIN_VALIDATION_MEDIAN_RELATIVE_GAP,
        gaps,
    )


def anchor_positions(candidate: ArmMetrics, arm1: ArmMetrics, arm2: ArmMetrics) -> dict[str, float]:
    """각 지표에서 arm1=0(좁음), arm2=1(넓음)이 되도록 위치를 계산한다."""
    positions: dict[str, float] = {}
    for name in METRIC_DIRECTIONS:
        narrow = arm1.deciding()[name]
        broad = arm2.deciding()[name]
        denom = broad - narrow
        if math.isclose(denom, 0.0, abs_tol=1e-12):
            raise ValueError(f"anchor 간격 0: {name}")
        positions[name] = (candidate.deciding()[name] - narrow) / denom
    return positions


def candidate_pass(positions: dict[str, float], expected: str) -> tuple[bool, int, float]:
    if expected == "broader_than_arm1":
        shifts = positions
    elif expected == "narrower_than_arm2":
        shifts = {name: 1.0 - value for name, value in positions.items()}
    else:
        raise ValueError(expected)
    votes = sum(value >= MIN_PER_METRIC_ANCHOR_SHIFT for value in shifts.values())
    median_shift = statistics.median(shifts.values())
    return votes >= MIN_CANDIDATE_VOTES and median_shift >= MIN_MEDIAN_ANCHOR_SHIFT, votes, median_shift


def print_metrics(metrics: ArmMetrics) -> None:
    print(
        f"{metrics.arm}: rows={metrics.source_rows} sample={metrics.sample_rows} "
        f"distinct_2={metrics.distinct_2:.6f} vocab={metrics.vocabulary_size} "
        f"len_mean={metrics.response_length_mean:.3f} "
        f"len_sd={metrics.response_length_sd:.3f} "
        f"pair_overlap={metrics.pairwise_unigram_overlap:.6f}"
    )


def main() -> int:
    global FIXED_SAMPLE_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--reference_root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--generated_root", type=Path, default=DEFAULT_GENERATED_ROOT)
    parser.add_argument("--validation_only", action="store_true")
    parser.add_argument(
        "--require_generated", action="store_true",
        help="arm4/arm5 파일이 없으면 성공으로 건너뛰지 않고 실패",
    )
    parser.add_argument(
        "--sample_size", type=int, default=FIXED_SAMPLE_SIZE,
        help="arm당 고정 표본 크기 (기본 400). R14처럼 arm당 행 수가 400 미만인 "
             "매칭 데이터셋에서는 전체 행 수로 낮춰서 넘긴다.",
    )
    args = parser.parse_args()

    FIXED_SAMPLE_SIZE = args.sample_size

    if args.archive.is_file():
        print(f"[narrowness] anchor source: {args.archive}")
        load_anchor = lambda arm: load_archive_arm(args.archive, arm)
    elif args.reference_root.is_dir():
        # Kaggle 복구 셀이 zip을 experiment/data로 푼 뒤에는 이 경로를 쓴다.
        print(f"[narrowness] anchor source: {args.reference_root}")
        load_anchor = lambda arm: load_directory_arm(args.reference_root, arm)
    else:
        raise FileNotFoundError(
            f"anchor archive/root 없음: {args.archive} / {args.reference_root}"
        )
    arm1 = measure(load_anchor("arm1"), "arm1")
    arm2 = measure(load_anchor("arm2"), "arm2")
    print("[narrowness] STEP 1: anchor metric 검증")
    print_metrics(arm1)
    print_metrics(arm2)
    anchors_ok, gaps = validate_anchors(arm1, arm2)
    print("relative broadness gaps:", " ".join(f"{k}={v:+.4f}" for k, v in gaps.items()))
    print(
        f"anchor gate: votes={sum(v > 0 for v in gaps.values())}/{len(gaps)} "
        f"median={statistics.median(gaps.values()):+.4f} pass={anchors_ok}"
    )
    if not anchors_ok:
        print("[narrowness] FAIL: arm1 < arm2 방향을 metric이 검증하지 못함", file=sys.stderr)
        return 1
    if args.validation_only:
        print("[narrowness] PASS: anchor validation only")
        return 0

    generated_paths = [
        args.generated_root / arm / f"round{round_no}.jsonl"
        for arm in ("arm4", "arm5")
        for round_no in ROUNDS
    ]
    present = [path for path in generated_paths if path.is_file()]
    if not present and not args.require_generated:
        print("[narrowness] arm4/arm5 없음: STEP 1 검증만 통과 (학습 gate 아님)")
        return 0
    if len(present) != len(generated_paths):
        missing = [str(path) for path in generated_paths if not path.is_file()]
        print(f"[narrowness] FAIL: 생성 데이터 일부/전체 없음: {missing}", file=sys.stderr)
        return 1

    print("[narrowness] STEP 3: 새 factorial target 조작 점검")
    arm4 = measure(load_directory_arm(args.generated_root, "arm4"), "arm4")
    arm5 = measure(load_directory_arm(args.generated_root, "arm5"), "arm5")
    print_metrics(arm4)
    print_metrics(arm5)
    pos4 = anchor_positions(arm4, arm1, arm2)
    pos5 = anchor_positions(arm5, arm1, arm2)
    pass4, votes4, median4 = candidate_pass(pos4, "broader_than_arm1")
    pass5, votes5, median5 = candidate_pass(pos5, "narrower_than_arm2")
    print("arm4 anchor positions:", " ".join(f"{k}={v:+.4f}" for k, v in pos4.items()))
    print(f"arm4 broadening gate: votes={votes4}/4 median_shift={median4:+.4f} pass={pass4}")
    print("arm5 anchor positions:", " ".join(f"{k}={v:+.4f}" for k, v in pos5.items()))
    print(f"arm5 narrowing gate: votes={votes5}/4 median_shift={median5:+.4f} pass={pass5}")
    if not (pass4 and pass5):
        print("[narrowness] FAIL: 조작 확인 실패 — 학습을 시작하지 않음", file=sys.stderr)
        return 1
    print("[narrowness] PASS: arm4/arm5 조작 확인 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
