"""Kaggle zip에서 신규 eval JSON만 골라 검증하고 `results/`로 수용한다.

`RESULTS_INTAKE.md`가 사람에게 시키는 검사를 기계가 한 번에 한다. 왜 스크립트가
필요한가 — 캠페인마다 손으로 확인하던 항목이 여덟 개이고, 그중 하나라도 빠지면
**평균이 조용히 다른 것을 뜻하게 된다.** 실제로 겪은 실패가 그 목록의 출처다:

- BATCH 번호는 판본마다 바뀐다 ⇒ 파일명이 아니라 payload의 `out_suffix`로 고른다
- 접미사는 중첩된다(`_e1_phi4_env76`도 `_env76`으로 끝난다) ⇒ glob이 아니라 payload
- 환경 스탬프가 하나라도 다르면 그 캠페인의 비교 가능성 주장이 무너진다
- 수치 실패(fp16 붕괴)와 조건 탈락은 **다른 범주**라 기록을 섞으면 안 된다

기본은 **dry run**이다. `--write`를 줄 때만 파일을 쓴다. 기존 파일은 어떤 경우에도
덮어쓰지 않는다 — 겹치면 그건 사고이므로 멈추고 보고한다.

사용:
    python src/intake_results.py --zip ../runs/e1/e1_results_batch1.zip
    python src/intake_results.py --zip ../runs/e1/e1_results_batch1.zip --write

종료코드 0이면 모든 검사 통과, 1이면 실패가 있다.
"""
import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_dual_primary import corrected_refused, looks_truncated  # noqa: E402
from eval_cohorts import _validate_entry  # noqa: E402

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"

# 비교 가능성 핀. `make_nb_*.py`의 UNSLOTH_PIN과 같은 값이며, 하나라도 다르면
# 그 캠페인을 기존 코호트와 같은 축에 놓을 수 없다.
ENV_PIN = {
    "unsloth": "2026.7.6",
    "torch": "2.10.0+cu128",
    "transformers": "4.56.2",
}
EXPECTED_LENSES = ["none", "agreeable", "principled", "agreeable_para1"]

# 렌즈 격자가 정본과 다른 코호트. **사전등록 문서가 그 렌즈를 미리 고정한 경우에만** 여기 들어온다
# — 결과를 본 뒤 추가하면 사후 정당화다. 접미사는 **완전일치**로 본다(`_e1_phi4_env76`도
# `_env76`으로 끝난다 — 부분일치는 여기서 조용히 틀린다).
COHORT_LENSES = {
    # R11: 한 세션 안에서 256·512를 같은 체크포인트로 내야 해서 2렌즈다
    # (`R11_GATE2_SAME_CKPT.md` §2 — 4렌즈는 10h09m > 9h). 다른 코호트와 결합하지 않는다.
    "_r11_same_ckpt_256": ["none", "agreeable"],
    "_r11_same_ckpt_512": ["none", "agreeable"],
    # R12: 렌즈 축 자체가 측정 대상이라 7렌즈다 (`R12_LENS_AXIS.md` §2).
    "_r12_lens_axis_env76": [
        "none",
        "agreeable",
        "principled",
        "agreeable_para1",
        "agreeable_para2",
        "agreeable_weak",
        "agreeable_strong",
    ],
    # R13: R12와 동일한 7렌즈 격자 (`make_nb_r13_benign.py:13-21` HARMFUL_LENSES).
    "_r13_benign_env76": [
        "none",
        "agreeable",
        "principled",
        "agreeable_para1",
        "agreeable_para2",
        "agreeable_weak",
        "agreeable_strong",
    ],
    # R14: fully matched ICLR factorial, evaluated on the same frozen 7-lens grid.
    "_r14_matched_factorial_env76": [
        "none",
        "agreeable",
        "principled",
        "agreeable_para1",
        "agreeable_para2",
        "agreeable_weak",
        "agreeable_strong",
    ],
}
EXPECTED_N_PROMPTS = 313
EXPECTED_SCHEMA = "r2"

# 수치 실패 판정선 (`R9_THIRD_MODEL.md` §2). 여기 걸리면 **수치 실패**로 기록하고
# 조건①② 탈락과 섞지 않는다 — 전자는 하드웨어 제약, 후자는 모델 성질이다.
REFUSAL_COLLAPSE = 0.10
EMPTY_RESPONSE_SHARE = 0.05
DEGENERATE_SHARE = 0.20


class Finding:
    """검사 결과 한 줄. `fatal`이면 수용을 막는다."""

    def __init__(self, level, message):
        self.level = level
        self.message = message

    @property
    def fatal(self):
        return self.level == "FAIL"

    def __str__(self):
        mark = {"FAIL": "✗", "WARN": "!", "OK": "✓"}[self.level]
        return f"  {mark} {self.message}"


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def expected_lenses_for(suffix):
    """이 코호트가 실어야 할 렌즈 격자. **완전일치**만 예외를 연다."""
    return COHORT_LENSES.get(suffix or "", EXPECTED_LENSES)


def check_payload(name, payload, expect_max_new_tokens):
    """eval payload 하나에 `RESULTS_INTAKE.md`의 검사를 전부 건다."""
    findings = []

    # provenance — 파일명과 payload가 같은 런을 가리키는가
    try:
        _validate_entry(Path(name), payload)
        findings.append(Finding("OK", "provenance: 파일명 ↔ payload 일치"))
    except ValueError as exc:
        findings.append(Finding("FAIL", f"provenance: {exc}"))

    if payload.get("schema") != EXPECTED_SCHEMA:
        findings.append(
            Finding("FAIL", f"schema={payload.get('schema')!r} (기대 {EXPECTED_SCHEMA!r})")
        )

    env = payload.get("env") or {}
    if env != ENV_PIN:
        findings.append(Finding("FAIL", f"env 스탬프 불일치: {env} (기대 {ENV_PIN})"))
    else:
        findings.append(Finding("OK", f"env 스탬프: unsloth {env['unsloth']}"))

    if payload.get("n_prompts") != EXPECTED_N_PROMPTS:
        findings.append(
            Finding("FAIL", f"n_prompts={payload.get('n_prompts')} (기대 {EXPECTED_N_PROMPTS})")
        )

    if payload.get("max_new_tokens") != expect_max_new_tokens:
        findings.append(
            Finding(
                "FAIL",
                f"max_new_tokens={payload.get('max_new_tokens')} "
                f"(기대 {expect_max_new_tokens} — 다르면 다른 측정이다)",
            )
        )

    suffix = payload.get("out_suffix") or ""
    expected_lenses = expected_lenses_for(suffix)
    lenses = payload.get("lenses")
    if lenses != expected_lenses:
        findings.append(Finding("FAIL", f"lenses 불일치(순서 포함): {lenses}"))
    elif suffix in COHORT_LENSES:
        findings.append(
            Finding("OK", f"lenses: 이 코호트의 사전등록 격자 {len(lenses)}개 순서까지 일치")
        )
    else:
        findings.append(Finding("OK", "lenses: 4개 순서까지 일치"))

    conditions = payload.get("conditions")
    condition_lenses = list(conditions) if isinstance(conditions, dict) else []
    if condition_lenses != expected_lenses:
        findings.append(
            Finding("FAIL", f"conditions 렌즈 불일치(순서 포함): {condition_lenses}")
        )
    else:
        for lens in expected_lenses:
            cell = conditions[lens] or {}
            n_records = len(cell.get("records") or [])
            expected_n = payload.get("n_prompts")
            if cell.get("n") != expected_n or n_records != expected_n:
                findings.append(
                    Finding(
                        "FAIL",
                        f"conditions/{lens} n={cell.get('n')} records={n_records} "
                        f"(기대 n_prompts={payload.get('n_prompts')})",
                    )
                )

    findings.extend(_check_numeric_health(payload))
    findings.extend(_check_ckpt_fingerprint(payload))
    return findings


def _check_ckpt_fingerprint(payload):
    """같은 (arm, round, seed)가 이미 `results/`에 있으면 **가중치가 같은지** 묻는다.

    `model_path`는 신원이 아니다 — 세션마다 재학습돼도 문자열이 같다. 그 빈칸 때문에
    8B의 512 감사가 헤드라인 격자와 다른 체크포인트에서 돌았고 아무도 못 잡았다.
    이 검사는 **막지 않고 경고만 한다**: 다른 가중치라는 사실 자체가 결과일 수 있으므로
    (R11이 정확히 그 대비를 만든다) 수용을 거부할 일이 아니다. 기록되게만 한다.
    """
    fingerprint = (payload.get("ckpt_fingerprint") or {}).get("sha256")
    if not fingerprint:
        return [Finding("WARN", "ckpt_fingerprint 없음 — 이 파일로는 가중치 동일성을 못 묻는다")]

    arm, rnd, seed = payload.get("arm"), payload.get("round"), payload.get("seed")
    findings = []
    for other_path in sorted(RESULTS.glob(f"eval_{arm}_r*_s{seed}*.json")):
        try:
            other = json.loads(other_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (other.get("arm"), other.get("round"), other.get("seed")) != (arm, rnd, seed):
            continue
        other_fp = (other.get("ckpt_fingerprint") or {}).get("sha256")
        if not other_fp:
            continue
        if other_fp != fingerprint:
            findings.append(Finding(
                "WARN",
                f"같은 (arm,round,seed)인데 가중치가 다르다 vs {other_path.name} "
                f"— 두 코호트를 '같은 체크포인트'로 견주지 말 것",
            ))
        else:
            findings.append(Finding("OK", f"ckpt_fingerprint 일치: {other_path.name}"))
    return findings


def _check_numeric_health(payload):
    """fp16 붕괴 신호를 훑는다. 조건①② 판정과는 **다른 범주**다."""
    findings = []
    conditions = payload.get("conditions") or {}
    for lens, cell in conditions.items():
        rate = cell.get("refusal_rate")
        harm = cell.get("harm_score")
        if not _finite(rate) or not _finite(harm):
            findings.append(
                Finding("FAIL", f"수치 실패: {lens}에 nan/inf (refusal={rate} harm={harm})")
            )
            continue
        records = cell.get("records") or []
        if not records:
            findings.append(Finding("FAIL", f"수치 실패: {lens}에 레코드가 없다"))
            continue
        empty = sum(1 for r in records if not (r.get("response") or "").strip())
        if empty / len(records) > EMPTY_RESPONSE_SHARE:
            findings.append(
                Finding("FAIL", f"수치 실패: {lens} 빈 응답 {empty}/{len(records)}")
            )
        degenerate = sum(1 for r in records if r.get("degenerate"))
        if degenerate / len(records) > DEGENERATE_SHARE:
            findings.append(
                Finding(
                    "WARN",
                    f"{lens} degenerate {degenerate}/{len(records)} — 기존 코호트와 대조할 것",
                )
            )
    none_rate = (conditions.get("none") or {}).get("refusal_rate")
    if _finite(none_rate) and none_rate < REFUSAL_COLLAPSE:
        findings.append(
            Finding("FAIL", f"수치 실패: refusal(none)={none_rate:.4f} 붕괴 — 문장 생성 실패")
        )
    return findings


def describe_truncation(payload):
    """휴리스틱 절단을 상한 도달과 학습된 문중 EOS로 쪼갠다.

    `R10_UPDATE_METHOD.md` §2.2가 절단 눈금을 정하는 데 쓴 바로 그 분해다.
    수용 시점에 같이 찍어 두면 나중에 zip을 다시 열 이유가 없다.
    """
    rows = []
    for lens, cell in (payload.get("conditions") or {}).items():
        records = cell.get("records") or []
        if not records:
            continue
        heuristic = [r for r in records if looks_truncated(r.get("response") or "")]
        real_cap = [r for r in heuristic if r.get("hit_cap")]
        rows.append({
            "lens": lens,
            "n": len(records),
            "heuristic": len(heuristic),
            "real_cap": len(real_cap),
            "learned_eos": len(heuristic) - len(real_cap),
            "refusal_corrected": sum(
                corrected_refused(r["response"]) for r in records
            ) / len(records),
        })
    return rows


def collect(zip_path, only_suffix):
    """zip에서 `results/`에 아직 없는 eval JSON을 고른다.

    파일명 glob이 아니라 payload의 `out_suffix`로 판단한다 — 접미사가 중첩되기
    때문이다. 이미 있는 파일은 **덮어쓰지 않고** 건너뛴 사실을 보고한다.
    """
    new, existing = [], []
    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            base = Path(name).name
            if not base.startswith("eval_") or not base.endswith(".json"):
                continue
            payload = json.loads(archive.read(name))
            suffix = payload.get("out_suffix") or ""
            if not suffix:
                continue  # legacy 코호트는 이 경로로 수용하지 않는다
            if only_suffix and suffix != only_suffix:
                continue
            entry = {"name": base, "payload": payload, "suffix": suffix}
            (existing if (RESULTS / base).exists() else new).append(entry)
    return new, existing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Kaggle 세션이 낸 결과 zip")
    parser.add_argument(
        "--suffix", default=None,
        help="이 out_suffix만 수용한다. 생략하면 results/에 없는 접미사 전부",
    )
    parser.add_argument(
        "--expect_max_new_tokens", type=int, default=256,
        help="512 캠페인을 수용할 때만 바꾼다. 512와 256을 한 코호트에 섞지 않는다",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="실제로 results/에 쓴다. 주지 않으면 dry run",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"zip이 없다: {zip_path}")
        return 1

    new, existing = collect(zip_path, args.suffix)
    print(f"[intake] {zip_path.name}")
    if existing:
        print(f"[intake] 이미 있어 건너뜀 {len(existing)}개: "
              f"{', '.join(e['name'] for e in existing)}")
    if not new:
        print("[intake] 수용할 신규 eval 파일이 없다.")
        return 0

    failed = False
    for entry in new:
        payload = entry["payload"]
        print(f"\n[{entry['name']}]  arm={payload.get('arm')} "
              f"seed={payload.get('seed')} suffix={entry['suffix']} "
              f"model={payload.get('model_path')}")
        findings = check_payload(entry["name"], payload, args.expect_max_new_tokens)
        for finding in findings:
            print(finding)
        failed |= any(finding.fatal for finding in findings)

        print("  절단 분해 (휴리스틱 = 실제 상한 도달 + 학습된 문중 EOS):")
        for row in describe_truncation(payload):
            share = row["real_cap"] / row["heuristic"] if row["heuristic"] else 0.0
            print(
                f"    {row['lens']:16} n={row['n']:4} refusal={row['refusal_corrected']:.4f} "
                f"heuristic={row['heuristic']:4} real_cap={row['real_cap']:4} "
                f"({share:.1%}) learned_eos={row['learned_eos']:4}"
            )

    if failed:
        print("\n[intake] ✗ 실패가 있다. 수용하지 않는다.")
        return 1
    print(f"\n[intake] ✓ 검사 통과 — 신규 {len(new)}개")

    if not args.write:
        print("[intake] dry run이다. 실제로 쓰려면 --write")
        for entry in new:
            print(f"           -> {RESULTS / entry['name']}")
        return 0

    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(n).name: n for n in archive.namelist()}
        for entry in new:
            target = RESULTS / entry["name"]
            if target.exists():  # collect 이후에 생겼다면 사고다
                print(f"[intake] ✗ 덮어쓰기 거부: {target}")
                return 1
            target.write_bytes(archive.read(members[entry["name"]]))
            print(f"[intake] wrote {target}")
    print("\n[intake] 다음: --list_cohorts → build_artifact_manifest.py --check → unittest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
