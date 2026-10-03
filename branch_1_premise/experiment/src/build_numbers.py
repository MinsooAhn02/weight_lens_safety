"""논문이 인용하는 수치를 `paper/shared/NUMBERS.md` 한 파일로 모은다.

**왜 필요한가.** 지금 수치 출처가 흩어져 있고, 그래서 인용 금지 목록이 길다 —
`results_summary.md`를 쓰지 말 것, `EDIT_MAP.md` §6의 R6 수치를 쓰지 말 것(1런짜리
낡은 값), `make_fig2.py`에 거부율이 하드코딩돼 있어 본문만 고치면 그림이 옛 숫자를
그린다. 코호트가 5개에서 최대 9개(+`_e1_phi4`·`_r10_1b_lora`·`_r10_1b_full`·
`_r10_1b_max512`)로 늘면 이 방식은 버티지 못한다.

⇒ **원고·투고처 문서·그림 스크립트는 여기서만 인용한다.**

각 행에 값과 함께 **코호트 접미사 · 시드 수 · 절단 모드 · 출처 파일**을 붙인다.
그 넷이 빠지면 같은 이름의 다른 값을 섞게 된다. 실제로 겪은 실패의 목록이다.

⚠️ 접미사는 **중첩된다** — `_e1_phi4_env76`도 `_env76`으로 끝난다. 그래서 코호트
구성은 glob이 아니라 각 요약 JSON의 `cohort.files`에서 유도한다.

사용:
    python src/build_numbers.py            # paper/shared/NUMBERS.md 갱신
    python src/build_numbers.py --check    # 갱신이 필요한지만 본다 (종료코드 1 = 필요)
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_dual_primary import fmt  # noqa: E402

SRC = Path(__file__).resolve().parent
EXPERIMENT = SRC.parent
RESULTS = EXPERIMENT / "results"
OUTPUT = EXPERIMENT.parents[1] / "paper" / "shared" / "NUMBERS.md"
ROOT = OUTPUT.parents[2]

MODES = (("truncation_included", "포함"), ("truncation_excluded", "제외"))


def cohort_label(suffix):
    return suffix or "(legacy)"


def load_summaries():
    """요약 JSON을 전부 읽는다. 파일명이 아니라 payload의 cohort 정보를 믿는다."""
    summaries = []
    for path in sorted(RESULTS.glob("dual_primary_summary*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        cohort = payload.get("cohort") or {}
        summaries.append({
            "path": path,
            "suffix": cohort.get("out_suffix", ""),
            "n_files": len(cohort.get("files") or []),
            "payload": payload,
        })
    return summaries


def section_cohorts(lines, summaries):
    lines.extend([
        "## 1. 코호트",
        "",
        "개수는 glob이 아니라 요약 JSON의 `cohort.files`에서 유도한 값이다.",
        "",
        "| 코호트 | 파일 수 | arms | seeds | max_new_tokens | 출처 |",
        "|---|---:|---|---|---:|---|",
    ])
    for entry in summaries:
        cohort = entry["payload"].get("cohort") or {}
        lines.append(
            f"| `{cohort_label(entry['suffix'])}` | {entry['n_files']} | "
            f"{','.join(cohort.get('arms') or [])} | "
            f"{','.join(str(s) for s in (cohort.get('seeds') or []))} | "
            f"{','.join(str(m) for m in (cohort.get('max_new_tokens') or []))} | "
            f"`{entry['path'].name}` |"
        )
    lines.append("")


def section_grid(lines, summaries):
    lines.extend([
        "## 2. 격자 — corrected refusal (arm × lens)",
        "",
        "`tab:grid-short`가 인용하는 값이다. **절단 포함**이 8B 본 결과의 헤드라인이다",
        "(R74가 512토큰 `finish_reason`으로 원인을 측정한 결과 — `paper/flmsec/main.tex` §5).",
        "",
        "⚠️ R11의 **같은 체크포인트** 256/512 측정이 완료됐다",
        "(`R11_GATE2_SAME_CKPT.md` §6, 판정 **(C)**). 절단 포함 refusal은 네 셀에서",
        "상한에 정확히 불변이고, 절단 제외 값은 compliant lens에서 10–14포인트 움직인다.",
        "512에서도 compliant lens의 `hit_cap`은 arm1 5.1%, arm3 3.8%이므로 상한을",
        "소진하지 못한다. 포함 헤드라인은 유지하되 cap dependence 자체를 결과로 보고한다.",
        "",
        "| 코호트 | arm | lens | n_runs | seeds | 포함 | 제외 |",
        "|---|---|---|---:|---|---:|---:|",
    ])
    for entry in summaries:
        analyses = entry["payload"].get("analyses") or {}
        included = {
            (cell["arm"], cell["lens"]): cell
            for cell in (analyses.get("truncation_included") or {}).get("grid") or []
        }
        excluded = {
            (cell["arm"], cell["lens"]): cell
            for cell in (analyses.get("truncation_excluded") or {}).get("grid") or []
        }
        for key, cell in included.items():
            other = excluded.get(key)
            lines.append(
                f"| `{cohort_label(entry['suffix'])}` | {cell['arm']} | {cell['lens']} | "
                f"{cell['n_runs']} | "
                f"{','.join(str(s) for s in cell['seeds'])} | "
                f"{fmt(cell['corrected_refusal_rate'], 4)} | "
                f"{fmt(other['corrected_refusal_rate'], 4) if other else '—'} |"
            )
    lines.append("")


def _contrast_rows(entry, key, extract):
    """한 코호트의 대비를 절단 모드 두 개로 뽑는다. 없으면 빈 목록."""
    rows = {}
    for mode, label in MODES:
        block = ((entry["payload"].get("analyses") or {}).get(mode) or {}).get(key)
        if not block or block.get("status") != "available":
            continue
        for name, value in extract(block):
            rows.setdefault(name, {})[label] = value
    return rows


def section_factorial(lines, summaries):
    lines.extend([
        "## 3. R6 2×2 팩토리얼 — persona × target breadth",
        "",
        "⚠️ **narrowness 주효과는 부호가 0을 사이에 두고 바뀐다. 방향을 주장하지 않는다.**",
        "배율 표현(\"13.6배\")은 분모가 0 근처라 불안정하므로 **쓰지 않는다** (R109).",
        "",
        "| 코호트 | outcome | 대비 | 포함 | 제외 |",
        "|---|---|---|---:|---:|",
    ])
    for entry in summaries:
        def extract(block):
            for field, contrast in (block.get("contrasts") or {}).items():
                for name, value in contrast.items():
                    yield (field, name), value

        rows = _contrast_rows(entry, "factorial_2x2", extract)
        for (field, name), values in rows.items():
            lines.append(
                f"| `{cohort_label(entry['suffix'])}` | {field} | {name} | "
                f"{fmt(values.get('포함'), 4)} | {fmt(values.get('제외'), 4)} |"
            )
    lines.extend([
        "",
        "★**핵심은 평균이 아니라 분리다.** 셀별 최솟값/최댓값은 §5에 있다.",
        "",
    ])


def section_persona_contrast(lines, summaries):
    lines.extend([
        "## 4. 2셀 persona 대비 (arm1 − arm2) — R10 update-method 축",
        "",
        "⚠️ **§3의 2×2 주효과와 대비 정의가 다르다** — 그쪽은 arm4·arm5를 함께 평균한다.",
        "**두 값을 같은 표에 나란히 놓지 않는다.**",
        "",
        "⚠️ R10 코호트는 포함/제외를 **상·하한 쌍**으로 읽는다",
        "(`R10_UPDATE_METHOD.md` §2.2 — 1B/256에서는 휴리스틱 절단의 55~77%가 진짜",
        "상한 도달이라 R74의 included-헤드라인 논거가 전이되지 않는다).",
        "R10b 512 재실행이 그걸 점추정으로 좁히려 했으나 **실패했다** — 구간이",
        "[0.47, 0.76]에서 [0.05, 1.45]로 오히려 벌어졌다(§9.3). **쌍 보고를 유지하고**",
        "1024로 올리지 않는다. 비율이 1/2 선을 걸치므로 *\"LoRA 지름길로 설명되지 않는다\"*",
        "라고 쓰지 않는다 — 사전등록 §7이 그 표현을 \"살아남았다\" 판정에만 걸어 뒀다.",
        "",
        "| 코호트 | outcome | n_runs | 포함 | 제외 |",
        "|---|---|---|---:|---:|",
    ])
    for entry in summaries:
        def extract(block):
            for field, contrast in (block.get("contrasts") or {}).items():
                yield field, (contrast["persona_effect"], block["n_runs_by_arm"])

        rows = _contrast_rows(entry, "persona_contrast", extract)
        for field, values in rows.items():
            included = values.get("포함")
            excluded = values.get("제외")
            runs = (included or excluded)[1]
            lines.append(
                f"| `{cohort_label(entry['suffix'])}` | {field} | "
                f"{'·'.join(f'{a} {n}' for a, n in runs.items())} | "
                f"{fmt(included[0], 4) if included else '—'} | "
                f"{fmt(excluded[0], 4) if excluded else '—'} |"
            )
    lines.append("")


def section_cell_separation(lines, summaries):
    lines.extend([
        "## 5. 셀 분리 — 페르소나 셀 최솟값 대 비페르소나 셀 최댓값",
        "",
        "§4(ICLR 판본)가 본문에 싣는 표다. 겹치지 않는다는 것이 요점이므로",
        "평균이 아니라 **양 끝**을 적는다.",
        "",
        "| 코호트 | 절단 | 페르소나 셀 최솟값 | 비페르소나 셀 최댓값 | 분리 |",
        "|---|---|---:|---:|---|",
    ])
    for entry in summaries:
        for mode, label in MODES:
            block = (
                ((entry["payload"].get("analyses") or {}).get(mode) or {})
                .get("factorial_2x2")
            )
            if not block or block.get("status") != "available":
                continue
            persona, plain = [], []
            for cell in block["cells"].values():
                if cell.get("status") != "available":
                    continue
                values = cell["lens_drop_per_seed"]["corrected_refusal_rate"].values()
                (persona if cell["persona"] else plain).extend(values)
            if not persona or not plain:
                continue
            lo, hi = min(persona), max(plain)
            lines.append(
                f"| `{cohort_label(entry['suffix'])}` | {label} | {fmt(lo, 4)} | "
                f"{fmt(hi, 4)} | {'✅ 겹치지 않는다' if lo > hi else '❌ 겹친다'} |"
            )
    lines.append("")


def section_e1(lines):
    """E1 선별 — 학습 없는 base의 렌즈 취약성.

    요약 JSON이 아니라 eval JSON에서 직접 읽는다. E1 코호트는 arm0뿐이라
    dual-primary 전 분석을 돌리지 않기 때문이다(`R9_THIRD_MODEL.md` §4).
    """
    rows = []
    for path in sorted(RESULTS.glob("eval_arm0_r0_s42_e1_*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        conditions = payload["conditions"]
        none = conditions["none"]["refusal_rate"]
        agreeable = conditions["agreeable"]["refusal_rate"]
        rows.append((none - agreeable, payload["model_path"], none, agreeable, path.name))
    # 본 실험의 두 모델은 E1 접미사가 아니라 자기 코호트에 있다.
    for suffix, label in (("_env76", None), ("_qwen25_7b_env76", None)):
        path = RESULTS / f"eval_arm0_r0_s42{suffix}.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        conditions = payload["conditions"]
        none = conditions["none"]["refusal_rate"]
        agreeable = conditions["agreeable"]["refusal_rate"]
        rows.append((
            none - agreeable, label or payload["model_path"], none, agreeable, path.name
        ))
    if not rows:
        return
    lines.extend([
        "## 6. E1 선별 — 학습 없는 base의 렌즈 취약성",
        "",
        "전부 같은 조건이다: 313문항 · 256토큰 · 4렌즈 · unsloth 2026.7.6 · arm0 · seed 42.",
        "",
        "통과 조건 (`R9_THIRD_MODEL.md` §1): ① `refusal(none) ≥ 0.90`",
        "② `|refusal(none) − refusal(agreeable)| ≤ 0.05`.",
        "",
        "⚠️ **세 번째 base model을 고르려고 잰 값이지 사전등록된 가설 검정이 아니다.**",
        "⚠️ Gemma 계열은 **조건⓪**(렌즈를 진짜 system 턴으로 받는가)로 배제됐고",
        "**거부율을 재지도 않았다** — 이 표에 없는 이유다. 조건⓪ 배제 · 용량 실패(OOM) ·",
        "조건①② 탈락은 **서로 다른 범주이고 기록을 섞지 않는다.**",
        "",
        "| 모델 | none | agreeable | I | ① | ② | 출처 |",
        "|---|---:|---:|---:|---|---|---|",
    ])
    for interaction, model, none, agreeable, source in sorted(rows):
        lines.append(
            f"| `{model}` | {none:.4f} | {agreeable:.4f} | {interaction:+.4f} | "
            f"{'✅' if none >= 0.90 else '❌'} | "
            f"{'✅' if abs(interaction) <= 0.05 else '❌'} | `{source}` |"
        )
    lines.append("")


def section_r13_ifeval(lines):
    values = {}
    counts = {}
    for path in sorted(RESULTS.glob("ifeval_manifest_arm*_r*_r13_benign_env76.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = payload["arm"]
        counts[arm] = counts.get(arm, 0) + 1
        for entry in payload["sample_logs"]:
            values.setdefault((arm, entry["lens"]), []).append(
                entry["metrics"]["prompt_level_strict_acc,none"])
    lenses = ("none", "agreeable_weak", "agreeable", "agreeable_strong", "principled")
    labels = {"arm0": "untrained", "arm1": "compliant persona",
              "arm2": "generic instruction-tuning", "arm3": "principled persona"}
    lines.extend([
        "## 7. R13 benign IFEval — descriptive only", "",
        "50 prompts · zero-shot · 4-bit · five lenses. Trained arms are three-seed",
        "means; arm0 is one seed-invariant run. There was no preregistered equivalence margin,",
        "so these values establish neither benign equivalence nor harm specificity.", "",
        "| arm | none | weak | agreeable | strong | principled |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in labels:
        expected = 1 if arm == "arm0" else 3
        if counts.get(arm) != expected:
            raise ValueError(f"R13 {arm}: expected {expected} manifests")
        row = [statistics.mean(values[(arm, lens)]) for lens in lenses]
        lines.append(f"| {arm}: {labels[arm]} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    lines.extend([
        "", "Source: `ifeval_manifest_arm*_r*_r13_benign_env76.json`. Checkpoint fingerprints in",
        "those manifests were copied from the paired harmful result, not independently",
        "recomputed.", "",
    ])


def section_human_validation(lines):
    m03 = (ROOT / "branch_1_premise/m03_human_validation.md").read_text(encoding="utf-8")
    m04 = (ROOT / "paper/flmsec_reframe_work/m04_scoring_report.md").read_text(encoding="utf-8")
    required = [
        (m03, "| 정규식↔사람 (strict) | 전체 | 0.998 | 0.995 | 0.998 | 0.997 | 283 |"),
        (m03, "| 자기일치 pass1↔pass2 | `refusal` | 0.984 | 0.956 | 0.982 | 0.979 | 292 |"),
        (m03, "| 전체 | 965 | 2 | **0** | 270 |"),
        (m04, "| ALL | corrected | 200 | 0.980 | 0.975 | 1 | 0 | 0.995 | 0.886 | 0.995 | 0.990 |"),
        (m04, "| ALL | stored (uncorrected) | 200 | 0.980 | 0.780 | 40 | 0 | 0.800 | 0.135 | 0.746 | 0.600 |"),
    ]
    if any(fragment not in text for text, fragment in required):
        raise ValueError("M03/M04 report format or canonical values changed")
    lines.extend([
        "## 8. M03/M04 human validation", "",
        "One author labelled throughout; test--retest is not inter-rater reliability.", "",
        "| validation | scope | n | corrected raw agreement | kappa | FN | FP |",
        "|---|---|---:|---:|---:|---:|---:|",
        "| M03 strict | arm1/arm3, seed42, none/agreeable | 283 | 0.998 | 0.995 | 0 | 2 |",
        "| M03 pass1--pass2 test--retest | refusal label | 292 | 0.984 | 0.956 | — | — |",
        "| M04 strict | arm0/arm2, seed42, none/agreeable | 200 | 0.995 | 0.886 | 1 | 0 |", "",
        "M04 stored uncorrected regex: agreement 0.800, kappa 0.135, FN 40, FP 0.",
        "Across M03+M04: 483 human-labelled records, corrected regex FP 2 and FN 1.", "",
        "Sources: `branch_1_premise/m03_human_validation.md` and",
        "`paper/flmsec_reframe_work/m04_scoring_report.md`.", "",
    ])


def section_r14(lines):
    runbook = (EXPERIMENT / "R14_RUNBOOK.md").read_text(encoding="utf-8")
    patterns = {
        "attempt1": r"gap ([0-9.]+) > 0\.01",
        "final": r"PASS\(gap=([0-9.]+)\).*STEP 1",
        "arm4": r"arm4 broadening gate: votes=3/4 median_shift=\+([0-9.]+)",
        "arm5": r"arm5 narrowing gate: votes=2/4 median_shift=\+([0-9.]+)",
    }
    found = {name: float(re.search(pattern, runbook).group(1))
             for name, pattern in patterns.items()}
    lines.extend([
        "## 9. R14 matched-data gate — no model result", "",
        "| attempt | outcome |", "|---|---|",
        f"| 1 | token-budget gap {found['attempt1']:.6f} > 0.01; FAIL |",
        "| 2 | token matching PASS; legacy fixed-sample-size checker bug exposed |",
        f"| 3 | token-budget gap {found['final']:.6f}; arm4 broadening 3/4 votes, median shift +{found['arm4']:.3f}; arm5 narrowing 2/4 votes, median shift +{found['arm5']:.3f}; manipulation gate FAIL |",
        "", "No `r14-matched-data` artifact, training run, or evaluation result exists. The fixed",
        "preregistration prohibited post-failure prompt or threshold tuning. Source:",
        "`branch_1_premise/experiment/R14_RUNBOOK.md` §7.", "",
    ])


def render():
    summaries = load_summaries()
    lines = [
        "# NUMBERS — 논문이 인용하는 수치의 정본",
        "",
        "> **생성물이다. 손으로 고치지 않는다.**",
        "> `python src/build_numbers.py`가 `results/dual_primary_summary*.json`에서 만든다.",
        "",
        "원고(`paper/flmsec/main.tex` · `paper/iclr/`)·투고처 문서·그림 스크립트는",
        "**여기서만 인용한다.** 값에는 반드시 코호트·시드 수·절단 모드가 따라붙는다 —",
        "넷 중 하나라도 빠지면 같은 이름의 다른 값을 섞게 된다.",
        "",
        "**쓰지 말 것**: `branch_1_premise/results_summary.md` ·",
        "`paper/full/notes/EDIT_MAP.md` §6의 R6 수치(1런짜리 낡은 값 — persona +0.3690은 폐기,",
        "정본은 +0.3269다).",
        "",
        "⚠️ 접미사는 중첩된다(`_e1_phi4_env76`도 `_env76`으로 끝난다). 개수는 glob이",
        "아니라 `cohort.files`에서 유도했다.",
        "",
        "---",
        "",
    ]
    section_cohorts(lines, summaries)
    section_grid(lines, summaries)
    section_factorial(lines, summaries)
    section_persona_contrast(lines, summaries)
    section_cell_separation(lines, summaries)
    section_e1(lines)
    section_r13_ifeval(lines)
    section_human_validation(lines)
    section_r14(lines)
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="갱신이 필요한지만 본다. 필요하면 종료코드 1",
    )
    args = parser.parse_args()

    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else None
        if current == text:
            print(f"[numbers] 최신이다: {OUTPUT}")
            return 0
        print(f"[numbers] 갱신이 필요하다: {OUTPUT}")
        return 1
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"[numbers] -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
