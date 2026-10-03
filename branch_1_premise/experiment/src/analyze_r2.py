"""
가지1 실험 R2 — 집계·분해. PLAN_R2.md §2b/§2c.

로컬에서 돈다(GPU 불필요). results/eval_*_s*.json(schema="r2")을 읽어서:
  1. arm × round × lens 표 — 거부율(Wilson) · harm(부트스트랩) · 조건부 harm **점추정**
  2. 승낙 분해 {일관 승낙 / 비일관} — degeneration 질문(readiness_review §8.4)의 답
  3. 시드 간 range — ⚠️표본 3이면 CI가 아니라 range로 보고하는 편이 정직하다
  4. cross-lens grid — 학습 페르소나 × 평가 페르소나

1차 산출물(schema 없음)은 per-prompt 레코드가 없으므로 건너뛴다. 그게 이번 재실행의 이유다.

사용법:
    python src/analyze_r2.py                    # results/ 전체
    python src/analyze_r2.py --out r2_summary.md
"""
import argparse
from collections import defaultdict
from pathlib import Path

from common import RESULTS_DIR
from eval_cohorts import (
    cohort_output_path,
    format_cohort_listing,
    markdown_cohort_lines,
    select_eval_cohort,
)


def load_r2(results_dir: Path, suffix="", with_cohort=False):
    entries, cohort = select_eval_cohort(results_dir, suffix)
    rows = []
    for entry in entries:
        p = entry["path"]
        d = entry["payload"]
        if d.get("schema") != "r2":
            print(f"[analyze] 건너뜀(1차 스키마, per-prompt 없음): {p.name}")
            continue
        for lens, c in d["conditions"].items():
            rows.append({
                "arm": d["arm"], "round": d["round"], "seed": d["seed"],
                "lens": lens, "n": c["n"],
                "refusal": c["refusal_rate"], "wilson": c["refusal_ci95_wilson"],
                "harm": c["harm_score"], "harm_ci": c.get("harm_ci95_bootstrap"),
                "n_comp": c["n_complied"],
                "n_comp_deg": c["n_complied_degenerate"],
                "n_comp_coh": c["n_complied_coherent"],
                "harm_cond": c.get("harm_given_compliance_coherent"),
                "harm_cond_ci": c.get("harm_given_compliance_ci95_bootstrap"),
            })
    return (rows, cohort) if with_cohort else rows


def fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def fmt_ci(ci, nd=3):
    return "—" if not ci else f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=str(RESULTS_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--list_cohorts", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if args.list_cohorts:
        print(format_cohort_listing(results_dir))
        return

    rows, cohort = load_r2(results_dir, args.suffix, with_cohort=True)
    if not rows:
        print("[analyze] R2 스키마 결과가 없다. 아직 재실행 전이거나 경로가 다르다.")
        return

    L = []
    L.append("# R2 실험 집계\n")
    L.extend(markdown_cohort_lines(cohort))
    L.append("> `analyze_r2.py` 자동 생성. 원본 = `results/eval_*_s*.json`.\n")

    L.append("\n## 1. arm × round × lens\n")
    L.append("| arm | round | seed | lens | n | 거부율 | Wilson 95% | harm | 조건부 harm(일관승낙) | 95% |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["arm"], x["round"], x["seed"], x["lens"])):
        L.append(f"| {r['arm']} | {r['round']} | {r['seed']} | {r['lens']} | {r['n']} | "
                 f"{fmt(r['refusal'])} | {fmt_ci(r['wilson'])} | {fmt(r['harm'])} | "
                 f"{fmt(r['harm_cond'])} | {fmt_ci(r['harm_cond_ci'])} |")

    L.append("\n## 2. 승낙 분해 — 붕괴는 승낙인가 비일관인가\n")
    L.append("> 거부 판정은 정규식이라 횡설수설도 승낙으로 계수된다. 이 표가 그 몫을 분리한다.")
    L.append("> ⚠️`degenerate` 휴리스틱은 저자 정의이지 검증된 지표가 아니다. 응답 전문이 저장돼 있으므로")
    L.append("> 재판정 가능하다.\n")
    L.append("| arm | round | seed | lens | 승낙 | 일관 | 비일관 | 비일관 비율 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["arm"], x["round"], x["seed"], x["lens"])):
        frac = (r["n_comp_deg"] / r["n_comp"]) if r["n_comp"] else None
        L.append(f"| {r['arm']} | {r['round']} | {r['seed']} | {r['lens']} | {r['n_comp']} | "
                 f"{r['n_comp_coh']} | {r['n_comp_deg']} | {fmt(frac)} |")

    L.append("\n## 3. 시드 간 range (라운드3)\n")
    L.append("> ⚠️시드 3개는 표본 3이다. **CI가 아니라 range로 보고한다.**\n")
    byk = defaultdict(list)
    for r in rows:
        if float(r["round"]) == 3.0:
            byk[(r["arm"], r["lens"])].append(r["refusal"])
    L.append("| arm | lens | n_seeds | 거부율 min | max | 폭 |")
    L.append("|---|---|---|---|---|---|")
    for (arm, lens), vals in sorted(byk.items()):
        L.append(f"| {arm} | {lens} | {len(vals)} | {fmt(min(vals))} | {fmt(max(vals))} | "
                 f"{fmt(max(vals) - min(vals))} |")

    L.append("\n## 4. cross-lens grid (arm별 최종 라운드, 시드별 평균)\n")
    L.append("> 학습 페르소나 × 평가 페르소나. **비대각 셀이 붕괴하지 않으면 맥락 일치 설명이 부활하고")
    L.append("> §4의 방어 문단을 철회해야 한다** — PLAN_R2 §6에 미리 적어둔 불리한 결과 1번.")
    L.append("> arm0는 통제군이라 학습이 없다 — 최종 라운드가 r0이다. 라운드3으로 못박으면")
    L.append("> baseline 행이 영원히 비고, 비교 대상이 사라진다.\n")
    trained = {"arm1": "agreeable", "arm3": "principled", "arm2": "(none)", "arm0": "(none)"}
    last_round = {}
    for r in rows:
        a, rd = r["arm"], float(r["round"])
        last_round[a] = max(last_round.get(a, rd), rd)
    grid = defaultdict(list)
    for r in rows:
        if float(r["round"]) == last_round[r["arm"]]:
            grid[(r["arm"], r["lens"])].append(r["refusal"])
    lens_order = ["none", "agreeable", "principled"]
    L.append("| arm (학습) | " + " | ".join(f"eval={l}" for l in lens_order) + " |")
    L.append("|---|" + "---|" * len(lens_order))
    for arm in ["arm0", "arm1", "arm2", "arm3"]:
        cells = []
        for lens in lens_order:
            v = grid.get((arm, lens))
            mark = " ←학습일치" if trained.get(arm) == lens else ""
            cells.append((fmt(sum(v) / len(v)) + mark) if v else "—")
        L.append(f"| {arm} ({trained.get(arm, '?')}) | " + " | ".join(cells) + " |")

    text = "\n".join(L) + "\n"
    if args.out:
        requested = Path(args.out)
        out_path = cohort_output_path(
            requested.parent, requested.stem, args.suffix, requested.suffix
        )
        out_path.write_text(text, encoding="utf-8")
        print(f"[analyze] -> {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
