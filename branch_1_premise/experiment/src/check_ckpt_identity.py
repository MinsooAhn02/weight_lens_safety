"""같은 (arm, round, seed)가 두 코호트에서 **같은 가중치였는지**를 저장된 응답으로 되묻는다.

왜 필요한가 — 산출 JSON에 가중치 식별자가 없다. `model_path`는 두 코호트에서 문자열까지
같고(`/kaggle/tmp/ckpt/arm1_r3_s42`), `env` 스탬프도 같다. 그래서 기존 감사
(`audit_manuscript_numbers.py`, `audit_saved_outcomes.ps1`)는 **재학습으로 가중치가 갈린 것을
잡을 수 없다.** 실제로 못 잡았다: `_env76_max512`(512)는 `_env76`(256)과 다른 체크포인트에서
나왔고(`src/make_nb_r5.py:168-169`가 재학습 사실을 적어 뒀다), 그 사실이 Gate 2 정본
(`paper/full/notes/긴_논문_개선해야할점.md:173` "같은 checkpoints를 512 tokens 이상으로 재평가한다")의
첫 조항 위반인데도 게이트가 ✅로 통과했다.

무엇으로 되묻는가 — 생성은 `temperature=0.0` 그리디다(`eval_refusal.py:281`). 그러면 같은
가중치는 같은 프롬프트에 **바이트까지 같은** 응답을 낸다. 그래서 두 지표를 센다:

- `byte_identical` : 낮은 상한 쪽이 **상한에 닿지 않은** 응답만 짝지어 비교한다. 상한에 닿은
  응답은 높은 상한에서 더 이어지는 게 정상이므로 제외해야 가중치 신호만 남는다
- `too_short`     : 40자 미만 응답 수. **상한을 올려서 바뀔 수 있는 값이 아니다** — 이쪽이
  움직이면 상한이 아니라 모델이 달라진 것이다

⇒ **어댑터가 없는 arm0가 대조군이다.** arm0는 학습이 없어 두 코호트가 같은 base 가중치이므로
100%가 나와야 하고, 실제로 나온다. arm0가 100%인데 학습 arm이 아니면, 차이는 상한이 아니라
가중치다. 이 스크립트는 그 대조를 재현 가능하게 만든다.

사용:
    python src/check_ckpt_identity.py                      # 기본 쌍 전부, JSON 기록
    python src/check_ckpt_identity.py --no_write           # 화면에만
    python src/check_ckpt_identity.py --pair _env76 _env76_max512

종료코드 0이면 모든 쌍을 잴 수 있었고, 1이면 하나 이상이 `not_available`이다.
"""
import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"
OUT_PATH = RESULTS / "ckpt_identity_audit.json"

# 기본 쌍. (낮은 상한 접미사, 높은 상한 접미사) — 순서가 의미를 갖는다: `hit_cap` 필터는
# **낮은 상한 쪽**에 건다. 세 쌍 모두 같은 시드·같은 학습 데이터인데 세션이 달라 재학습됐다.
DEFAULT_PAIRS = [
    ("_env76", "_env76_max512"),
    ("_r10_1b_lora_env76", "_r10_1b_lora_max512"),
    ("_r10_1b_full_env76", "_r10_1b_full_max512"),
]

# 판정에 쓰는 두 렌즈. 원고 §"What the cap was hiding"이 인용하는 것도 이 둘이다.
LENSES = ("none", "agreeable")

TOO_SHORT_CHARS = 40  # `eval_cohorts` 규칙과 같은 경계. 여기서 다시 재지 않고 저장값을 읽는다.


def round_tag(value):
    """`eval_refusal.round_tag`와 같은 규약 — 3.0 -> '3', 0.5 -> '0p5'."""
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def load(suffix, arm, rnd, seed):
    path = RESULTS / f"eval_{arm}_r{round_tag(rnd)}_s{seed}{suffix}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_lens(lo_recs, hi_recs):
    """한 렌즈의 두 응답 목록을 견준다. 저장 필드가 없으면 None을 돌려준다."""
    if len(lo_recs) != len(hi_recs):
        return None
    if any("hit_cap" not in r or "too_short" not in r for r in lo_recs):
        return None  # legacy 산출물에는 hit_cap/too_short가 없다

    pool = [(a, b) for a, b in zip(lo_recs, hi_recs) if not a["hit_cap"]]
    same = sum(1 for a, b in pool if a["response"] == b["response"])
    return {
        "n": len(lo_recs),
        "n_compared": len(pool),
        "n_byte_identical": same,
        "byte_identical_rate": (same / len(pool)) if pool else None,
        "too_short_low_cap": sum(1 for r in lo_recs if r["too_short"]),
        "too_short_high_cap": sum(1 for r in hi_recs if r["too_short"]),
    }


def compare_pair(lo_suffix, hi_suffix):
    """두 코호트에 **둘 다 있는** 모든 (arm, round, seed) 셀을 견준다."""
    cells = []
    for hi_path in sorted(RESULTS.glob(f"eval_*{hi_suffix}.json")):
        hi = json.loads(hi_path.read_text(encoding="utf-8"))
        lo = load(lo_suffix, hi["arm"], hi["round"], hi["seed"])
        if lo is None:
            continue
        if lo["max_new_tokens"] >= hi["max_new_tokens"]:
            continue  # 상한이 안 올라간 쌍은 이 검사의 대상이 아니다

        lenses = {}
        for lens in LENSES:
            if lens not in lo["conditions"] or lens not in hi["conditions"]:
                continue
            result = compare_lens(
                lo["conditions"][lens]["records"], hi["conditions"][lens]["records"]
            )
            if result is not None:
                lenses[lens] = result

        cells.append({
            "arm": hi["arm"],
            "round": hi["round"],
            "seed": hi["seed"],
            # ★어댑터가 없으면 학습이 없다 = 두 코호트가 같은 base 가중치다. 대조군의 정의.
            "retrained": bool(hi.get("adapter_path")) or hi["arm"] != "arm0",
            "max_new_tokens": [lo["max_new_tokens"], hi["max_new_tokens"]],
            "low_cap_file": f"eval_{hi['arm']}_r{round_tag(hi['round'])}_s{hi['seed']}{lo_suffix}.json",
            "high_cap_file": hi_path.name,
            "status": "available" if lenses else "not_available",
            "lenses": lenses,
        })
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", metavar=("LOW", "HIGH"),
                    help="접미사 쌍. 생략하면 DEFAULT_PAIRS 전부")
    ap.add_argument("--no_write", action="store_true")
    args = ap.parse_args()

    pairs = [tuple(p) for p in args.pair] if args.pair else DEFAULT_PAIRS
    payload = {
        "what": "같은 (arm, round, seed)가 두 코호트에서 같은 가중치였는지 — 저장된 응답으로 되묻는다",
        "why_this_works": (
            "생성이 temperature=0.0 그리디이므로 같은 가중치는 바이트까지 같은 응답을 낸다. "
            "낮은 상한에서 상한에 닿은 응답은 제외한다 — 그것만 높은 상한에서 더 이어지는 게 정상이다."
        ),
        "control": "arm0는 어댑터가 없어 재학습이 없다. 100%가 아니면 이 검사 자체가 틀린 것이다.",
        "ceiling": "가중치를 직접 해싱하지 않는다. 신규 실행의 가중치 신원은 eval JSON의 ckpt_fingerprint가 맡는다.",
        "too_short_chars": TOO_SHORT_CHARS,
        "lenses": list(LENSES),
        "pairs": [],
    }

    incomplete = False
    for lo_suffix, hi_suffix in pairs:
        cells = compare_pair(lo_suffix, hi_suffix)
        payload["pairs"].append({"low_cap": lo_suffix, "high_cap": hi_suffix, "cells": cells})

        print(f"\n=== {lo_suffix or '(legacy)'}  ↔  {hi_suffix}")
        if not cells:
            print("  겹치는 셀이 없다")
            continue
        print(f"  {'arm':6} {'seed':>5} {'재학습':>6}  "
              f"{'lens':11} {'too_short 저→고':>16} {'바이트 동일':>18}")
        for c in cells:
            if c["status"] != "available":
                incomplete = True
                print(f"  {c['arm']:6} {c['seed']:>5}  not_available (저장 필드 없음)")
                continue
            for lens, v in c["lenses"].items():
                rate = "n/a" if v["byte_identical_rate"] is None else f"{v['byte_identical_rate']:.1%}"
                print(f"  {c['arm']:6} {c['seed']:>5} {'예' if c['retrained'] else '아니오':>6}  "
                      f"{lens:11} {v['too_short_low_cap']:>6} → {v['too_short_high_cap']:<7} "
                      f"{v['n_byte_identical']:>5}/{v['n_compared']:<5} {rate:>6}")

    if not args.no_write:
        RESULTS.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[ckpt-identity] -> {OUT_PATH}")

    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main())
