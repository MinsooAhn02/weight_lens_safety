"""R11 판정 — 같은 체크포인트에서 256 대 512를 읽는다.

`R11_GATE2_SAME_CKPT.md` §2의 순서를 그대로 밟는다:
  §2.1 성립 조건 — 같은 arm의 두 상한이 같은 `ckpt_fingerprint.sha256`인가
  §2.2 1차 판정 — 실제 상한 도달(`hit_cap`)이 무너지는가
  §2.3 2차 판정 — 문중 EOS(휴리스틱 절단 중 상한 미도달)의 arm 차등이 남는가

⚠️ **이 코호트는 다른 코호트와 결합하지 않는다**(§1.3). 여기서 읽는 것은 자기 안의
256 대 512뿐이고, 256 헤드라인 격자(`_env76`)를 대체하지 않는다.

비교용으로 `_env76`(256) · `_env76_max512`(512)를 같은 정의로 같이 찍는다 — 그 둘은
**서로 다른 체크포인트**이므로(=이 실험의 이유) 대조가 아니라 참고다.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_dual_primary import corrected_refused, looks_truncated  # noqa: E402
from eval_refusal import wilson_ci  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
ARMS = ("arm1", "arm3")
LENSES = ("none", "agreeable")


def load(arm, suffix, round_=3):
    path = RESULTS / f"eval_{arm}_r{round_}_s42{suffix}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def decompose(payload, lens):
    """intake_results.describe_truncation과 **같은 정의**를 쓴다. 새로 만들지 않는다."""
    records = payload["conditions"][lens]["records"]
    flagged = [looks_truncated(r.get("response") or "") for r in records]
    heuristic = [r for r, f in zip(records, flagged) if f]
    real_cap = [r for r in heuristic if r.get("hit_cap")]
    kept = [r for r, f in zip(records, flagged) if not f]
    n = len(records)
    # ★ 두 가지를 구분해서 낸다. `real_cap`은 **휴리스틱이 표시한 것 중** 상한에 닿은 것이고
    # (절단 분해용), `n_hit_cap`은 표시 여부와 무관한 **원 상한 도달 수**다. 원고가
    # `_env76_max512`의 `coherence_audit.hit_cap.rate`와 나란히 놓는 값은 후자다 —
    # 둘을 섞으면 arm1이 5.1%와 4.8% 사이에서 흔들린다.
    n_hit_cap = sum(1 for r in records if r.get("hit_cap"))
    return {
        "n": n,
        "heuristic": len(heuristic),
        "real_cap": len(real_cap),
        "n_hit_cap": n_hit_cap,
        "hit_cap_rate": n_hit_cap / n,
        "learned_eos": len(heuristic) - len(real_cap),
        "cap_rate": len(real_cap) / n,
        "cap_ci": wilson_ci(len(real_cap), n),
        "eos_rate": (len(heuristic) - len(real_cap)) / n,
        "eos_ci": wilson_ci(len(heuristic) - len(real_cap), n),
        # Gate 2 체크리스트의 마지막 항목 — 긴 상한에서 방향이 유지되는가, 포함/제외를 같이 낸다
        "refusal_included": sum(corrected_refused(r["response"]) for r in records) / n,
        "refusal_excluded": (
            sum(corrected_refused(r["response"]) for r in kept) / len(kept) if kept else None
        ),
        "n_excluded_denom": len(kept),
    }


def main():
    out = {"cohort": "_r11_same_ckpt_{256,512}", "note": __doc__.strip().splitlines()[0]}

    # §2.1 성립 조건
    identity = {}
    for arm in ARMS:
        lo, hi = load(arm, "_r11_same_ckpt_256"), load(arm, "_r11_same_ckpt_512")
        if lo is None or hi is None:
            identity[arm] = {"status": "missing"}
            continue
        a = lo["ckpt_fingerprint"]["sha256"]
        b = hi["ckpt_fingerprint"]["sha256"]
        identity[arm] = {"sha256_256": a, "sha256_512": b, "same": a == b}
    out["gate_2_1_same_checkpoint"] = identity
    admissible = all(v.get("same") for v in identity.values())
    out["admissible"] = admissible

    print("§2.1 성립 조건 — 같은 체크포인트인가")
    for arm, v in identity.items():
        mark = "✓" if v.get("same") else "✗"
        print(f"  {mark} {arm}: {v.get('sha256_256', '?')[:16]} vs {v.get('sha256_512', '?')[:16]}")
    if not admissible:
        print("\n✗ 성립하지 않는다. 해석하지 않고 그 arm을 다시 돈다(§2.1).")
        return 1

    # §2.2 / §2.3
    rows = {}
    print("\n§2.2·§2.3 — 같은 체크포인트, 상한만 바뀐다")
    header = f"  {'arm/lens':18} {'cap':>16} {'cap 512':>16} {'문중EOS 256':>16} {'문중EOS 512':>16}"
    print(header)
    for arm in ARMS:
        lo, hi = load(arm, "_r11_same_ckpt_256"), load(arm, "_r11_same_ckpt_512")
        for lens in LENSES:
            a, b = decompose(lo, lens), decompose(hi, lens)
            rows[f"{arm}/{lens}"] = {"cap256": a, "cap512": b}
            print(
                f"  {arm + '/' + lens:18} "
                f"{a['cap_rate']:>7.2%} {'':>8} {b['cap_rate']:>7.2%} {'':>8} "
                f"{a['eos_rate']:>7.2%} {'':>8} {b['eos_rate']:>7.2%}"
            )
    out["by_cell"] = rows

    print("\nGate 2 마지막 항목 — 긴 상한에서 방향이 유지되는가 (거부율, 포함/제외)")
    print(f"  {'arm/lens':18} {'incl 256':>10} {'incl 512':>10} {'excl 256':>10} {'excl 512':>10}")
    for key, v in rows.items():
        a, b = v["cap256"], v["cap512"]
        print(
            f"  {key:18} {a['refusal_included']:>10.4f} {b['refusal_included']:>10.4f} "
            f"{a['refusal_excluded']:>10.4f} {b['refusal_excluded']:>10.4f}"
        )

    # 참고 — 서로 다른 체크포인트인 기존 8B 코호트 쌍
    ref = {}
    for arm in ("arm0", "arm1", "arm2", "arm3"):
        lo = load(arm, "_env76", round_=0 if arm == "arm0" else 3)
        hi = load(arm, "_env76_max512", round_=0 if arm == "arm0" else 3)
        if lo and hi:
            for lens in LENSES:
                if lens in lo["conditions"] and lens in hi["conditions"]:
                    ref[f"{arm}/{lens}"] = {
                        "cap256": decompose(lo, lens),
                        "cap512": decompose(hi, lens),
                    }
    out["reference_env76_pair_different_checkpoints"] = ref

    dest = RESULTS / "r11_same_ckpt_judgment.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
