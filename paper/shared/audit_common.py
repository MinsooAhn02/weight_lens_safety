"""원고 수치 감사의 공용 부품. 원고마다 **값 목록만** 다르고 기계는 같다.

왜 공용인가: 장문 원고(`paper/full/main.tex`)가 2026-08-07에 동결된 뒤 조용히 낡은 이유는
그것을 지키는 검사가 하나도 없었기 때문이다. FLMSec판에는 `flmsec/audit_manuscript_numbers.py`
가 있었고 장문판에는 없었다. 검사를 하나 더 **복사**하면 두 벌이 어긋난다 — 이 리포가
반복해 온 사고다(금지문구 목록이 지금 두 벌이고 서로 다르다). 그래서 기계는 여기 한 벌만
두고, 원고별 스크립트는 **인쇄된 값의 목록**만 갖는다.

쓰는 쪽:
    from audit_common import Audit
    a = Audit(tex_path)
    a.check("grid arm1/agreeable", 0.6656, a.cell(inc, "arm1", "agreeable", "corrected_refusal_rate"), 4)
    a.require(strong > para1, "강도가 패러프레이즈를 넘지 않는다")
    raise SystemExit(a.report(notes))
"""

import json
import re
from pathlib import Path

SHARED = Path(__file__).resolve().parent
ROOT = SHARED.parents[1]
RESULTS = ROOT / "branch_1_premise" / "experiment" / "results"


def require_text_layer():
    """Both FLMSec audits recount from response text. The public release strips that text
    (source-dataset licence), so stop with a clear message instead of a deep KeyError."""
    probe = RESULTS / "eval_arm1_r3_s42_env76.json"
    zip_ = ROOT / "branch_1_premise" / "data" / "gen_data_backup.zip"
    conds = json.loads(probe.read_text(encoding="utf-8"))["conditions"]
    rec = next(iter(conds.values()))["records"][0]
    if "response" not in rec or not zip_.exists():
        raise SystemExit("audit needs the prompt/response text layer, which this copy does not "
                         "contain (see README: obtain StrongREJECT and re-run src/eval_refusal.py; "
                         "training data: branch_1_premise/data/gen_data_backup.zip).")


def load(suffix):
    """코호트 요약 JSON. 접미사는 **완전일치**로 쓴다 — `_e1_phi4_env76`도 `_env76`으로 끝난다."""
    with (RESULTS / f"dual_primary_summary{suffix}.json").open(encoding="utf-8") as f:
        return json.load(f)


def load_result(name):
    """요약이 아닌 판정·감사 산출물(예: `r11_same_ckpt_judgment.json`)."""
    with (RESULTS / name).open(encoding="utf-8") as f:
        return json.load(f)


def _dual_primary():
    """정본 계측 함수를 분석 모듈에서 직접 가져온다."""
    import sys

    src = ROOT / "branch_1_premise" / "experiment" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import analyze_dual_primary

    return analyze_dual_primary


def seed42_truncation_grid():
    """`_env76` seed-42 4×4 격자의 셀별 절단 휴리스틱 비율."""
    looks_truncated = _dual_primary().looks_truncated
    files = {
        "arm0": "eval_arm0_r0_s42_env76.json",
        "arm1": "eval_arm1_r3_s42_env76.json",
        "arm2": "eval_arm2_r3_s42_env76.json",
        "arm3": "eval_arm3_r3_s42_env76.json",
    }
    out = {}
    for arm, filename in files.items():
        payload = load_result(filename)
        for lens in ("none", "agreeable", "principled", "agreeable_para1"):
            records = payload["conditions"][lens]["records"]
            flagged = sum(looks_truncated(r["response"]) for r in records)
            out.setdefault(lens, {})[arm] = {
                "flagged": flagged,
                "n": len(records),
                "rate": flagged / len(records),
            }
    return out


def paired_refusal_stats(eval_filename, left_lens, right_lens):
    """한 eval JSON 안의 prompt-paired 정정 거부 전이와 양쪽 비율."""
    corrected_refused = _dual_primary().corrected_refused
    payload = load_result(eval_filename)

    def labels(lens):
        records = payload["conditions"][lens]["records"]
        keyed = {r["prompt"]: corrected_refused(r["response"]) for r in records}
        if len(keyed) != len(records):
            raise ValueError(f"duplicate prompts in {eval_filename}/{lens}")
        return keyed

    left, right = labels(left_lens), labels(right_lens)
    if left.keys() != right.keys():
        raise ValueError(f"unpaired prompts in {eval_filename}: {left_lens}/{right_lens}")
    return {
        "n": len(left),
        "left_rate": sum(left.values()) / len(left),
        "right_rate": sum(right.values()) / len(right),
        "left_to_right_false": sum(left[p] and not right[p] for p in left),
        "left_false_to_right": sum(not left[p] and right[p] for p in left),
    }


def untrained_lens_gap(eval_filename):
    """E1 선별용 — 학습 없는 base의 `|refusal(none) − refusal(agreeable)|`.

    E1 후보들은 코호트 요약(`dual_primary_summary*`)이 없고 eval JSON만 있다. 그래서
    격자와 **같은 정의**(`analyze_dual_primary.corrected_refused`)로 여기서 재계산한다 —
    저장된 `refusal_rate`를 쓰면 아포스트로피 정정 이전 값이 섞인다.
    """
    corrected_refused = _dual_primary().corrected_refused

    payload = load_result(eval_filename)
    rates = {}
    for lens in ("none", "agreeable"):
        records = payload["conditions"][lens]["records"]
        rates[lens] = sum(corrected_refused(r["response"]) for r in records) / len(records)
    return abs(rates["none"] - rates["agreeable"])


def cell(analysis, arm, lens, field):
    for row in analysis["grid"]:
        if row["arm"] == arm and row["lens"] == lens:
            return row[field]
    raise KeyError((arm, lens, field))


def interaction(analysis, arm, lens, outcome, sub=None):
    for row in analysis["interactions"]:
        if row["arm"] == arm and row["lens"] == lens:
            o = row["outcomes"][outcome]
            return o[sub]["I_a"] if sub else o["I_a"]
    raise KeyError((arm, lens))


def per_seed(analysis, arm, lens):
    for row in analysis["interactions"]:
        if row["arm"] == arm and row["lens"] == lens:
            return row["per_seed_paired_ci"]
    raise KeyError((arm, lens))


def factorial(payload, effect, outcome="corrected_refusal_rate", mode="truncation_included"):
    """R6 2×2의 주효과·상호작용. `effect`는 `persona_main_effect` 등이다.

    ⚠️ 여기의 `persona_main_effect`는 ((arm1+arm4)/2) − ((arm2+arm5)/2)이고,
    `persona_contrast()`의 값은 arm1 − arm2다. **정의가 다르므로 같은 표에 놓지 않는다**
    (요약 JSON이 `not_the_factorial_main_effect`로 직접 경고한다).
    """
    return payload["analyses"][mode]["factorial_2x2"]["contrasts"][outcome][effect]


def persona_contrast(payload, outcome="corrected_refusal_rate", mode="truncation_included"):
    """2셀 arm1 − arm2 대비. R10 update-method 축이 코호트마다 비교하는 값이다."""
    return payload["analyses"][mode]["persona_contrast"]["contrasts"][outcome]["persona_effect"]


class Audit:
    """원고 하나에 대한 대조 결과를 모은다."""

    # 본문에서 뽑을 수치. 표·캡션·수식 안의 것까지 잡되 각주 번호는 피한다.
    NUMERAL = re.compile(r"-?\d[\d{},]*\.\d+|-?\d[\d{},]*")

    def __init__(self, tex_path):
        self.tex_path = Path(tex_path)
        self.failures = []
        self.checked = 0
        self.matched = set()

    def _numerals(self):
        """원고 본문이 인쇄하는 모든 수치 문자열."""
        if not hasattr(self, "_numeral_cache"):
            self._numeral_cache = set(self.NUMERAL.findall(self._body()))
        return self._numeral_cache

    @staticmethod
    def _renderings(printed, places):
        """이 값이 LaTeX 본문에 나타날 수 있는 표기들.

        `0.485` · `$0.485$` · 정수 천단위 `15{,}024` / `15,024` 를 모두 덮는다.
        부호는 그대로 둔다 — `-0.012`와 `0.012`는 다른 주장이다.
        """
        value = float(printed)
        out = {f"{printed}", f"{value:.{places}f}"}
        if value == int(value):
            n = int(value)
            out |= {str(n), f"{n:,}", f"{n:,}".replace(",", "{,}")}
        # 비율은 원고에서 백분율로 인쇄되는 일이 많다: 0.051 → `$5.1\%$`, 0.20 → `$20$--`
        if -1.0 <= value <= 1.0:
            pct = value * 100
            out |= {f"{pct:.1f}", f"{pct:g}", f"{pct:.2f}"}
        return {r for r in out if r}

    def check(self, label, printed, actual, places=3, in_tex=True):
        """원고에 인쇄된 값이 결과 파일의 값과 같은가 — **그리고 원고에 실제로 있는가.**

        두 번째 절반이 2026-08-10까지 없었다. `check`가 파이썬 리터럴과 JSON만
        비교했으므로, 원고가 legacy 값을 인쇄하는 동안에도 감사는 통과했다
        (장문판 절단 감사가 그렇게 사흘을 버텼다). 이제 값이 본문에 없으면 실패한다.

        `in_tex=False`는 **원고가 직접 인쇄하지 않는 중간값**에만 쓴다 — 이유를 적을 것.
        """
        self.checked += 1
        if round(float(actual), places) != round(float(printed), places):
            self.failures.append(f"{label}: 원고={printed} 결과={round(float(actual), 6)}")
            return
        if in_tex:
            hits = self._renderings(printed, places) & self._numerals()
            if not hits:
                self.failures.append(
                    f"{label}: 값 {printed}이(가) 결과와는 맞지만 **원고에 없다** — "
                    "원고가 다른 값을 인쇄하고 있거나 문장이 사라졌다")
            else:
                self.matched |= hits

    def require(self, condition, message):
        """수치가 아니라 **문장이 성립하는가**. 값이 다 맞아도 서열이 뒤집히면 죽는다."""
        self.checked += 1
        if not condition:
            self.failures.append(message)

    def forbid(self, pattern, why):
        """렌더되는 본문에 있으면 안 되는 표현. 주장 상한을 기계가 지킨다.

        ⚠️ **주석은 보지 않는다.** 헤더 주석은 철회된 표현을 *인용*해 두는 자리이고
        (예: 왜 "nineteen times smaller"를 걷었는지), 그것까지 막으면 기록을 못 남긴다.
        """
        self.checked += 1
        if re.search(pattern, self._body(), re.I | re.S):
            self.failures.append(f"금지 표현이 본문에 있다 ({why}): /{pattern}/")

    def _tex(self):
        if not hasattr(self, "_tex_cache"):
            self._tex_cache = self.tex_path.read_text(encoding="utf-8")
        return self._tex_cache

    def _body(self):
        """LaTeX 주석을 지운 본문. 이스케이프된 `\\%`는 주석이 아니므로 남긴다."""
        if not hasattr(self, "_body_cache"):
            self._body_cache = re.sub(r"(?<!\\)%.*", "", self._tex())
        return self._body_cache

    def uncovered(self, min_decimals=3):
        """원고가 인쇄하지만 **어떤 검사도 걸지 않은** 수치.

        소수 셋 자리 이상만 센다 — 쪽 번호·연도·`n=313` 같은 정수까지 세면 잡음이
        결과를 덮는다. 여기 이름이 오르는 값은 사람이 아티팩트까지 따라가야 한다.
        """
        out = []
        for numeral in sorted(self._numerals()):
            if numeral in self.matched:
                continue
            if "." not in numeral:
                continue
            if len(numeral.split(".")[1]) < min_decimals:
                continue
            out.append(numeral)
        return out

    def report(self, notes=()):
        """0이면 불일치 없음. `notes`는 기계가 못 거는 provenance 경고다."""
        print(f"{self.tex_path.relative_to(ROOT)} — {self.checked}개 검사")
        if self.failures:
            print(f"\n불일치 {len(self.failures)}건:")
            for f in self.failures:
                print("  " + f)
        else:
            print("불일치 없음")
        gaps = self.uncovered()
        if gaps:
            print(f"\n검사 없는 인쇄값 {len(gaps)}개 — 사람이 아티팩트까지 따라갈 것:")
            print("  " + "  ".join(gaps))
        if notes:
            print("\n기계 대조 밖 — 바뀌면 손으로 확인할 것:")
            for label, pattern, why in notes:
                state = "원고에 있음" if re.search(pattern, self._tex(), re.S) else "원고에 없음"
                print(f"  - {label} ({state}): {why}")
        return 1 if self.failures else 0
