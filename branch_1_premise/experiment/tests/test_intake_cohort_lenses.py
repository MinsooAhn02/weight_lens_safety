"""인수 검사의 렌즈 격자 — 사전등록된 코호트만 정본 4렌즈에서 벗어날 수 있다.

R11은 한 세션에 256·512를 같은 체크포인트로 내야 해서 2렌즈이고(`R11_GATE2_SAME_CKPT.md`
§1.3), R12는 렌즈 축 자체가 측정 대상이라 7렌즈다(`R12_LENS_AXIS.md` §2), R13은 같은
7렌즈 유해 격자에 5렌즈 양성 IFEval을 쌍으로 묶는다(`R13_BENIGN.md` §1.2). 셋 다
설계 기록에 고정돼 있다. R13 문서는 사후 정리됐지만 실행 생성기와 테스트는 결과 전에
커밋됐다. 이 시험이 지키는 것은 그 예외가 **접미사 완전일치**로만 열린다는 것이다 —
부분일치면 `_e1_phi4_env76`가 `_env76`에 걸리는 종류의 사고가 난다.
"""
import copy
import json
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import intake_results  # noqa: E402
import make_nb_r13_benign  # noqa: E402


def lens_findings(suffix, lenses):
    """프로덕션이 쓰는 바로 그 조회를 통과하는가."""
    return lenses == intake_results.expected_lenses_for(suffix)


class CohortLensTest(unittest.TestCase):
    def test_default_cohort_still_requires_the_canonical_four(self):
        self.assertTrue(lens_findings("_env76", intake_results.EXPECTED_LENSES))
        self.assertFalse(lens_findings("_env76", ["none", "agreeable"]))

    def test_r11_accepts_its_two_lenses_and_nothing_else(self):
        for suffix in ("_r11_same_ckpt_256", "_r11_same_ckpt_512"):
            self.assertTrue(lens_findings(suffix, ["none", "agreeable"]))
            self.assertFalse(lens_findings(suffix, intake_results.EXPECTED_LENSES))

    def test_order_matters(self):
        self.assertFalse(lens_findings("_r11_same_ckpt_256", ["agreeable", "none"]))

    def test_r12_requires_all_seven_in_one_file(self):
        seven = intake_results.COHORT_LENSES["_r12_lens_axis_env76"]
        self.assertEqual(len(seven), 7)
        self.assertTrue(lens_findings("_r12_lens_axis_env76", seven))
        self.assertFalse(lens_findings("_r12_lens_axis_env76", seven[:6]))

    def test_r13_matches_the_frozen_generator_grid(self):
        suffix = "_r13_benign_env76"
        seven = intake_results.expected_lenses_for(suffix)
        self.assertEqual(seven, make_nb_r13_benign.HARMFUL_LENSES)
        self.assertFalse(lens_findings(suffix, seven[:6]))
        self.assertFalse(lens_findings(f"_derived{suffix}", seven))

    def test_r14_requires_the_preregistered_seven_lenses(self):
        suffix = "_r14_matched_factorial_env76"
        seven = intake_results.expected_lenses_for(suffix)
        self.assertEqual(len(seven), 7)
        self.assertTrue(lens_findings(suffix, seven))
        self.assertFalse(lens_findings(suffix, seven[:6]))
        self.assertFalse(lens_findings(f"_derived{suffix}", seven))

    def test_exception_is_exact_match_not_suffix_match(self):
        """`_e1_phi4_r11_same_ckpt_256` 같은 파생 접미사는 예외를 물려받지 않는다."""
        derived = "_e1_phi4_r11_same_ckpt_256"
        self.assertNotIn(derived, intake_results.COHORT_LENSES)
        self.assertFalse(lens_findings(derived, ["none", "agreeable"]))

    def test_every_exception_is_documented_in_a_design_record(self):
        design_docs = list((SRC.parent).glob("R14_*.md"))
        design_docs += list((SRC.parent / "archive" / "completed").glob("R1[1-3]_*.md"))
        docs = "\n".join(p.read_text(encoding="utf-8") for p in design_docs)
        for suffix in intake_results.COHORT_LENSES:
            self.assertIn(suffix, docs, f"{suffix}가 어느 설계 기록에도 없다")

    def test_payload_rejects_missing_or_short_condition_cells(self):
        path = SRC.parent / "results" / "eval_arm1_r3_s42_r13_benign_env76.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases = {}
        cases["missing"] = copy.deepcopy(payload)
        cases["missing"]["conditions"].pop("agreeable_strong")
        cases["short"] = copy.deepcopy(payload)
        cases["short"]["conditions"]["agreeable_strong"]["records"].pop()

        for case, altered in cases.items():
            with self.subTest(case=case):
                failures = [
                    f.message
                    for f in intake_results.check_payload(path.name, altered, 256)
                    if f.fatal
                ]
                self.assertTrue(any("conditions" in message for message in failures), failures)


if __name__ == "__main__":
    unittest.main()
