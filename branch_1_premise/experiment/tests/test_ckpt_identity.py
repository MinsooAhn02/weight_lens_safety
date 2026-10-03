"""`check_ckpt_identity.compare_lens`의 판정 규칙을 고정한다.

핵심은 **낮은 상한에서 상한에 닿은 응답을 빼는 것**이다. 그걸 빼지 않으면 "높은 상한에서 더
이어졌다"가 "가중치가 다르다"로 오독되고, 이 검사가 재는 것이 사라진다.
"""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import check_ckpt_identity as cci  # noqa: E402


def rec(response, hit_cap=False, too_short=False):
    return {"response": response, "hit_cap": hit_cap, "too_short": too_short}


class CompareLensTest(unittest.TestCase):
    def test_same_weights_read_as_fully_identical(self):
        lo = [rec("a"), rec("bb")]
        hi = [rec("a"), rec("bb")]
        out = cci.compare_lens(lo, hi)
        self.assertEqual(out["n_compared"], 2)
        self.assertEqual(out["byte_identical_rate"], 1.0)

    def test_cap_hit_rows_are_excluded_not_counted_as_different(self):
        """상한에 닿은 응답은 높은 상한에서 더 이어지는 게 정상이다 — 비교에서 빠져야 한다."""
        lo = [rec("truncated here", hit_cap=True), rec("done")]
        hi = [rec("truncated here and then some more"), rec("done")]
        out = cci.compare_lens(lo, hi)
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["n_compared"], 1)          # 상한 도달분이 빠졌다
        self.assertEqual(out["byte_identical_rate"], 1.0)  # 100%이지 50%가 아니다

    def test_differing_weights_show_up_below_the_cap(self):
        lo = [rec("alpha"), rec("beta")]
        hi = [rec("alpha"), rec("gamma")]
        self.assertEqual(cci.compare_lens(lo, hi)["byte_identical_rate"], 0.5)

    def test_too_short_is_read_from_the_stored_field_on_both_sides(self):
        lo = [rec("x", too_short=True), rec("yyy")]
        hi = [rec("x", too_short=True), rec("yyy", too_short=True)]
        out = cci.compare_lens(lo, hi)
        self.assertEqual((out["too_short_low_cap"], out["too_short_high_cap"]), (1, 2))

    def test_legacy_without_hit_cap_degrades_instead_of_raising(self):
        lo = [{"response": "a"}]           # legacy 산출물에는 hit_cap/too_short가 없다
        hi = [rec("a")]
        self.assertIsNone(cci.compare_lens(lo, hi))

    def test_length_mismatch_degrades(self):
        self.assertIsNone(cci.compare_lens([rec("a")], [rec("a"), rec("b")]))

    def test_all_rows_capped_leaves_no_comparison(self):
        lo = [rec("a", hit_cap=True)]
        hi = [rec("a longer continuation")]
        out = cci.compare_lens(lo, hi)
        self.assertEqual(out["n_compared"], 0)
        self.assertIsNone(out["byte_identical_rate"])   # 0으로 나누지 않는다


class RoundTagTest(unittest.TestCase):
    def test_matches_eval_refusal_convention(self):
        self.assertEqual(cci.round_tag(3.0), "3")
        self.assertEqual(cci.round_tag(0.5), "0p5")
        self.assertEqual(cci.round_tag(0.25), "0p25")


class ControlArmTest(unittest.TestCase):
    def test_arm0_is_fully_identical_in_the_live_cohorts(self):
        """arm0는 어댑터가 없어 재학습이 없다. 100%가 아니면 이 검사의 전제가 무너진다."""
        cells = cci.compare_pair("_env76", "_env76_max512")
        arm0 = [c for c in cells if c["arm"] == "arm0"]
        self.assertTrue(arm0, "arm0 셀을 찾지 못했다 — 코호트가 바뀌었는지 확인할 것")
        for cell in arm0:
            self.assertFalse(cell["retrained"])
            for lens, v in cell["lenses"].items():
                self.assertEqual(v["byte_identical_rate"], 1.0, f"arm0/{lens}")


if __name__ == "__main__":
    unittest.main()
