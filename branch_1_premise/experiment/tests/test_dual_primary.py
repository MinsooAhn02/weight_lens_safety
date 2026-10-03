import sys
import unittest
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT / "src"
RESULTS = EXPERIMENT / "results"
sys.path.insert(0, str(SRC))

from analyze_dual_primary import (  # noqa: E402
    build_analysis,
    build_factorial,
    factorial_inference,
    render_factorial,
)

# 절단으로 읽히지 않도록 둘 다 종결부호로 끝낸다(`looks_truncated` 참고).
REFUSAL = "I cannot help with that."
COMPLY = "Sure, here is the answer."
SINGLE_RUN_INFERENCE = (
    "셀당 런이 하나다. 기술통계이며 검정하지 않는다 — "
    "이 설계로는 시드 분산을 추정할 수 없다."
)
N_SYNTHETIC_PROMPTS = 4


def synthetic_run(arm, seed, n_lens_refusals, lens_harm):
    """네 프롬프트짜리 합성 런. none 조건은 전부 거부·harm 0.2로 고정한다.

    실제 results/ 파일을 위조하지 않고 셀당 시드가 여럿인 경로를 밟기 위한 것이다.
    """
    def records(n_refusals, harm):
        return [
            {
                "prompt": f"p{index}",
                "response": REFUSAL if index < n_refusals else COMPLY,
                "harm_score": harm,
            }
            for index in range(N_SYNTHETIC_PROMPTS)
        ]

    return {
        "arm": arm,
        "round": 3.0,
        "seed": seed,
        "_source_file": f"eval_{arm}_r3_s{seed}_synthetic.json",
        "_cohort_suffix": "_synthetic",
        "conditions": {
            "none": {"records": records(N_SYNTHETIC_PROMPTS, 0.2)},
            "agreeable": {"records": records(n_lens_refusals, lens_harm)},
        },
    }


# 시드마다 낙폭이 다른 arm1을 하나 두어 range가 한 점으로 무너지지 않게 한다.
SYNTHETIC_CELL_SPEC = {
    "arm1": {42: (1, 0.9), 1337: (0, 0.8), 2718: (2, 1.0)},
    "arm2": {42: (4, 0.2), 1337: (4, 0.2), 2718: (4, 0.2)},
    "arm4": {42: (1, 0.9), 1337: (1, 0.9), 2718: (1, 0.9)},
    "arm5": {42: (3, 0.3), 1337: (3, 0.3), 2718: (3, 0.3)},
}


def synthetic_primary(seeds_by_arm=None):
    seeds_by_arm = seeds_by_arm or {arm: [42, 1337, 2718] for arm in SYNTHETIC_CELL_SPEC}
    return [
        synthetic_run(arm, seed, *SYNTHETIC_CELL_SPEC[arm][seed])
        for arm in ("arm1", "arm2", "arm4", "arm5")
        for seed in sorted(seeds_by_arm[arm])
    ]


class DualPrimaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = build_analysis(RESULTS)

    def test_truncation_port_reproduces_audit_total(self):
        audit = self.payload["truncation_audit"]["heuristic"]
        self.assertTrue(audit["matches_reference"], audit["discrepancy"])
        self.assertEqual(audit["observed"], {"n": 5008, "truncated": 549})

    def test_truncated_and_non_truncated_sr_reference_values(self):
        audit = self.payload["truncation_audit"]
        expected = {
            "truncated": (549, 0.4783388582517975, 0.5336976320582878),
            "non_truncated": (4459, 0.021172227520503174, 0.018838304552590265),
        }
        for group, (n, sr_mean, binary) in expected.items():
            with self.subTest(group=group):
                self.assertEqual(audit[group]["n"], n)
                self.assertAlmostEqual(audit[group]["sr_mean"], sr_mean, places=15)
                self.assertAlmostEqual(
                    audit[group]["sr_binary_rates"]["0.5"], binary, places=15
                )

    def test_corrected_multi_seed_refusal_interactions(self):
        rows = self.payload["analyses"]["truncation_included"]["interactions"]
        observed = {
            (row["arm"], row["lens"]): row["outcomes"]["corrected_refusal"]["I_a"]
            for row in rows
        }
        expected = {
            ("arm1", "agreeable"): 0.4132,
            ("arm2", "agreeable"): 0.0224,
            ("arm3", "agreeable"): 0.4377,
            ("arm1", "agreeable_para1"): 0.4675,
            ("arm3", "agreeable_para1"): 0.4132,
            ("arm2", "agreeable_para1"): 0.0319,
        }
        for key, value in expected.items():
            with self.subTest(arm=key[0], lens=key[1]):
                self.assertAlmostEqual(observed[key], value, delta=1e-3)

    def test_default_cohort_reproduces_pinned_grid(self):
        self.assertEqual(self.payload["cohort"]["out_suffix"], "")
        grid = self.payload["analyses"]["truncation_included"]["grid"]
        self.assertEqual(len(grid), 16)
        observed = {
            (row["arm"], row["lens"]): row["corrected_refusal_rate"]
            for row in grid
        }
        expected = {
            ("arm0", "none"): 0.948881789,
            ("arm2", "none"): 0.968051118,
            ("arm1", "agreeable"): 0.553780618,
            ("arm3", "agreeable"): 0.548455,
        }
        for key, value in expected.items():
            with self.subTest(arm=key[0], lens=key[1]):
                self.assertAlmostEqual(observed[key], value, delta=1e-6)


class FactorialRunCountTest(unittest.TestCase):
    """`inference`가 하드코딩이 아니라 실제 n_runs에서 나오는지 본다.

    R6 배치 3·4가 arm4·arm5에 시드 1337·2718을 채우면 네 칸이 전부 3런이 된다.
    그때 "셀당 런이 하나다"는 산출물에 찍히는 **거짓 문장**이 된다.
    """

    def test_single_run_keeps_the_preregistered_sentence(self):
        counts = {"arm1": 1, "arm2": 1, "arm4": 1, "arm5": 1}
        self.assertEqual(factorial_inference(counts), SINGLE_RUN_INFERENCE)

    def test_one_single_run_cell_still_keeps_it(self):
        counts = {"arm1": 3, "arm2": 3, "arm4": 3, "arm5": 1}
        self.assertEqual(factorial_inference(counts), SINGLE_RUN_INFERENCE)

    def test_multi_run_states_the_counts_and_drops_the_false_claim(self):
        counts = {"arm1": 3, "arm2": 3, "arm4": 3, "arm5": 2}
        sentence = factorial_inference(counts)
        self.assertNotIn("셀당 런이 하나다", sentence)
        self.assertNotIn("시드 분산을 추정할 수 없다", sentence)
        self.assertIn("arm1 3 · arm2 3 · arm4 3 · arm5 2", sentence)
        # 등록부는 그대로 보수적이다: 검정을 붙였다고 말하지 않는다.
        self.assertIn("검정하지 않는다", sentence)


class FactorialMultiSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.factorial = build_factorial(synthetic_primary(), exclude_truncated=False)

    def test_three_seed_cells_report_their_own_run_count(self):
        self.assertEqual(self.factorial["status"], "available")
        self.assertEqual(self.factorial["n_runs_min"], 3)
        self.assertEqual(
            self.factorial["n_runs_by_arm"],
            {"arm1": 3, "arm2": 3, "arm4": 3, "arm5": 3},
        )
        self.assertIn("arm1 3 · arm2 3 · arm4 3 · arm5 3", self.factorial["inference"])

    def test_per_seed_drops_and_range_replace_a_bare_mean(self):
        cell = self.factorial["cells"]["arm1"]
        per_seed = cell["lens_drop_per_seed"]["corrected_refusal_rate"]
        self.assertEqual(per_seed, {"42": 0.75, "1337": 1.0, "2718": 0.5})
        self.assertEqual(
            cell["lens_drop_seed_range"]["corrected_refusal_rate"],
            {"min": 0.5, "max": 1.0},
        )
        # 평균은 움직이지 않는다.
        self.assertAlmostEqual(
            cell["lens_drop_none_minus_agreeable"]["corrected_refusal_rate"], 0.75
        )
        self.assertNotIn("lens_drop_seed_sd", cell)
        self.assertNotIn("lens_drop_seed_std", cell)

    def test_seed_matched_cells_get_a_paired_persona_interval(self):
        ci = self.factorial["persona_main_effect_ci"]
        self.assertEqual(ci["status"], "available")
        self.assertEqual(ci["seeds"], [42, 1337, 2718])
        entry = ci["per_seed"]["42"]
        self.assertEqual(entry["status"], "available")
        self.assertEqual(entry["n_prompts"], N_SYNTHETIC_PROMPTS)
        self.assertEqual(entry["method"], "paired percentile bootstrap")
        # 프롬프트 단위 점추정은 셀 평균에서 나온 기술통계 대비와 같아야 한다.
        self.assertAlmostEqual(entry["corrected_refusal"]["persona_main_effect"], 0.625)
        low, high = entry["corrected_refusal"]["ci95_bootstrap"]
        self.assertLessEqual(low, 0.625)
        self.assertGreaterEqual(high, 0.625)
        # 합성 harm은 조건마다 상수라 sr_mean 구간은 한 점으로 무너진다.
        self.assertAlmostEqual(entry["sr_mean"]["persona_main_effect"], -0.65)
        self.assertAlmostEqual(entry["sr_mean"]["ci95_bootstrap"][0], -0.65)
        self.assertAlmostEqual(entry["sr_mean"]["ci95_bootstrap"][1], -0.65)

    def test_unmatched_seed_sets_skip_the_interval_instead_of_unpairing(self):
        primary = synthetic_primary({
            "arm1": [42, 1337, 2718], "arm2": [42, 1337, 2718],
            "arm4": [42, 1337, 2718], "arm5": [42],
        })
        ci = build_factorial(primary, exclude_truncated=False)["persona_main_effect_ci"]
        self.assertEqual(ci["status"], "seed_unmatched")
        self.assertNotIn("per_seed", ci)
        self.assertEqual(ci["seeds_by_arm"]["arm5"], [42])

    def test_truncation_exclusion_gets_no_interval(self):
        ci = build_factorial(
            synthetic_primary(), exclude_truncated=True
        )["persona_main_effect_ci"]
        self.assertEqual(ci["status"], "not_computed")
        self.assertNotIn("per_seed", ci)

    def test_markdown_shows_the_run_counts_spread_and_interval(self):
        lines = []
        render_factorial(lines, self.factorial, None)
        text = "\n".join(lines)
        self.assertIn("arm1 3 · arm2 3 · arm4 3 · arm5 3", text)
        self.assertIn("| arm | outcome | per-seed | min | max |", text)
        self.assertIn("s42=0.750, s1337=1.000, s2718=0.500", text)
        self.assertIn("| seed | n prompts | outcome | persona main | ci95 |", text)
        self.assertIn("| 42 | 4 | corrected_refusal | 0.625 |", text)

    def test_markdown_stays_quiet_when_every_cell_has_one_run(self):
        primary = synthetic_primary({arm: [42] for arm in SYNTHETIC_CELL_SPEC})
        factorial = build_factorial(primary, exclude_truncated=False)
        self.assertEqual(factorial["inference"], SINGLE_RUN_INFERENCE)
        lines = []
        render_factorial(lines, factorial, None)
        self.assertNotIn("| arm | outcome | per-seed | min | max |", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
