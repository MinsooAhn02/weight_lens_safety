"""두 생성기가 같은 데이터를 두고 다른 말을 하지 않는지 지킨다.

`recompute_grid.py`는 시드42 기저를 고정으로 쓰고 `analyze_dual_primary.py`는 기저를
시드매칭한다. 둘 다 의도된 설계이고, 산출된 `I_a`는 지금 자릿수까지 같다 —
**그러나 그건 arm 0이 어댑터가 없어 시드 불변이기 때문일 뿐이다.**

arm 0이 시드에 따라 달라지는 순간(무학습이 아닌 기저를 쓰거나, 평가에 비결정성이
들어오거나) 두 산출물은 조용히 갈라진다. 값이 아니라 **전제**를 테스트로 붙잡아 둔다.

전제가 깨지면: 시드매칭 쪽(`analyze_dual_primary.py`)이 정본이다. 논문이 기저의 시드
불변성을 논지로 쓰고 있으므로(R80), 그때는 `corrected_grid_*`를 인용에서 빼야 한다.
"""

import json
import unittest
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
RESULTS = EXPERIMENT / "results"

COHORT = "_env76"
GRID = RESULTS / f"corrected_grid{COHORT}.json"
SUMMARY = RESULTS / f"dual_primary_summary{COHORT}.json"


def _load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _baseline_evals(summary):
    """기저 eval 파일을 요약이 기록한 목록에서 유도한다.

    직접 globbing하면 안 된다. 코호트 접미사가 중첩되기 때문에
    `eval_arm0_r0_s*_env76.json`은 `..._s42_qwen25_7b_env76.json`까지 잡아서
    Llama 기저와 Qwen 기저를 비교하게 된다. 실제로 이 테스트를 처음 쓸 때 그렇게 됐다.
    """
    return [RESULTS / name for name in summary["cohort"]["files"]
            if name.startswith("eval_arm0_")]


class CrossGeneratorTest(unittest.TestCase):
    """두 생성기의 상호작용 값이 어긋나지 않는다."""

    @classmethod
    def setUpClass(cls):
        if not (GRID.exists() and SUMMARY.exists()):
            raise unittest.SkipTest(f"{COHORT} 산출물이 없다")
        cls.grid = _load(GRID)
        cls.summary = _load(SUMMARY)

    def test_interaction_values_agree(self):
        summary_rows = {
            (row["arm"], row["lens"]): row
            for row in self.summary["analyses"]["truncation_included"]["interactions"]
        }
        compared = 0
        for row in self.grid["interaction_contrasts"]["multi_seed_means"]:
            key = (row["arm"], row["lens"])
            if row.get("status") != "available" or key not in summary_rows:
                continue
            compared += 1
            self.assertAlmostEqual(
                row["values"]["normalized"]["I_a"],
                summary_rows[key]["outcomes"]["corrected_refusal"]["I_a"],
                places=12,
                msg=(
                    f"{key}: recompute_grid와 analyze_dual_primary의 I_a가 갈라졌다. "
                    "기저 규약이 다르므로(고정 시드42 vs 시드매칭) 이는 arm 0의 시드 "
                    "불변성이 깨졌다는 뜻이다. 시드매칭 쪽이 정본이다."
                ),
            )
        self.assertGreater(compared, 0, "비교된 셀이 없다 — 산출물 구조가 바뀌었나")

    def test_the_reason_they_agree_still_holds(self):
        """arm 0의 시드별 평가가 실제로 동일한지 — 위 일치의 전제다."""
        paths = _baseline_evals(self.summary)
        if len(paths) < 2:
            self.skipTest("기저 eval이 둘 미만이라 비교할 것이 없다")
        first, *rest = [_load(path) for path in paths]
        for other, path in zip(rest, paths[1:]):
            self.assertEqual(
                first["conditions"].keys(),
                other["conditions"].keys(),
                f"{path.name}: 기저의 렌즈 집합이 다르다",
            )
            for lens, block in first["conditions"].items():
                self.assertEqual(
                    block.get("refusal_rate"),
                    other["conditions"][lens].get("refusal_rate"),
                    msg=(
                        f"{path.name} / {lens}: 무학습 기저가 시드에 따라 달라진다. "
                        "두 생성기의 기저 규약이 다르므로 이제부터 corrected_grid_*와 "
                        "dual_primary_summary_*는 서로 다른 값을 낸다. 논문은 기저의 "
                        "시드 불변성을 논지로 쓰고 있으니(R80) 그 문장부터 다시 본다."
                    ),
                )

    def test_baseline_metadata_difference_is_the_known_one(self):
        """메타데이터 차이는 존재해도 되지만, 알려진 형태여야 한다."""
        summary_rows = {
            (row["arm"], row["lens"]): row
            for row in self.summary["analyses"]["truncation_included"]["interactions"]
        }
        for row in self.grid["interaction_contrasts"]["multi_seed_means"]:
            key = (row["arm"], row["lens"])
            if row.get("status") != "available" or key not in summary_rows:
                continue
            self.assertEqual(
                row["baseline_seeds"], [42],
                f"{key}: recompute_grid의 기저가 고정 시드42가 아니게 됐다",
            )
            self.assertEqual(
                summary_rows[key]["baseline_seeds"],
                summary_rows[key]["treatment_seeds"],
                f"{key}: analyze_dual_primary의 기저가 시드매칭이 아니게 됐다",
            )


if __name__ == "__main__":
    unittest.main()
