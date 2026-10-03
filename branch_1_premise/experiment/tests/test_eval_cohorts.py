import json
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT / "src"
sys.path.insert(0, str(SRC))

from eval_cohorts import (  # noqa: E402
    cohort_output_path,
    format_cohort_listing,
    select_eval_cohort,
)
from analyze_dual_primary import load_eval_payloads  # noqa: E402
from recompute_grid import load_runs  # noqa: E402


def write_eval(results_dir, filename, suffix=None, max_new_tokens=None):
    payload = {
        "arm": "arm1",
        "round": 3,
        "seed": 1337,
        "conditions": {},
    }
    if suffix is not None:
        payload["out_suffix"] = suffix
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    (results_dir / filename).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


class EvalCohortTest(unittest.TestCase):
    def test_legacy_and_suffixed_same_run_do_not_collide(self):
        with tempfile.TemporaryDirectory(prefix="eval-cohort-") as td:
            results = Path(td)
            write_eval(results, "eval_arm1_r3_s1337.json")
            write_eval(
                results,
                "eval_arm1_r3_s1337_env76.json",
                suffix="_env76",
                max_new_tokens=256,
            )

            legacy, legacy_meta = select_eval_cohort(results)
            env76, env76_meta = select_eval_cohort(results, "_env76")
            legacy_payloads = load_eval_payloads(results)
            env76_payloads = load_eval_payloads(results, "_env76")
            _, legacy_files = load_runs(results)
            _, env76_files = load_runs(results, "_env76")

            self.assertEqual([entry["path"].name for entry in legacy], [
                "eval_arm1_r3_s1337.json"
            ])
            self.assertEqual([entry["path"].name for entry in env76], [
                "eval_arm1_r3_s1337_env76.json"
            ])
            self.assertEqual(legacy_meta["max_new_tokens"], [None])
            self.assertEqual(env76_meta["max_new_tokens"], [256])
            self.assertEqual(list(legacy_payloads), [("arm1", 3.0, 1337)])
            self.assertEqual(list(env76_payloads), [("arm1", 3.0, 1337)])
            self.assertEqual(legacy_files, ["eval_arm1_r3_s1337.json"])
            self.assertEqual(env76_files, ["eval_arm1_r3_s1337_env76.json"])

    def test_absent_cohort_names_available_cohorts(self):
        with tempfile.TemporaryDirectory(prefix="eval-cohort-absent-") as td:
            results = Path(td)
            write_eval(results, "eval_arm1_r3_s1337.json")
            write_eval(
                results,
                "eval_arm1_r3_s1337_env76.json",
                suffix="_env76",
                max_new_tokens=256,
            )

            with self.assertRaisesRegex(
                ValueError, r"사용 가능 cohort: .*legacy.*_env76"
            ):
                select_eval_cohort(results, "_env7")

    def test_list_includes_provenance_summary(self):
        with tempfile.TemporaryDirectory(prefix="eval-cohort-list-") as td:
            results = Path(td)
            write_eval(
                results,
                "eval_arm1_r3_s1337_env76.json",
                suffix="_env76",
                max_new_tokens=256,
            )
            listing = format_cohort_listing(results)
            self.assertIn("suffix='_env76'", listing)
            self.assertIn('arms=["arm1"]', listing)
            self.assertIn("rounds=[3.0]", listing)
            self.assertIn("seeds=[1337]", listing)
            self.assertIn("max_new_tokens=[256]", listing)
            self.assertIn("env_stamps=[null]", listing)

    def test_duplicate_still_fails_inside_one_cohort(self):
        with tempfile.TemporaryDirectory(prefix="eval-cohort-duplicate-") as td:
            results = Path(td)
            write_eval(results, "eval_arm1_r3_s1337_env76.json", suffix="_env76")
            write_eval(results, "eval_arm1_r3p0_s1337_env76.json", suffix="_env76")
            with self.assertRaisesRegex(ValueError, "내부 중복 eval 런"):
                select_eval_cohort(results, "_env76")

    def test_filename_payload_suffix_disagreement_is_hard_error(self):
        with tempfile.TemporaryDirectory(prefix="eval-cohort-mismatch-") as td:
            results = Path(td)
            write_eval(
                results,
                "eval_arm1_r3_s1337_env76.json",
                suffix="_env76_max512",
                max_new_tokens=512,
            )
            with self.assertRaisesRegex(ValueError, "파일명/payload provenance 불일치"):
                select_eval_cohort(results, "_env76")

    def test_nondefault_output_name_keeps_cohorts_separate(self):
        results = Path("results")
        self.assertEqual(
            cohort_output_path(results, "corrected_grid", "", ".json"),
            results / "corrected_grid.json",
        )
        self.assertEqual(
            cohort_output_path(results, "corrected_grid", "_env76", ".json"),
            results / "corrected_grid_env76.json",
        )


if __name__ == "__main__":
    unittest.main()
