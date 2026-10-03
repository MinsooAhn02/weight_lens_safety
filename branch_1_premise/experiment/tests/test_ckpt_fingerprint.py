"""체크포인트 지문 — 다른 가중치를 실제로 갈라내는지, 그리고 인수 검사가 그걸 읽는지.

이 지문이 있었다면 8B의 512 감사가 헤드라인 격자와 다른 체크포인트에서 돌았다는 것을
기계가 잡았을 것이다(`results/ckpt_identity_audit.json`, `R11_GATE2_SAME_CKPT.md`).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402

import intake_results  # noqa: E402
from eval_refusal import FINGERPRINT_ELEMS, ckpt_fingerprint  # noqa: E402


class TinyModel(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.a = torch.nn.Linear(8, 8)
        self.b = torch.nn.Linear(8, 4)


class FingerprintTest(unittest.TestCase):
    def test_same_weights_give_the_same_digest(self):
        m = TinyModel(0)
        self.assertEqual(ckpt_fingerprint(m)["sha256"], ckpt_fingerprint(m)["sha256"])

    def test_different_training_run_gives_a_different_digest(self):
        self.assertNotEqual(ckpt_fingerprint(TinyModel(0))["sha256"],
                            ckpt_fingerprint(TinyModel(1))["sha256"])

    def test_a_single_changed_weight_is_caught(self):
        m = TinyModel(0)
        before = ckpt_fingerprint(m)["sha256"]
        with torch.no_grad():
            m.a.weight[0, 0] += 1.0
        self.assertNotEqual(before, ckpt_fingerprint(m)["sha256"])

    def test_shape_metadata_is_recorded(self):
        fp = ckpt_fingerprint(TinyModel(0))
        self.assertEqual(fp["n_tensors"], 4)          # 두 Linear의 weight·bias
        self.assertEqual(fp["elems_per_tensor"], FINGERPRINT_ELEMS)


class IntakeFingerprintCheckTest(unittest.TestCase):
    """`_check_ckpt_fingerprint`는 **막지 않고 경고한다** — 다름 자체가 결과일 수 있다."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = intake_results.RESULTS
        intake_results.RESULTS = Path(self._tmp.name)
        self.addCleanup(self._restore)

    def _restore(self):
        intake_results.RESULTS = self._orig
        self._tmp.cleanup()

    @staticmethod
    def _payload(digest):
        return {"arm": "arm1", "round": 3.0, "seed": 42,
                "ckpt_fingerprint": {"sha256": digest}}

    def _write_existing(self, name, payload):
        (intake_results.RESULTS / name).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_missing_fingerprint_warns_but_does_not_block(self):
        findings = intake_results._check_ckpt_fingerprint({"arm": "arm1", "round": 3.0, "seed": 42})
        self.assertEqual([f.level for f in findings], ["WARN"])
        self.assertFalse(any(f.fatal for f in findings))

    def test_matching_fingerprint_is_reported_ok(self):
        self._write_existing("eval_arm1_r3_s42_env76.json", self._payload("abc"))
        findings = intake_results._check_ckpt_fingerprint(self._payload("abc"))
        self.assertEqual([f.level for f in findings], ["OK"])

    def test_differing_fingerprint_warns_and_is_not_fatal(self):
        self._write_existing("eval_arm1_r3_s42_env76.json", self._payload("abc"))
        findings = intake_results._check_ckpt_fingerprint(self._payload("zzz"))
        self.assertEqual([f.level for f in findings], ["WARN"])
        self.assertFalse(any(f.fatal for f in findings))
        self.assertIn("가중치가 다르다", findings[0].message)

    def test_a_different_cell_is_not_compared(self):
        """같은 시드라도 arm이나 라운드가 다르면 같은 체크포인트일 이유가 없다."""
        other = dict(self._payload("zzz"), arm="arm3")
        self._write_existing("eval_arm3_r3_s42_env76.json", other)
        self.assertEqual(intake_results._check_ckpt_fingerprint(self._payload("abc")), [])


if __name__ == "__main__":
    unittest.main()
