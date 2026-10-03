import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT / "src"
ROOT = EXPERIMENT.parents[1]
sys.path.insert(0, str(SRC))

import build_artifact_manifest as bam  # noqa: E402

SCRIPT = SRC / "build_artifact_manifest.py"
MANIFEST_ABS = EXPERIMENT / "results" / "artifact_manifest.json"

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def run_cli(args, root):
    cmd = [sys.executable, str(SCRIPT)] + list(args) + ["--root", str(root)]
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def run_cli_raw(args):
    """Run without --root, for the mutual-exclusion/missing-flag error cases."""
    cmd = [sys.executable, str(SCRIPT)] + list(args)
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def write_real_manifest():
    proc = run_cli(["--write"], ROOT)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(MANIFEST_ABS.read_text(encoding="utf-8"))


def copy_into(temp_root, rel):
    src = ROOT / rel
    dst = temp_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


class ArtifactManifestTest(unittest.TestCase):
    def test_write_twice_is_byte_identical(self):
        first = run_cli(["--write"], ROOT)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        first_bytes = MANIFEST_ABS.read_bytes()

        second = run_cli(["--write"], ROOT)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        second_bytes = MANIFEST_ABS.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(first_bytes.endswith(b"\n"))

    def test_paths_are_relative_posix(self):
        manifest = write_real_manifest()
        paths = [a["path"] for a in manifest["artifacts"]]
        paths += [r["path"] for r in manifest["run_archives"]]
        self.assertTrue(paths, "expected at least one artifact")
        for p in paths:
            self.assertIsNone(_WINDOWS_DRIVE.match(p), p)
            self.assertFalse(p.startswith("/"), p)
            self.assertNotIn("\\", p, p)
            self.assertTrue(p.startswith("branch_1_premise/") or p.startswith("paper/"), p)

    def test_artifacts_sorted_by_path(self):
        manifest = write_real_manifest()
        paths = [a["path"] for a in manifest["artifacts"]]
        self.assertEqual(paths, sorted(paths))

    def test_manifest_does_not_list_itself(self):
        manifest = write_real_manifest()
        paths = {a["path"] for a in manifest["artifacts"]}
        self.assertNotIn(bam.MANIFEST_REL, paths)
        self.assertNotIn(
            "branch_1_premise/experiment/results/artifact_manifest.json", paths
        )

    def test_check_succeeds_against_current_manifest(self):
        write = run_cli(["--write"], ROOT)
        self.assertEqual(write.returncode, 0, write.stdout + write.stderr)
        check = run_cli(["--check"], ROOT)
        self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

    def test_check_fails_on_bit_flip_of_immutable_artifact(self):
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-flip-") as td:
            temp = Path(td)
            rel = "branch_1_premise/experiment/results/eval_arm0_r0_s42.json"
            dst = copy_into(temp, rel)

            write = run_cli(["--write"], temp)
            self.assertEqual(write.returncode, 0, write.stdout + write.stderr)

            data = bytearray(dst.read_bytes())
            data[0] ^= 0xFF
            dst.write_bytes(bytes(data))

            check = run_cli(["--check"], temp)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn(rel, check.stdout)

    def test_check_fails_on_deleted_artifact(self):
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-delete-") as td:
            temp = Path(td)
            rel = "branch_1_premise/experiment/results/eval_arm0_r0_s42.json"
            dst = copy_into(temp, rel)

            write = run_cli(["--write"], temp)
            self.assertEqual(write.returncode, 0, write.stdout + write.stderr)

            dst.unlink()

            check = run_cli(["--check"], temp)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn(rel, check.stdout)
            self.assertIn("MISSING", check.stdout)

    def test_check_fails_on_same_length_different_content(self):
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-samelen-") as td:
            temp = Path(td)
            rel = "branch_1_premise/experiment/results/eval_arm0_r0_s42.json"
            dst = copy_into(temp, rel)

            write = run_cli(["--write"], temp)
            self.assertEqual(write.returncode, 0, write.stdout + write.stderr)

            original = bytearray(dst.read_bytes())
            self.assertGreaterEqual(len(original), 2, "need at least 2 bytes to swap")
            # Swap two bytes that differ, so length is unchanged but content (and sha256) differs.
            swapped = False
            for i in range(len(original) - 1):
                if original[i] != original[i + 1]:
                    original[i], original[i + 1] = original[i + 1], original[i]
                    swapped = True
                    break
            self.assertTrue(swapped, "could not find two differing adjacent bytes to swap")
            tampered = bytes(original)
            self.assertEqual(len(tampered), dst.stat().st_size)
            dst.write_bytes(tampered)

            check = run_cli(["--check"], temp)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn(rel, check.stdout)
            self.assertIn("sha256", check.stdout)

    def test_envless_eval_and_probe_do_not_cause_failure(self):
        # eval_arm0_r0_s42.json / probe_arm0_r0_s1337.json are both canonical result
        # files without an `env` key -- the normal case (only 2/16 eval and 7/18 probe
        # files in the real repo carry one). Must not be treated as an error.
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-noenv-") as td:
            temp = Path(td)
            eval_rel = "branch_1_premise/experiment/results/eval_arm0_r0_s42.json"
            probe_rel = "branch_1_premise/experiment/results/probe_arm0_r0_s1337.json"
            copy_into(temp, eval_rel)
            copy_into(temp, probe_rel)

            real_eval = json.loads((ROOT / eval_rel).read_text(encoding="utf-8"))
            real_probe = json.loads((ROOT / probe_rel).read_text(encoding="utf-8"))
            self.assertNotIn("env", real_eval)
            self.assertNotIn("env", real_probe)

            write = run_cli(["--write"], temp)
            self.assertEqual(write.returncode, 0, write.stdout + write.stderr)
            manifest = json.loads((temp / bam.MANIFEST_REL).read_text(encoding="utf-8"))

            paths = {a["path"] for a in manifest["artifacts"]}
            self.assertIn(eval_rel, paths)
            self.assertIn(probe_rel, paths)
            self.assertEqual(manifest["environments"], [])

            check = run_cli(["--check"], temp)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)

    def test_suffixed_eval_is_added_as_a_distinct_artifact(self):
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-cohort-") as td:
            temp = Path(td)
            legacy_rel = "branch_1_premise/experiment/results/eval_arm0_r0_s42.json"
            legacy = copy_into(temp, legacy_rel)
            payload = json.loads(legacy.read_text(encoding="utf-8"))
            payload["out_suffix"] = "_env76"
            payload["max_new_tokens"] = 256
            suffixed = legacy.with_name("eval_arm0_r0_s42_env76.json")
            suffixed.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            manifest = bam.build_manifest(temp)
            paths = {entry["path"] for entry in manifest["artifacts"]}
            suffixed_rel = (
                "branch_1_premise/experiment/results/eval_arm0_r0_s42_env76.json"
            )
            self.assertIn(legacy_rel, paths)
            self.assertIn(suffixed_rel, paths)

    def test_r13_ifeval_manifest_is_discovered(self):
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-ifeval-") as td:
            temp = Path(td)
            rel = (
                "branch_1_premise/experiment/results/"
                "ifeval_manifest_arm1_r3_s42_r13_benign_env76.json"
            )
            copy_into(temp, rel)

            manifest = bam.build_manifest(temp)
            entry = next(a for a in manifest["artifacts"] if a["path"] == rel)
            self.assertEqual(entry["role"], "ifeval_manifest")

    def test_check_fails_on_new_unrecorded_artifact(self):
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-unrecorded-") as td:
            temp = Path(td)
            first = "branch_1_premise/experiment/results/eval_arm0_r0_s42.json"
            copy_into(temp, first)
            write = run_cli(["--write"], temp)
            self.assertEqual(write.returncode, 0, write.stdout + write.stderr)

            added = "branch_1_premise/experiment/results/eval_arm1_r1_s42.json"
            copy_into(temp, added)
            check = run_cli(["--check"], temp)
            self.assertNotEqual(check.returncode, 0)
            self.assertIn(f"UNRECORDED {added}", check.stdout)

    def test_write_and_check_together_or_neither_is_an_error(self):
        both = run_cli_raw(["--write", "--check", "--root", str(ROOT)])
        self.assertNotEqual(both.returncode, 0)

        neither = run_cli_raw(["--root", str(ROOT)])
        self.assertNotEqual(neither.returncode, 0)

    def test_mutable_artifact_change_is_a_note_not_a_failure(self):
        with tempfile.TemporaryDirectory(prefix="artifact-manifest-mutable-") as td:
            temp = Path(td)
            rel = "branch_1_premise/experiment/annotation/m03_labels_pass2.tsv"
            dst = copy_into(temp, rel)

            write = run_cli(["--write"], temp)
            self.assertEqual(write.returncode, 0, write.stdout + write.stderr)
            manifest = json.loads((temp / bam.MANIFEST_REL).read_text(encoding="utf-8"))
            entry = next(a for a in manifest["artifacts"] if a["path"] == rel)
            self.assertTrue(entry.get("mutable"))

            data = bytearray(dst.read_bytes())
            data.extend(b"\ntampered-row\n")
            dst.write_bytes(bytes(data))

            check = run_cli(["--check"], temp)
            self.assertEqual(check.returncode, 0, check.stdout + check.stderr)
            self.assertIn("NOTE", check.stdout)
            self.assertIn(rel, check.stdout)


if __name__ == "__main__":
    unittest.main()
