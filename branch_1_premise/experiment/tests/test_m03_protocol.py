import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT / "src"
sys.path.insert(0, str(SRC))

import label_m03  # noqa: E402
import sample_m03  # noqa: E402
import score_m03  # noqa: E402
from label_m03 import parse_entry  # noqa: E402
from sample_m03 import consistency_errors, looks_truncated  # noqa: E402
from score_m03 import agreement_stats, paired_var  # noqa: E402


class M03ProtocolTest(unittest.TestCase):
    @staticmethod
    def set_valid_pass_gap(annotation):
        now = time.time()
        os.utime(annotation / "m03_labels_pass1.tsv", (now - 8 * 86400, now - 8 * 86400))
        os.utime(annotation / "m03_labels_pass2.tsv", (now, now))

    def test_consistency_invariants(self):
        valid = [
            dict(refusal="R", substantive="0", hedge="0", harmful="0", conf="3"),
            dict(refusal="R", substantive="1", hedge="0", harmful="0", conf="2"),
            dict(refusal="P", substantive="2", hedge="1", harmful="2", conf="2"),
            dict(refusal="C", substantive="0", hedge="0", harmful="0", conf="1"),
            dict(refusal="X", substantive="X", hedge="X", harmful="X", conf="1"),
        ]
        for values in valid:
            self.assertEqual(consistency_errors(values), [], values)

        invalid = [
            (dict(refusal="R", substantive="2", hedge="0", harmful="1", conf="3"), "E1"),
            (dict(refusal="P", substantive="1", hedge="0", harmful="1", conf="2"), "E2"),
            (dict(refusal="R", substantive="0", hedge="1", harmful="0", conf="3"), "E3"),
            (dict(refusal="C", substantive="1", hedge="1", harmful="1", conf="2"), "E4"),
            (dict(refusal="C", substantive="0", hedge="0", harmful="2", conf="2"), "E5"),
            (dict(refusal="X", substantive="X", hedge="X", harmful="X", conf="3"), "E6"),
        ]
        for values, code in invalid:
            self.assertTrue(any(x.startswith(code) for x in consistency_errors(values)), values)

    def test_cli_parser_uses_protocol_field_order(self):
        parsed, error = parse_entry("C 2 1 1 2 0 2 visible-note")
        self.assertIsNone(error)
        values, note = parsed
        self.assertEqual(
            values,
            dict(refusal="C", substantive="2", coherent="1", hedge="1",
                 harmful="2", offtopic="0", conf="2"),
        )
        self.assertEqual(note, "visible-note")
        self.assertEqual(consistency_errors(values), [])

    def test_multicategory_pabak_and_singleton_variance(self):
        stats = agreement_stats([(1, "A", "A"), (1, "A", "B")], ("A", "B", "C", "X"))
        self.assertAlmostEqual(stats["raw"], 0.5)
        self.assertAlmostEqual(stats["pabak"], 1 / 3)
        self.assertIsNone(paired_var({"S": [1]}, {"S": {"N": 10}}))
        self.assertEqual(paired_var({"S": [1]}, {"S": {"N": 1}}), 0.0)

    def test_generated_document_hashes_and_blinding(self):
        manifest = json.loads(
            (EXPERIMENT / "results" / "m03_sample_manifest.json").read_text(encoding="utf-8"))
        for filename, key in (
                ("M03_CODEBOOK.md", "codebook_sha"),
                ("M03_ANALYSIS_PROTOCOL.md", "analysis_protocol_sha")):
            content = (EXPERIMENT / "annotation" / filename).read_text(encoding="utf-8")
            self.assertEqual(hashlib.sha256(content.encode("utf-8")).hexdigest()[:12], manifest[key])

        for pass_no in (1, 2):
            path = EXPERIMENT / "results" / f"m03_items_pass{pass_no}.jsonl"
            item_meta = manifest["item_files"][f"pass{pass_no}"]
            self.assertEqual(item_meta["file"], path.name)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item_meta["sha256"])
            for line in path.read_text(encoding="utf-8").splitlines():
                self.assertEqual(set(json.loads(line)), {"item_id", "prompt", "response"})

    def test_validate_only_treats_wholly_blank_rows_as_pending(self):
        # 라이브 annotation/을 보면 pass1이 채워지는 순간 이 테스트가 썩는다.
        # 빈 표본을 따로 만들어서 "빈 행 = 미완, 오류 아님"만 못박는다.
        with tempfile.TemporaryDirectory(prefix="m03-blank-") as td:
            temp = Path(td)
            results = temp / "results"
            annotation = temp / "annotation"
            results.mkdir()
            for filename in ("eval_arm1_r3_s42.json", "eval_arm3_r3_s42.json"):
                shutil.copy2(EXPERIMENT / "results" / filename, results / filename)
            generate = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "sample_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(annotation),
                    "--seed", "42",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(generate.returncode, 0, generate.stdout + generate.stderr)

            run = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(annotation),
                    "--validate_only",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("pass1 0/292", run.stdout)
            self.assertIn("오류 0건", run.stdout)

    def test_default_generator_stdout_does_not_leak_analysis_direction(self):
        with tempfile.TemporaryDirectory(prefix="m03-generate-") as td:
            temp = Path(td)
            results = temp / "results"
            annotation = temp / "annotation"
            results.mkdir()
            for filename in ("eval_arm1_r3_s42.json", "eval_arm3_r3_s42.json"):
                shutil.copy2(EXPERIMENT / "results" / filename, results / filename)
            run = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "sample_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(annotation),
                    "--seed", "42",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("분석 방향·arm·층 수치는 콘솔에서 숨겼다", run.stdout)
            for leaked in ("| arm |", "Δ_regex", "사전등록 민감도 범위", "역전쌍"):
                self.assertNotIn(leaked, run.stdout)
            protected = [
                annotation / "M03_CODEBOOK.md",
                annotation / "m03_labels_pass1.tsv",
                results / "m03_sample_manifest.json",
            ]
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
            collision = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "sample_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(annotation),
                    "--seed", "42",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(collision.returncode, 0)
            self.assertIn("아무 파일도 바꾸지 않았다", collision.stdout + collision.stderr)
            self.assertEqual(
                before,
                {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
            )

            # 항목 JSONL만 바뀌면 라벨러가 화면을 열기 전에 막는다.
            items_path = results / "m03_items_pass1.jsonl"
            original_items = items_path.read_bytes()
            manifest_path = results / "m03_sample_manifest.json"
            original_manifest = manifest_path.read_bytes()
            item_lines = items_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(item_lines[0])
            first["response"] += " tampered"
            item_lines[0] = json.dumps(first, ensure_ascii=False)
            items_path.write_text("\n".join(item_lines) + "\n", encoding="utf-8")
            label = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "label_m03.py"),
                    "--pass", "1",
                    "--results_dir", str(results),
                    "--annotation_dir", str(annotation),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(label.returncode, 0)
            self.assertIn("문항이 생성 당시와 달라졌다", label.stdout + label.stderr)

            # manifest의 항목 SHA까지 함께 바꿔도 채점기는 원본 eval 대조로 막는다.
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["item_files"]["pass1"]["sha256"] = hashlib.sha256(
                items_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            score = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(annotation),
                    "--validate_only",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(score.returncode, 0)
            self.assertIn("prompt/response가 원본 eval과 다르다", score.stdout + score.stderr)

            # 층·가중치 manifest를 손으로 바꿔도 원본에서 결정적으로 재계산해 막는다.
            items_path.write_bytes(original_items)
            manifest_path.write_bytes(original_manifest)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["strata"]["arm1"]["P0"]["w"] += 0.25
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            design = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(annotation),
                    "--validate_only",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(design.returncode, 0)
            self.assertIn("표본 설계 무결성 오류", design.stdout + design.stderr)

    def test_full_scorer_smoke_with_synthetic_consistent_labels(self):
        with tempfile.TemporaryDirectory(prefix="m03-smoke-") as td:
            temp = Path(td)
            results = temp / "results"
            ann = temp / "annotation"
            results.mkdir()
            ann.mkdir()
            # ★results도 격리한다. 이 케이스는 pass1≡pass2(불일치 0)인 합성 라벨을 쓰므로
            # 실제 results/를 가리키면 **거기 있는 진짜 pass3 산출물**과 어긋난다 —
            # 2026-08-08에 실제 조정 패스를 돌리자마자 이 테스트가 빨간불이 됐다.
            # `test_pass3_field_level_end_to_end`와 같은 방식으로 채점기가 읽는 것만 복사한다.
            for filename in (
                    "eval_arm1_r3_s42.json", "eval_arm3_r3_s42.json",
                    "m03_sample_manifest.json", "m03_items_pass1.jsonl",
                    "m03_items_pass2.jsonl"):
                shutil.copy2(EXPERIMENT / "results" / filename, results / filename)
            for filename in ("M03_CODEBOOK.md", "M03_ANALYSIS_PROTOCOL.md"):
                shutil.copy2(EXPERIMENT / "annotation" / filename, ann / filename)

            for pass_no in (1, 2):
                source = EXPERIMENT / "annotation" / f"m03_labels_pass{pass_no}.tsv"
                lines = []
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line or line.startswith("item_id"):
                        lines.append(line)
                    else:
                        iid = line.split("\t", 1)[0]
                        lines.append(f"{iid}\tC\t0\t1\t0\t0\t0\t3\t")
                (ann / source.name).write_text("\n".join(lines) + "\n", encoding="utf-8")

            too_soon = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                    "--out", str(temp / "too-soon.md"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(too_soon.returncode, 0)
            self.assertIn("최소 7일", too_soon.stdout + too_soon.stderr)
            self.set_valid_pass_gap(ann)
            report = temp / "report.md"
            run = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                    "--out", str(report),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            text = report.read_text(encoding="utf-8")
            for heading in (
                    "### 8.1 `conf≥2` 사전 지정 부분집합",
                    "### 8.2 유효하지만 내용 없는 비거절 진단",
                    "### 8.3 `offtopic` 진단",
                    "### 8.4 사람 `harmful`과 자동 judge",
                    "### 8.5 필드 간 불변식"):
                self.assertIn(heading, text)

            # 고확신 쌍이 한 건도 없어 모든 층이 비어도 수치를 0으로 오인해 인용하지 않아야 한다.
            for pass_no in (1, 2):
                path = ann / f"m03_labels_pass{pass_no}.tsv"
                path.write_text(
                    path.read_text(encoding="utf-8").replace("\t0\t3\t", "\t0\t1\t"),
                    encoding="utf-8",
                )
            self.set_valid_pass_gap(ann)
            low_report = temp / "report-low-conf.md"
            low_run = subprocess.run(
                [
                    sys.executable,
                    str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                    "--out", str(low_report),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(low_run.returncode, 0, low_run.stdout + low_run.stderr)
            self.assertIn("인용 금지(빈 층:", low_report.read_text(encoding="utf-8"))

    def test_pass3_field_level_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix="m03-pass3-") as td:
            temp = Path(td)
            results = temp / "results"
            ann = temp / "annotation"
            results.mkdir()
            ann.mkdir()
            for filename in (
                    "eval_arm1_r3_s42.json", "eval_arm3_r3_s42.json",
                    "m03_sample_manifest.json", "m03_items_pass1.jsonl",
                    "m03_items_pass2.jsonl"):
                shutil.copy2(EXPERIMENT / "results" / filename, results / filename)
            for filename in ("M03_CODEBOOK.md", "M03_ANALYSIS_PROTOCOL.md"):
                shutil.copy2(EXPERIMENT / "annotation" / filename, ann / filename)

            first_id = None
            for pass_no in (1, 2):
                source = EXPERIMENT / "annotation" / f"m03_labels_pass{pass_no}.tsv"
                lines = []
                for line in source.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line or line.startswith("item_id"):
                        lines.append(line)
                        continue
                    iid = line.split("\t", 1)[0]
                    if first_id is None:
                        first_id = iid
                    coherent = "0" if pass_no == 2 and iid == first_id else "1"
                    lines.append(f"{iid}\tC\t0\t{coherent}\t0\t0\t0\t3\t")
                (ann / source.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.set_valid_pass_gap(ann)

            emit = subprocess.run(
                [
                    sys.executable, str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                    "--emit_adjudication",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(emit.returncode, 0, emit.stdout + emit.stderr)
            pass3_items = [
                json.loads(x) for x in
                (results / "m03_items_pass3.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()
            ]
            self.assertEqual(len(pass3_items), 1)
            self.assertEqual(pass3_items[0]["item_id"], first_id)
            self.assertEqual(pass3_items[0]["required_fields"], ["coherent"])

            # pass3 문항도 시트 헤더의 SHA와 묶여 있어 변조된 화면으로 라벨링할 수 없다.
            pass3_items_path = results / "m03_items_pass3.jsonl"
            original_pass3_items = pass3_items_path.read_bytes()
            tampered = dict(pass3_items[0])
            tampered["response"] += " tampered"
            pass3_items_path.write_text(
                json.dumps(tampered, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            blocked_label = subprocess.run(
                [
                    sys.executable, str(SRC / "label_m03.py"),
                    "--pass", "3",
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(blocked_label.returncode, 0)
            self.assertIn("문항이 생성 당시와 달라졌다",
                          blocked_label.stdout + blocked_label.stderr)
            pass3_items_path.write_bytes(original_pass3_items)

            premature = subprocess.run(
                [
                    sys.executable, str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                    "--out", str(temp / "premature.md"),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(premature.returncode, 0)
            self.assertIn("라벨 행 전체가 비었다", premature.stdout + premature.stderr)
            self.assertFalse((temp / "premature.md").exists())

            label = subprocess.run(
                [
                    sys.executable, str(SRC / "label_m03.py"),
                    "--pass", "3",
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                ],
                input="1\n",
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(label.returncode, 0, label.stdout + label.stderr)
            self.assertIn("재판정 필드: coherent", label.stdout)

            report = temp / "final.md"
            final = subprocess.run(
                [
                    sys.executable, str(SRC / "score_m03.py"),
                    "--results_dir", str(results),
                    "--annotation_dir", str(ann),
                    "--out", str(report),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(final.returncode, 0, final.stdout + final.stderr)
            self.assertTrue(report.exists())

    def test_truncation_verdict_is_one_shared_definition(self):
        """절단 판정은 정의가 하나여야 한다.

        샘플러·라벨링 CLI·집계가 각자 휴리스틱을 들고 있으면 문서의 비율과 화면 경고와
        보고서 수치가 조용히 어긋난다. 세 모듈이 같은 함수 객체를 쓰는지 직접 못박는다.
        """
        self.assertIs(label_m03.looks_truncated, sample_m03.looks_truncated)
        self.assertIs(score_m03.looks_truncated, sample_m03.looks_truncated)

        truncated = [
            "다음 절차를 따르십시오:",            # `:`는 종결이 아니다
            "먼저 준비물을 모은다;",              # `;`도 아니다
            "도와드릴 수 없습니다",                # 종결부호 자체가 없다
            "재료:\n1. 소금 한 스푼.\n2.",        # 항목 번호만 나오고 끊겼다
            "목록:\n- ",                          # 맨 불릿으로 끝났다
        ]
        for s in truncated:
            self.assertTrue(looks_truncated(s), repr(s))

        intact = [
            "",                                   # 빈 응답은 절단이 아니다
            "   \n  ",
            "도와드릴 수 없습니다.",              # 단문 거절도 정상 종결이다
            "정말입니까?",
            "그만!",
            '그는 "안 된다"라고 했다.',
            "재료:\n1. 소금 한 스푼.\n2. 물 두 컵.",
            "답변입니다.\n\n",                    # 뒤 공백은 rstrip된다
        ]
        for s in intact:
            self.assertFalse(looks_truncated(s), repr(s))


if __name__ == "__main__":
    unittest.main()
