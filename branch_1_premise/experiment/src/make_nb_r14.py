"""Build the R14 matched-factorial Kaggle notebook from the proven R12 shell.

Do not hand-edit ``nb_r14.ipynb``.  Edit this generator and rebuild:

    python src/make_nb_r14.py
    python src/check_notebooks.py nb_r14.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "nb_r12.ipynb"
OUTPUT = ROOT / "nb_r14.ipynb"


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


nb = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = nb["cells"]

cells[0]["source"] = lines("""# R14 — ICLR fully matched persona × target-breadth factorial

R12의 검증된 Kaggle 설치·학습·7렌즈 평가 셸을 재사용한다. BATCH 0은 동일한 새
instruction catalog에서 arm1/2/4/5 응답을 대칭 생성하고, BATCH 1–5는 그 고정 ZIP만
복원해 12개 학습 셀을 실행한다. 세션에서 바꾸는 값은 `BATCH` 하나뿐이다.
""")

cells[6]["source"] = lines(r'''import os, shutil, sys, glob

EXP_ROOT = "/kaggle/working/experiment"
CKPT_ROOT = "/kaggle/tmp/ckpt"

def _input_tree():
    return sorted(glob.glob("/kaggle/input/*")
                  + glob.glob("/kaggle/input/*/*")
                  + glob.glob("/kaggle/input/*/*/*"))

if not os.path.exists(EXP_ROOT):
    hits = glob.glob("/kaggle/input/**/src/eval_refusal.py", recursive=True)
    if not hits:
        print("\n".join(_input_tree()))
        raise SystemExit("branch-1-premise code Dataset을 이 노트북에 붙이세요")
    input_dir = os.path.dirname(os.path.dirname(hits[0]))
    shutil.copytree(input_dir, EXP_ROOT)
os.makedirs(CKPT_ROOT, exist_ok=True)
os.chdir(EXP_ROOT)
sys.path.insert(0, os.path.join(EXP_ROOT, "src"))

for path, needle in [
    ("src/gen_r14_matched_data.py", "r14_matched_factorial_data_v1"),
    ("src/train_round.py", "save_adapter_dir"),
    ("src/eval_refusal.py", "out_suffix"),
    ("configs/arms.json", '"nb_r14": "R14a"'),
]:
    assert os.path.exists(path), f"R14 code Dataset 아님: {path} 없음"
    assert needle in open(path, encoding="utf-8").read(), f"R14 code Dataset이 낡음: {path}"
print("[gate] R14 code payload 확인")
''')

cells[9]["source"] = lines("""## 4. 이번 세션 — `BATCH` 하나만 바꾼다

- `0`: 공통 데이터 생성 전용. 결과 ZIP을 회수·수동검토한 뒤 Kaggle Dataset으로 올린다.
- `1`–`5`: 고정 데이터 ZIP을 attach하고 지정된 셀만 학습·7렌즈 평가한다.
""")

cells[10]["source"] = lines(r'''import glob, hashlib, json, os, shutil, zipfile
from pathlib import Path

NB_NAME = "nb_r14"
NB_REV = "R14a"
revision = json.load(open("configs/arms.json", encoding="utf-8"))["nb_revisions"].get(NB_NAME)
assert revision == NB_REV, f"노트북/Dataset revision 불일치: {NB_REV} != {revision}"

UNSLOTH_PIN = "2026.7.6"
LENS_LIST = ["none", "agreeable", "principled", "agreeable_para1",
             "agreeable_para2", "agreeable_weak", "agreeable_strong"]
LENSES = ",".join(LENS_LIST)
OUT_SUFFIX = "_r14_matched_factorial_env76"
MAX_NEW_TOKENS = 256
ADAPTER_ROOT = "/kaggle/working/adapters"
DATA_ZIP_OUT = Path("/kaggle/working/r14_matched_data.zip")

from common import env_versions
runtime = env_versions()
assert runtime.get("unsloth") == UNSLOTH_PIN, runtime
assert BASE == "unsloth/Meta-Llama-3.1-8B-Instruct", BASE

# ★ Kaggle notebook copy마다 이것 하나만 바꾼다.
BATCH = 0

# BATCH 0은 데이터 생성. 나머지는 (arm, training seed) 목록.
BATCHES = {
    1: [("arm0", 42), ("arm1", 42), ("arm2", 42)],
    2: [("arm4", 42), ("arm5", 42), ("arm1", 1337)],
    3: [("arm2", 1337), ("arm4", 1337), ("arm5", 1337)],
    4: [("arm1", 2718), ("arm2", 2718), ("arm4", 2718)],
    5: [("arm5", 2718)],
}
assert BATCH in range(0, 6)

def restore_r14_data():
    zip_hits = sorted(glob.glob("/kaggle/input/**/r14_matched_data.zip", recursive=True))
    manifest_hits = sorted(glob.glob("/kaggle/input/**/manifest.json", recursive=True))
    source = None
    if zip_hits:
        if os.path.exists("data"):
            shutil.rmtree("data")
        shutil.unpack_archive(zip_hits[0], "data")
        source = zip_hits[0]
    else:
        for candidate in manifest_hits:
            try:
                probe = json.load(open(candidate, encoding="utf-8"))
            except Exception:
                continue
            if probe.get("schema") == "r14_matched_factorial_data_v1":
                if os.path.exists("data"):
                    shutil.rmtree("data")
                shutil.copytree(os.path.dirname(candidate), "data")
                source = os.path.dirname(candidate)
                break
    assert source, "training BATCH에는 frozen r14-matched-data Dataset을 attach해야 함"
    manifest = json.load(open("data/manifest.json", encoding="utf-8"))
    assert manifest["schema"] == "r14_matched_factorial_data_v1"
    assert manifest["target_pairs_per_arm"] == 384
    assert manifest["rows_per_round"] == 128
    assert manifest["observed_budget_gap"] <= manifest["max_budget_gap"] <= 0.01
    assert manifest["arms_config_sha256"] == hashlib.sha256(Path("configs/arms.json").read_bytes()).hexdigest()
    assert manifest["train_config_sha256"] == hashlib.sha256(Path("configs/train_config.json").read_bytes()).hexdigest()
    for member in manifest["members"]:
        path = Path("data") / member
        assert path.is_file(), member
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"][member]
    ordered = manifest["ordered_instruction_ids"]
    for arm in ("arm1", "arm2", "arm4", "arm5"):
        ids = []
        for round_no in (1, 2, 3):
            rows = [json.loads(x) for x in open(f"data/{arm}/round{round_no}.jsonl", encoding="utf-8") if x.strip()]
            assert len(rows) == 128
            ids.extend(hashlib.sha256(row["instruction"].encode("utf-8")).hexdigest() for row in rows)
        assert ids == ordered, f"{arm}: instruction ID/order mismatch"
    print(f"[data] frozen R14 data restored: {source}")
    print(f"[data] token totals={manifest['nonpadding_tokens']} gap={manifest['observed_budget_gap']:.6f}")

if BATCH:
    restore_r14_data()
''')

cells[11]["source"] = lines("""## 5. 실행

BATCH 0은 데이터 ZIP만 만든다. BATCH 1–5는 해당 셀을 순서대로 학습하고 각 셀 직후
results snapshot을 남긴다. 한 셀의 7렌즈는 같은 세션에서 끝내야 한다.
""")

cells[12]["source"] = lines(r'''import os, shutil

def snap(tag):
    path = shutil.make_archive(f"/kaggle/working/r14_results_{tag}", "zip",
                               "/kaggle/working/experiment/results")
    print(f"[snap] {path} | {os.path.getsize(path)/1e6:.2f} MB", flush=True)

if BATCH == 0:
    run("python src/gen_r14_matched_data.py --seed 20260824 --candidate_pairs 1000 --target_pairs 384 --max_budget_gap 0.01 --output_root r14_data")
    run("python src/check_narrowness.py --archive __missing_r14__.zip --reference_root r14_data --generated_root r14_data --require_generated --sample_size 384")
    DATA_ZIP_OUT = Path(shutil.make_archive(
        "/kaggle/working/r14_matched_data", "zip", root_dir="r14_data"))
    print(f"[artifact] {DATA_ZIP_OUT} | {DATA_ZIP_OUT.stat().st_size/1e6:.2f} MB")
    print("[STOP] manual_review_sample.jsonl 80건 확인 후에만 이 ZIP을 Kaggle Dataset으로 올린다")
else:
    print(f"R14 BATCH {BATCH}: {BATCHES[BATCH]} | 7 lenses @ {MAX_NEW_TOKENS}")
    for arm, seed in BATCHES[BATCH]:
        print(f"\n===== {arm} x seed {seed} =====", flush=True)
        if arm == "arm0":
            run_baseline(seed, lenses=LENSES, mmlu=False,
                         max_new_tokens=MAX_NEW_TOKENS, out_suffix=OUT_SUFFIX)
        else:
            run_session(arm, seed, lenses=LENSES, eval_rounds=(3,),
                        probe=False, mmlu=False, max_new_tokens=MAX_NEW_TOKENS,
                        out_suffix=OUT_SUFFIX, save_adapters=ADAPTER_ROOT)
        snap(f"batch{BATCH}_{arm}_s{seed}")
''')

cells[13]["source"] = lines("""## 6. 결과 회수

BATCH 0은 `r14_matched_data.zip`, 학습 BATCH는 results와 adapters ZIP을 모두 받는다.
""")

cells[14]["source"] = lines(r'''import glob, os, shutil

if BATCH == 0:
    assert DATA_ZIP_OUT.is_file()
    print("data:", DATA_ZIP_OUT)
else:
    tag = f"batch{BATCH}"
    results_zip = shutil.make_archive(f"/kaggle/working/r14_results_{tag}", "zip",
                                      "/kaggle/working/experiment/results")
    adapters_zip = shutil.make_archive(f"/kaggle/working/r14_adapters_{tag}", "zip",
                                       ADAPTER_ROOT) if os.path.exists(ADAPTER_ROOT) else None
    print("results:", results_zip)
    print("adapters:", adapters_zip)
    for path in sorted(glob.glob("/kaggle/working/experiment/results/*.json")):
        print(os.path.basename(path), os.path.getsize(path))
''')

cells[15]["source"] = lines("""## 7. 최종 gate

데이터 BATCH는 manifest/ZIP을, 학습 BATCH는 지정된 eval 파일의 7렌즈·313 prompts·환경·
cap·suffix를 검사한다. 이 셀이 통과하지 않은 ZIP은 intake하지 않는다.
""")

cells[16]["source"] = lines(r'''import json, os

if BATCH == 0:
    manifest = json.load(open("r14_data/manifest.json", encoding="utf-8"))
    assert manifest["target_pairs_per_arm"] == 384
    assert manifest["rows_per_round"] == 128
    assert manifest["observed_budget_gap"] <= 0.01
    assert len(manifest["ordered_instruction_ids"]) == 384
    assert DATA_ZIP_OUT.is_file()
    print("[gate] R14 BATCH 0 PASS — frozen data candidate; manual review still required")
else:
    cells_expected = BATCHES[BATCH]
    for arm, seed in cells_expected:
        round_tag = 0 if arm == "arm0" else 3
        path = f"results/eval_{arm}_r{round_tag}_s{seed}{OUT_SUFFIX}.json"
        assert os.path.exists(path), path
        payload = json.load(open(path, encoding="utf-8"))
        assert payload["schema"] == "r2"
        assert payload["n_prompts"] == 313
        assert payload["max_new_tokens"] == 256
        assert payload["out_suffix"] == OUT_SUFFIX
        assert payload["env"].get("unsloth") == UNSLOTH_PIN
        assert payload["lenses"] == LENS_LIST
        assert list(payload["conditions"]) == LENS_LIST
        for lens in LENS_LIST:
            assert payload["conditions"][lens]["n"] == 313
            assert len(payload["conditions"][lens]["records"]) == 313
        print("[gate]", os.path.basename(path), "PASS")
    print(f"[gate] R14 BATCH {BATCH} PASS — {len(cells_expected)} cells")
''')

for cell in cells:
    cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs", [])
    if cell["cell_type"] == "code":
        cell["execution_count"] = None

OUTPUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"wrote {OUTPUT}")
