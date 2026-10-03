import hashlib
import json
import sys
import tempfile
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT / "src"
sys.path.insert(0, str(SRC))

import make_nb_r13_benign as generator  # noqa: E402
from run_ifeval_lensed import (  # noqa: E402
    BATCH_SIZE,
    BENIGN_LENSES,
    LIMIT,
    OUTPUT_SUFFIX,
    _locate_logs,
    _prompt_hash_from_logs,
    assert_same_checkpoint,
    build_command,
    output_dir,
)


def main():
    assert BENIGN_LENSES == (
        "none",
        "agreeable_weak",
        "agreeable",
        "agreeable_strong",
        "principled",
    )
    assert LIMIT == 50
    assert BATCH_SIZE == 2
    command = build_command("ckpt", "prompt", Path("out"))
    joined = " ".join(command)
    assert "load_in_4bit=True" in joined
    assert "--limit 50" in joined
    assert "--batch_size 2" in joined
    assert "--log_samples" in joined
    assert "--num_fewshot" not in joined

    paths = {
        output_dir(EXPERIMENT / "results", "arm1", 3, 42, lens)
        for lens in BENIGN_LENSES
    }
    assert len(paths) == 5
    assert all(OUTPUT_SUFFIX in str(path) for path in paths)
    assert (
        output_dir(EXPERIMENT / "results", "arm1", 3, 42, "none", dry_run=True).parts[-2]
        == "_smoke"
    )
    assert {1: generator.JOB_MAP[1], 2: generator.JOB_MAP[2], 3: generator.JOB_MAP[3]} == {
        1: {"seed": 42, "arms": ["arm0", "arm1"], "label": "seed42-arm0-arm1"},
        2: {"seed": 42, "arms": ["arm2"], "label": "seed42-arm2"},
        3: {"seed": 42, "arms": ["arm3"], "label": "seed42-arm3"},
    }
    assert [(job["seed"], job["arms"]) for key, job in generator.JOB_MAP.items() if key >= 4] == [
        (1337, ["arm1"]),
        (1337, ["arm2"]),
        (1337, ["arm3"]),
        (2718, ["arm1"]),
        (2718, ["arm2"]),
        (2718, ["arm3"]),
    ]

    harmful = {"ckpt_fingerprint": {"sha256": "same"}}
    benign = {"checkpoint_fingerprint": {"sha256": "same"}}
    assert assert_same_checkpoint(harmful, benign) == "same"
    try:
        assert_same_checkpoint(
            harmful, {"ckpt_fingerprint": {"sha256": "different"}}
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("different checkpoint fingerprints were accepted")

    with tempfile.TemporaryDirectory(prefix="r13-ifeval-") as td:
        log_dir = Path(td)
        results_path = log_dir / "results_2026-08-12.json"
        samples_path = log_dir / "samples_ifeval_2026-08-12.jsonl"
        results_path.write_text(
            json.dumps({"results": {"ifeval": {"strict-match": 1.0}}}),
            encoding="utf-8",
        )
        prompts = [f"fixed prompt {index}" for index in range(50)]
        samples_path.write_text(
            "\n".join(
                json.dumps({"doc_id": index, "doc": {"prompt": prompt}})
                for index, prompt in enumerate(prompts)
            )
            + "\n",
            encoding="utf-8",
        )
        located_results, located_samples = _locate_logs(log_dir)
        assert located_results == results_path
        assert located_samples == samples_path
        prompt_hash, metrics_payload, true_sample_log = _prompt_hash_from_logs(
            located_results, located_samples, 50
        )
        expected_hash = hashlib.sha256(
            json.dumps(
                prompts, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        assert prompt_hash == expected_hash
        assert metrics_payload["results"]["ifeval"]["strict-match"] == 1.0
        assert true_sample_log == samples_path

    notebook = generator.build_notebook()
    assert len(notebook["cells"]) == 19
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "run_ifeval_lensed.py" in source
    assert "_r13_benign_env76" in source
    assert "assert_same_checkpoint" in source
    assert "checkpoint_fingerprint_source" in source
    assert "not independently recomputed" in source
    assert 'pip("langdetect==1.0.9")' in source
    assert "(또는 P100)" not in source
    generated = EXPERIMENT / "nb_r13_benign.ipynb"
    assert generated.exists()
    assert json.loads(generated.read_text(encoding="utf-8"))["cells"] == notebook["cells"]
    print("test_r13_benign: OK")


if __name__ == "__main__":
    main()
