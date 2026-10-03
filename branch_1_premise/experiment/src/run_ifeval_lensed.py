"""Run the fixed benign IFEval slice once per evaluation lens.

This is intentionally separate from run_ifeval.sh and from the dual-primary
pipeline. lm-eval owns IFEval scoring and --log_samples preserves the
per-sample evidence; this wrapper only supplies the lens, enforces the frozen
run contract, and writes provenance.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from common import EXP_ROOT, RESULTS_DIR, env_versions, load_cfg
from eval_refusal import round_tag


BENIGN_LENSES = (
    "none",
    "agreeable_weak",
    "agreeable",
    "agreeable_strong",
    "principled",
)
HARMFUL_LENSES = (
    "none",
    "agreeable",
    "principled",
    "agreeable_para1",
    "agreeable_para2",
    "agreeable_weak",
    "agreeable_strong",
)
LIMIT = 50
SMOKE_LIMIT = 2
BATCH_SIZE = 2
OUTPUT_SUFFIX = "_r13_benign_env76"
SOURCE_LABEL = "r13_benign_ifeval_lens_conditioned"
CONFIG_REL = "configs/arms.json"


def lens_prompts(config_path=None):
    """Return the five prompts from the existing arms/eval-lenses config."""
    cfg = (
        json.loads(Path(config_path).read_text(encoding="utf-8"))
        if config_path
        else load_cfg("arms.json")
    )
    eval_lenses = cfg["eval_lenses"]
    arms = cfg["arms"]
    prompts = {
        "none": None,
        "agreeable_weak": eval_lenses["agreeable_weak"],
        "agreeable": arms["arm1"]["persona_prompt"],
        "agreeable_strong": eval_lenses["agreeable_strong"],
        "principled": arms["arm3"]["persona_prompt"],
    }
    assert tuple(prompts) == BENIGN_LENSES
    assert all(
        prompt is None or isinstance(prompt, str) and prompt
        for prompt in prompts.values()
    )
    return prompts


def _stem(arm, rnd, seed, lens, dry_run=False):
    dry = "_dryrun" if dry_run else ""
    return f"ifeval_{arm}_r{round_tag(rnd)}_s{seed}{OUTPUT_SUFFIX}_{lens}{dry}"


def output_dir(results_dir, arm, rnd, seed, lens, dry_run=False):
    results_dir = Path(results_dir)
    stem = _stem(arm, rnd, seed, lens, dry_run=dry_run)
    return results_dir / "_smoke" / stem if dry_run else results_dir / stem


def manifest_path(results_dir, arm, rnd, seed, dry_run=False):
    results_dir = Path(results_dir)
    stem = f"ifeval_manifest_{arm}_r{round_tag(rnd)}_s{seed}{OUTPUT_SUFFIX}"
    return (
        results_dir / "_smoke" / f"{stem}_dryrun.json"
        if dry_run
        else results_dir / f"{stem}.json"
    )


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value):
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def _relative(path):
    try:
        return path.resolve().relative_to(EXP_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _fingerprint_sha(payload):
    fingerprint = (
        payload.get("ckpt_fingerprint")
        or payload.get("checkpoint_fingerprint")
        or {}
    )
    return fingerprint.get("sha256")


def assert_same_checkpoint(harmful_manifest, benign_manifest):
    """Verify that the benign manifest faithfully records the harmful fingerprint."""
    harmful_sha = _fingerprint_sha(harmful_manifest)
    benign_sha = _fingerprint_sha(benign_manifest)
    assert harmful_sha and benign_sha, (
        "harmful/benign checkpoint fingerprint is missing"
    )
    assert harmful_sha == benign_sha, (
        "benign manifest did not preserve the harmful checkpoint fingerprint: "
        f"{harmful_sha[:16]}... != {benign_sha[:16]}..."
    )
    return harmful_sha


def build_command(model_path, system_prompt, out_path, limit=LIMIT):
    """Build the exact lm-eval command; keep it inspectable for the local gate."""
    command = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},load_in_4bit=True",
        "--tasks",
        "ifeval",
        "--limit",
        str(limit),
        "--batch_size",
        str(BATCH_SIZE),
        "--apply_chat_template",
    ]
    if system_prompt is not None:
        command.extend(["--system_instruction", system_prompt])
    command.extend(["--log_samples", "--output_path", str(out_path)])
    return command


def _sample_rows(payload):
    samples = payload.get("samples")
    if isinstance(samples, list):
        rows = samples
    elif isinstance(samples, dict):
        rows = samples.get("ifeval")
        if not isinstance(rows, list):
            rows = [
                row
                for value in samples.values()
                if isinstance(value, list)
                for row in value
            ]
    else:
        rows = None
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("lm_eval log has no usable --log_samples rows")
    if all("doc_id" in row for row in rows):
        rows = sorted(rows, key=lambda row: row["doc_id"])
    return rows


def _sample_prompt(row):
    candidates = [row.get("prompt")]
    doc = row.get("doc")
    if isinstance(doc, dict):
        candidates.extend([doc.get("prompt"), doc.get("instruction")])
    elif isinstance(doc, str):
        candidates.append(doc)
    arguments = row.get("arguments")
    if isinstance(arguments, list) and arguments:
        first = arguments[0]
        if isinstance(first, list) and first:
            first = first[0]
        candidates.append(first)
    for candidate in candidates:
        if isinstance(candidate, str):
            return candidate
    raise RuntimeError("IFEval sample log row has no prompt field")


def _jsonl_rows(path):
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{path}:{line_no}: invalid sample JSONL") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"{path}:{line_no}: sample JSONL row is not an object")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{path}: sample JSONL is empty")
    if all("doc_id" in row for row in rows):
        rows = sorted(rows, key=lambda row: row["doc_id"])
    return rows


def _prompt_hash_from_logs(results_path, samples_path, expected_n):
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = _jsonl_rows(samples_path) if samples_path else _sample_rows(payload)
    if len(rows) != expected_n:
        raise RuntimeError(
            f"{samples_path or results_path}: logged samples={len(rows)} expected={expected_n}"
        )
    prompts = [_sample_prompt(row) for row in rows]
    return _sha256_json(prompts), payload, samples_path or results_path


def _locate_logs(out_path):
    result_logs = sorted(out_path.rglob("results_*.json"))
    if len(result_logs) != 1:
        raise RuntimeError(
            f"{out_path}: expected exactly one lm_eval results_*.json, "
            f"found {len(result_logs)}"
        )
    sample_logs = sorted(out_path.rglob("samples_*.jsonl"))
    if len(sample_logs) > 1:
        raise RuntimeError(
            f"{out_path}: expected at most one lm_eval samples_*.jsonl, "
            f"found {len(sample_logs)}"
        )
    return result_logs[0], sample_logs[0] if sample_logs else None


def _adapter_paths(adapter_root, arm, seed):
    if not adapter_root:
        return []
    root = Path(adapter_root)
    if not root.exists():
        return []
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.glob(f"{arm}_r*_s{seed}"))
        if path.is_dir()
    ]


def _read_harmful_result(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data.get("schema") == "r2", f"{path}: harmful result is not schema r2"
    assert data.get("n_prompts") == 313, f"{path}: harmful n_prompts is not 313"
    assert data.get("lenses") == list(HARMFUL_LENSES), (
        f"{path}: harmful lens grid is not the frozen seven-lens R12 grid"
    )
    assert data.get("out_suffix") == OUTPUT_SUFFIX, (
        f"{path}: harmful result suffix is not {OUTPUT_SUFFIX}"
    )
    fingerprint = data.get("ckpt_fingerprint") or {}
    assert fingerprint.get("sha256"), f"{path}: harmful checkpoint fingerprint missing"
    return data, fingerprint


def run(args):
    results_dir = Path(args.results_dir).resolve() if args.results_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    prompts = lens_prompts(args.config)
    expected_n = SMOKE_LIMIT if args.dry_run else LIMIT

    harmful = None
    fingerprint = None
    harmful_path = None
    if not args.dry_run:
        harmful_path = (
            Path(args.harmful_result)
            if args.harmful_result
            else results_dir
            / f"eval_{args.arm}_r{round_tag(args.round)}_s{args.seed}{OUTPUT_SUFFIX}.json"
        )
        harmful, fingerprint = _read_harmful_result(harmful_path)
        assert harmful.get("model_path") == args.model_path, (
            "generated notebook must pass the same retained model_path to harmful and "
            f"benign evaluation: {harmful.get('model_path')} != {args.model_path}"
        )

    records = []
    prompt_hash = None
    for lens in BENIGN_LENSES:
        out_path = output_dir(
            results_dir, args.arm, args.round, args.seed, lens, dry_run=args.dry_run
        )
        if out_path.exists() and any(out_path.iterdir()):
            raise RuntimeError(f"refusing to reuse non-empty output path: {out_path}")
        out_path.mkdir(parents=True, exist_ok=True)
        command = build_command(
            args.model_path,
            prompts[lens],
            out_path,
            limit=expected_n,
        )
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            raise RuntimeError(f"lm_eval failed ({completed.returncode}): {command}")
        results_path, samples_path = _locate_logs(out_path)
        current_hash, payload, sample_log_path = _prompt_hash_from_logs(
            results_path, samples_path, expected_n
        )
        if prompt_hash is None:
            prompt_hash = current_hash
        else:
            assert current_hash == prompt_hash, (
                f"{lens}: prompt hash differs from the first lens; fixed first-{expected_n} "
                "IFEval slice was not reused"
            )
        records.append(
            {
                "lens": lens,
                "system_prompt_hash": _sha256_bytes(
                    (prompts[lens] or "").encode("utf-8")
                ),
                "prompt_hash": current_hash,
                "sample_count": expected_n,
                "output_dir": _relative(out_path),
                "results_log": _relative(results_path),
                "sample_log": _relative(sample_log_path),
                "sample_log_format": "jsonl" if samples_path else "embedded_results_json",
                "metrics": payload.get("results", {}).get("ifeval", {}),
            }
        )

    config_path = Path(args.config) if args.config else EXP_ROOT / CONFIG_REL
    config_path = config_path.resolve()
    adapter_paths = _adapter_paths(args.adapter_root, args.arm, args.seed)
    if not args.dry_run and args.arm != "arm0":
        assert adapter_paths, (
            f"{args.arm}: no saved adapter directory under {args.adapter_root}; "
            "training must use the generated notebook save_adapters path"
        )
    fingerprint = fingerprint or {}
    manifest = {
        "schema": "r13_benign_ifeval_manifest_v1",
        "source": "lm_eval",
        "source_label": SOURCE_LABEL,
        "task": "ifeval",
        "arm": args.arm,
        "round": args.round,
        "seed": args.seed,
        "model_path": args.model_path,
        "out_suffix": OUTPUT_SUFFIX,
        "lenses": list(BENIGN_LENSES),
        "system_prompts": prompts,
        "lens_hashes": {
            lens: _sha256_bytes((prompts[lens] or "").encode("utf-8"))
            for lens in BENIGN_LENSES
        },
        "prompt_hash": prompt_hash,
        "prompt_hashes": {record["lens"]: record["prompt_hash"] for record in records},
        "n_prompts": expected_n,
        "limit": expected_n,
        "batch_size": BATCH_SIZE,
        "num_fewshot": 0,
        "zero_shot": True,
        "log_samples": True,
        "load_in_4bit": True,
        "env": env_versions(),
        "checkpoint_fingerprint": fingerprint,
        "ckpt_fingerprint": fingerprint,
        "checkpoint_fingerprint_source": (
            {
                "kind": "copied_from_harmful_result",
                "path": _relative(harmful_path),
                "field": "ckpt_fingerprint",
            }
            if harmful_path
            else None
        ),
        "checkpoint_fingerprint_independently_recomputed": False,
        "same_model_path_before_cleanup": (
            harmful.get("model_path") == args.model_path if harmful else None
        ),
        "same_model_path_evidence": (
            {
                "harmful_result_model_path": harmful.get("model_path"),
                "benign_runner_model_path": args.model_path,
                "process_guarantee": (
                    "The generated notebook retains this model_path through harmful and "
                    "benign evaluation and deletes it only after this runner returns."
                ),
            }
            if harmful
            else None
        ),
        "harmful_result": _relative(harmful_path) if harmful_path else None,
        "harmful_lenses": list(HARMFUL_LENSES),
        "config": _relative(config_path),
        "config_sha256": _sha256_bytes(config_path.read_bytes()),
        "adapter_root": str(args.adapter_root) if args.adapter_root else None,
        "adapter_paths": adapter_paths,
        "sample_logs": records,
        "smoke": bool(args.dry_run),
    }
    path = manifest_path(
        results_dir, args.arm, args.round, args.seed, dry_run=args.dry_run
    )
    if path.exists():
        raise RuntimeError(f"refusing to overwrite manifest: {path}")
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ifeval-r13] -> {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--round", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--harmful_result")
    parser.add_argument("--adapter_root")
    parser.add_argument("--results_dir")
    parser.add_argument("--config")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
