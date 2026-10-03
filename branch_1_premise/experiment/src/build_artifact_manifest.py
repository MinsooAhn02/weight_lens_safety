"""Reproducibility artifact manifest builder/checker.

Produces a single JSON manifest (``branch_1_premise/experiment/results/artifact_manifest.json``)
that records the sha256 + byte size of every artifact that feeds the Mirror Trap paper's
claims: configs, source, notebooks, the training-data backup, canonical eval/probe/MMLU/IFEval
result JSONs, R13 IFEval result/sample logs and aggregate manifests, the M03/M04
human-annotation sets, and submission documentation. ``--check`` re-hashes everything
and reports drift or an unrecorded artifact.

MMLU and IFEval both live outside the dual-primary analysis pipeline (they are produced by
``src/run_mmlu.sh`` / ``src/run_ifeval.sh`` calling ``lm_eval``, not by
``analyze_dual_primary.py``), so they enter the manifest only through the
``{mmlu,ifeval}_*/*/results_*.json`` globs below, under roles ``mmlu`` and ``ifeval``.
R13's compact aggregate-only records use the top-level ``ifeval_manifest_*.json`` glob.

Design constraints (see task spec for the full rationale):
  * Python 3.9.13, standard library only, must run on Windows.
  * ``--write`` run twice must be byte-identical: no timestamps, no absolute paths, no
    machine info anywhere in the output. Key order is fixed by construction
    (``sort_keys=False``); only the ``artifacts`` list is explicitly sorted, by ``path``.
  * All recorded paths are repo-root-relative POSIX paths (forward slashes), even though
    this runs on Windows.

Mutable-artifact rationale: ``m03_labels_pass2.tsv`` (and, once they exist,
``m03_labels_pass3.tsv`` / ``m03_items_pass3.jsonl`` / ``m03_read_pass3.md``) are pass-2/3
annotation files that are deliberately unfilled right now and WILL change as labelling
resumes. Flagging them ``mutable: true`` lets ``--check`` report drift on those specific
paths as an informational note instead of a hard failure -- the rest of the manifest still
fails closed on any unexpected byte change.

``run_archives`` (``branch_1_premise/runs/**/*.zip`` and ``*.log``) are provenance, not
canonical results -- Kaggle session logs and output zips kept for audit trail. Git-tracked
archives are listed (path/kind/run_id/purpose) but deliberately NOT hashed because they
are large; local-only files are not claimed by the committed manifest.
The paper's aggregate evidence under ``experiment/results/`` is hashed; publishing or
hashing the preserved R13 raw IFEval logs remains explicit pending work.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA = "mirror-trap-artifacts.v1"
MANIFEST_REL = "branch_1_premise/experiment/results/artifact_manifest.json"

# Explicit mutable-artifact set (see module docstring). Pass-3 files are optional --
# they simply don't exist yet, and are skipped like any other missing explicit artifact.
MUTABLE_RELS = {
    "branch_1_premise/experiment/annotation/m03_labels_pass2.tsv",
    "branch_1_premise/experiment/annotation/m03_labels_pass3.tsv",
    "branch_1_premise/experiment/results/m03_items_pass3.jsonl",
    "branch_1_premise/experiment/annotation/m03_read_pass3.md",
}

CONFIG_FILES = [
    "branch_1_premise/experiment/configs/arms.json",
    "branch_1_premise/experiment/configs/train_config.json",
]

NOTEBOOK_FILES = [
    "branch_1_premise/experiment/nb_r4.ipynb",
    "branch_1_premise/experiment/nb_seeds.ipynb",
    "branch_1_premise/experiment/nb_r14.ipynb",
]

DATA_FILES = [
    "branch_1_premise/data/gen_data_backup.zip",
    "branch_1_premise/runs/r14/r14_matched_data.zip",
]

M03_FILES = [
    "branch_1_premise/experiment/annotation/M03_CODEBOOK.md",
    "branch_1_premise/experiment/annotation/M03_ANALYSIS_PROTOCOL.md",
    "branch_1_premise/experiment/annotation/m03_labels_pass1.tsv",
    "branch_1_premise/experiment/annotation/m03_labels_pass2.tsv",
    "branch_1_premise/experiment/annotation/m03_labels_pass3.tsv",
    "branch_1_premise/experiment/annotation/m03_read_pass1.md",
    "branch_1_premise/experiment/annotation/m03_read_pass2.md",
    "branch_1_premise/experiment/annotation/m03_read_pass3.md",
    "branch_1_premise/experiment/results/m03_items_pass1.jsonl",
    "branch_1_premise/experiment/results/m03_items_pass2.jsonl",
    "branch_1_premise/experiment/results/m03_items_pass3.jsonl",
    "branch_1_premise/experiment/results/m03_sample_manifest.json",
]

M04_FILES = [
    "branch_1_premise/experiment/annotation/m04_control_items.jsonl",
    "branch_1_premise/experiment/annotation/m04_labels_pass1.tsv",
    "branch_1_premise/experiment/annotation/m04_read_pass1.md",
    "paper/flmsec_reframe_work/m04_machine_labels.jsonl",
    "paper/flmsec_reframe_work/m04_scoring_report.md",
]

DOCUMENTATION_FILES = [
    "branch_1_premise/r2_summary.md",
    "branch_1_premise/R3_STATUS.md",
    "branch_1_premise/results_summary.md",
    "paper/full/main.tex",
    "paper/full/main.pdf",
    "branch_1_premise/experiment/R14_RUNBOOK.md",
    "paper/shared/NUMBERS.md",
    "paper/iclr/main.tex",
    "paper/iclr/main.pdf",
    "paper/iclr/ai_statement.tex",
    "paper/iclr/README.md",
]

CANONICAL_SOURCES = {
    "pass1_completed_at": "2026-08-01T05:23:46+09:00",
    "pass1_completed_at_source": "mtime of branch_1_premise/experiment/annotation/m03_labels_pass1.tsv",
    "pass2_earliest": "2026-08-08T05:23:46+09:00",
    "m03_pairs": 146,
    "m03_rows": 292,
}

PENDING_WORK = [
    "reconcile Gate 4: 긴_논문_개선해야할점.md:173-178 needs model family AND update method, RESULTS_INTAKE.md:145 says R7 alone opens it (R7 gave only the model axis and failed to replicate)",
    "EXP-M02 benign training-data audit (1235 rows in data/gen_data_backup.zip) — not started, schema must be written first (R3_STATUS.md:462)",
]

_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}_")


def default_repo_root() -> Path:
    # build_artifact_manifest.py lives at <root>/branch_1_premise/experiment/src/
    return Path(__file__).resolve().parents[3]


def sha256_and_size(path: Path) -> Tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


def to_posix_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def get_git_rev(root: Path) -> Optional[str]:
    git = ["git", "-c", f"safe.directory={root.resolve()}"]
    try:
        proc = subprocess.run(
            [*git, "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    rev = proc.stdout.strip()
    if not rev:
        return None
    try:
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=no"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{rev}-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else rev


def compute_nb_revision(root: Path) -> Optional[str]:
    path = root / "branch_1_premise" / "experiment" / "nb_r4.ipynb"
    if not path.is_file():
        return None
    sha, _ = sha256_and_size(path)
    return sha[:12]


def make_entry(root: Path, rel: str, role: str) -> Optional[Dict]:
    path = root / rel
    if not path.is_file():
        return None
    sha, size = sha256_and_size(path)
    entry = {
        "path": rel,
        "role": role,
        "bytes": size,
        "sha256": sha,
    }
    if rel in MUTABLE_RELS:
        entry["mutable"] = True
    return entry


def read_env_field(root: Path, rel: str) -> Optional[Dict]:
    try:
        data = json.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    env = data.get("env")
    return env if isinstance(env, dict) else None


def collect_artifacts(root: Path) -> Tuple[List[Dict], Dict]:
    """Returns (artifacts, stats) where stats carries counts needed for
    known_noncomparabilities / environments (computed fresh from what's on disk).
    """
    artifacts: List[Dict] = []

    def add_all(rels: List[str], role: str):
        for rel in rels:
            entry = make_entry(root, rel, role)
            if entry is not None:
                artifacts.append(entry)

    add_all(CONFIG_FILES, "config")

    src_dir = root / "branch_1_premise/experiment/src"
    if src_dir.is_dir():
        for path in sorted(src_dir.glob("*.py")):
            rel = to_posix_rel(path, root)
            entry = make_entry(root, rel, "source")
            if entry is not None:
                artifacts.append(entry)

    add_all(NOTEBOOK_FILES, "notebook")
    add_all(DATA_FILES, "data")

    results_dir = root / "branch_1_premise/experiment/results"
    eval_rels, probe_rels = [], []
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("eval_*.json")):
            eval_rels.append(to_posix_rel(path, root))
        for path in sorted(results_dir.glob("probe_*.json")):
            probe_rels.append(to_posix_rel(path, root))
        for path in sorted(results_dir.glob("mmlu_*/*/results_*.json")):
            rel = to_posix_rel(path, root)
            entry = make_entry(root, rel, "mmlu")
            if entry is not None:
                artifacts.append(entry)
        for path in sorted(results_dir.glob("ifeval_*/*/results_*.json")):
            rel = to_posix_rel(path, root)
            entry = make_entry(root, rel, "ifeval")
            if entry is not None:
                artifacts.append(entry)
        for path in sorted(results_dir.glob("ifeval_*/*/samples_*.jsonl")):
            rel = to_posix_rel(path, root)
            entry = make_entry(root, rel, "ifeval_sample")
            if entry is not None:
                artifacts.append(entry)
        for path in sorted(results_dir.glob("ifeval_manifest_*.json")):
            rel = to_posix_rel(path, root)
            entry = make_entry(root, rel, "ifeval_manifest")
            if entry is not None:
                artifacts.append(entry)
    add_all(eval_rels, "eval")
    add_all(probe_rels, "probe")

    add_all(M03_FILES, "m03")
    add_all(M04_FILES, "m04")
    add_all(DOCUMENTATION_FILES, "documentation")

    # The manifest must never list itself.
    artifacts = [a for a in artifacts if a["path"] != MANIFEST_REL]

    # environments: only eval_*/probe_* canonical result files sometimes carry `env`.
    env_groups: Dict[str, Dict] = {}
    eval_with_env = 0
    probe_with_env = 0
    for rel in eval_rels:
        env = read_env_field(root, rel)
        if env is not None:
            eval_with_env += 1
            key = json.dumps(env, sort_keys=True, ensure_ascii=False)
            env_groups.setdefault(key, {"env": env, "files": []})["files"].append(rel)
    for rel in probe_rels:
        env = read_env_field(root, rel)
        if env is not None:
            probe_with_env += 1
            key = json.dumps(env, sort_keys=True, ensure_ascii=False)
            env_groups.setdefault(key, {"env": env, "files": []})["files"].append(rel)

    environments = [
        {"env": v["env"], "files": sorted(v["files"])}
        for _, v in sorted(env_groups.items())
    ]

    stats = {
        "eval_total": len(eval_rels),
        "eval_with_env": eval_with_env,
        "eval_without_env": len(eval_rels) - eval_with_env,
        "probe_total": len(probe_rels),
        "probe_with_env": probe_with_env,
        "probe_without_env": len(probe_rels) - probe_with_env,
        "environments": environments,
    }
    return artifacts, stats


def derive_run_archive_purpose(run_id: str, stem: str, kind: str) -> str:
    label = _DATE_PREFIX.sub("", stem)
    label = label.replace("-", " ").replace("_", " ").strip()
    noun = "console log" if kind == "log" else "run output archive (zip)"
    if not label:
        return f"{run_id} {noun}"
    return f"{run_id} {label} {noun}"


def collect_run_archives(root: Path) -> List[Dict]:
    runs_dir = root / "branch_1_premise/runs"
    if not runs_dir.is_dir():
        return []
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--", "branch_1_premise/runs"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    tracked = (
        {p for p in proc.stdout.decode("utf-8").split("\0") if p}
        if proc.returncode == 0
        else None
    )
    found = []
    for pattern, kind in (("*.zip", "archive"), ("*.log", "log")):
        for path in runs_dir.rglob(pattern):
            if not path.is_file():
                continue
            rel = to_posix_rel(path, root)
            if tracked is not None and rel not in tracked:
                continue
            run_id = path.parent.name
            purpose = derive_run_archive_purpose(run_id, path.stem, kind)
            found.append({
                "path": rel,
                "kind": kind,
                "run_id": run_id,
                "purpose": purpose,
                "canonical": False,
            })
    found.sort(key=lambda e: e["path"])
    return found


def build_known_noncomparabilities(stats: Dict) -> List[Dict]:
    return [
        {
            "item": (
                "seed/environment confound in the three-seed spread "
                "-- legacy cohort only; closed by the _env76 cohort"
            ),
            "detail": (
                "In the legacy (no-suffix) cohort, seed 42 was trained under unsloth "
                "2026.7.6 while seeds 1337/2718 were trained under 2026.7.5, so its "
                "three-seed spread mixes seed variance and environment variance. R5 "
                "batches 2-6 re-ran the whole 4-arm x 3-seed grid in one environment; "
                "the _env76 cohort stamps unsloth 2026.7.6 on all twelve runs and does "
                "not carry this confound. Use _env76 for any three-seed claim. This is "
                "a prose claim from the R3 status log plus the env field on each result "
                "JSON; the environment stamps are machine-checkable, the legacy "
                "attribution is not."
            ),
            "source": "branch_1_premise/R3_STATUS.md (see the R5 section)",
        },
        {
            "item": "R2 round checkpoints vs R4 retrain checkpoints",
            "detail": (
                "R2's within-run round checkpoints (round 0->1->2->3 inside one training "
                "session) and R4's retrain checkpoints (seed 42 retrained from scratch "
                "under a different unsloth version) must not be joined into one "
                "dose-response curve -- they are different checkpoint lineages."
            ),
            "source": "branch_1_premise/runs/README.md (r4/ generation notes)",
        },
        {
            "item": "arm0 probe is effectively n=1",
            "detail": (
                "The arm0 probe files for seeds 42/1337/2718 carry identical "
                "kappa_v_ref/kappa_v_harm values and empty per-lens readings ({}) -- "
                "arm0 (no training) was probed once and the same numbers were copied "
                "across the three seed files, so it is not an independent 3-seed sample."
            ),
            "source": (
                "branch_1_premise/experiment/results/probe_arm0_r0_s42.json, "
                "probe_arm0_r0_s1337.json, probe_arm0_r0_s2718.json"
            ),
        },
        {
            "item": "legacy probe_r2-schema files excluded from canonical probe analysis",
            "detail": (
                "probe_arm1_r0p25_s42.json, probe_arm1_r0p5_s42.json, "
                "probe_arm3_r0p25_s42.json, probe_arm3_r0p5_s42.json use "
                "schema: probe_r2, lack a harm_frame field, and report "
                "kappa_v_harm >= 1.0 (one value is 1.0000001192092896, a "
                "floating-point artefact, not a real ceiling effect)."
            ),
            "source": (
                "branch_1_premise/experiment/results/probe_arm1_r0p25_s42.json (and the "
                "three sibling legacy probe_r2 files listed in `item`)"
            ),
        },
        {
            "item": "mmlu_arm1_r3 has two non-comparable result files",
            "detail": (
                "mmlu_arm1_r3/.../results_*.json contains two files: the legacy "
                "limit=30 run (1710 items, acc 0.6661, dated 2026-07-28) and the "
                "limit=10 run (570 items, dated 2026-07-30) used everywhere else. "
                "The limit=30 file is legacy and not comparable to the limit=10 runs."
            ),
            "source": "branch_1_premise/experiment/results/mmlu_arm1_r3/*/results_*.json",
        },
        {
            "item": "most eval/probe result JSONs have no env stamp",
            "detail": (
                "The `env` field (unsloth/torch/transformers versions) was added late; "
                "most result JSONs predate it. Counts computed by this run: "
                "{eval_with_env}/{eval_total} eval files and "
                "{probe_with_env}/{probe_total} probe files carry an `env` object; "
                "the remaining {eval_without_env} eval and {probe_without_env} probe "
                "files have none, and that is the normal case, not an error."
            ).format(**stats),
            "source": "computed from env_ field presence across branch_1_premise/experiment/results/{eval,probe}_*.json",
        },
        {
            "item": "principled lens only run at seed 42 in the legacy cohort",
            "detail": (
                "In the legacy no-suffix eval files, the `principled` lens appears "
                "only at seed 42; seeds 1337 and 2718 stop at "
                "none/agreeable/agreeable_para1. The frozen `_env76` cohort closes "
                "this gap and includes the principled lens at all three seeds."
            ),
            "source": (
                "branch_1_premise/experiment/results/eval_arm1_r3_s42.json (has "
                "'principled') vs eval_arm1_r3_s1337.json / eval_arm1_r3_s2718.json "
                "(do not)"
            ),
        },
    ]


def build_manifest(root: Path) -> Dict:
    artifacts, stats = collect_artifacts(root)
    artifacts.sort(key=lambda e: e["path"])
    manifest = {
        "schema": SCHEMA,
        "source_commit": get_git_rev(root),
        "nb_revision": compute_nb_revision(root),
        "canonical_sources": dict(CANONICAL_SOURCES),
        "environments": stats["environments"],
        "artifacts": artifacts,
        "run_archives": collect_run_archives(root),
        "known_noncomparabilities": build_known_noncomparabilities(stats),
        "pending_work": list(PENDING_WORK),
    }
    return manifest


def cmd_write(root: Path) -> int:
    manifest = build_manifest(root)
    out_path = root / MANIFEST_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    # Python 3.9's Path.write_text() has no `newline=` kwarg; open explicitly so the
    # output is byte-identical across platforms (no CRLF translation on Windows).
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    total_bytes = sum(a["bytes"] for a in manifest["artifacts"])
    print(
        f"wrote {out_path}: "
        f"{len(manifest['artifacts'])} artifacts, {total_bytes} bytes covered"
    )
    return 0


def cmd_check(root: Path) -> int:
    manifest_path = root / MANIFEST_REL
    if not manifest_path.is_file():
        print(f"CHECK FAILED: manifest not found at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: List[str] = []
    notes: List[str] = []

    recorded = {entry["path"] for entry in manifest.get("artifacts", [])}
    discovered = {entry["path"] for entry in collect_artifacts(root)[0]}
    for rel in sorted(discovered - recorded):
        failures.append(f"UNRECORDED {rel}")

    for entry in manifest.get("artifacts", []):
        rel = entry["path"]
        path = root / rel
        mutable = bool(entry.get("mutable"))
        if not path.is_file():
            msg = f"MISSING {rel}"
            (notes if mutable else failures).append(
                ("NOTE (mutable) " if mutable else "") + msg
            )
            continue
        sha, size = sha256_and_size(path)
        problems = []
        if size != entry.get("bytes"):
            problems.append(f"size manifest={entry.get('bytes')} actual={size}")
        if sha != entry.get("sha256"):
            problems.append(f"sha256 manifest={entry.get('sha256')} actual={sha}")
        if problems:
            msg = f"{rel}: " + "; ".join(problems)
            if mutable:
                notes.append(f"NOTE (mutable) {msg}")
            else:
                failures.append(msg)

    for note in notes:
        print(note)

    if failures:
        print(f"CHECK FAILED: {len(failures)} discrepancy(ies)")
        for f in failures:
            print(f" - {f}")
        return 1

    print(f"CHECK OK: {len(manifest.get('artifacts', []))} artifacts verified"
          + (f", {len(notes)} mutable note(s)" if notes else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or check the reproducibility artifact manifest.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="compute and write the manifest")
    group.add_argument("--check", action="store_true", help="verify artifacts against the manifest")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="repo root (default: derived from this file's location)",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else default_repo_root()
    if args.write:
        return cmd_write(root)
    return cmd_check(root)


if __name__ == "__main__":
    sys.exit(main())
