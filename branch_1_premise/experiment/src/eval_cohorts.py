"""eval 결과의 provenance 기반 cohort 발견과 선택."""
import json
import re
from pathlib import Path


_EVAL_FILENAME = re.compile(
    r"^eval_(?P<arm>[^_]+)_r(?P<round>[^_]+)_s(?P<seed>-?\d+)(?P<suffix>.*)\.json$"
)
_SAFE_SUFFIX = re.compile(r"^_[A-Za-z0-9][A-Za-z0-9_-]*$")


def _round_from_filename(tag):
    try:
        return float(tag.replace("p", "."))
    except ValueError as exc:
        raise ValueError(f"eval 파일명의 round를 해석할 수 없다: {tag!r}") from exc


def _cohort_label(suffix):
    return repr(suffix) if suffix else "'' (legacy)"


def _sorted_unique_json(values):
    """dict/list/None도 안정적으로 중복 제거해 JSON 값으로 돌려준다."""
    encoded = {
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in values
    }
    return [json.loads(value) for value in sorted(encoded)]


def _validate_entry(path, payload):
    if not isinstance(payload, dict):
        raise ValueError(f"eval payload가 객체가 아니다: {path.name}")

    try:
        arm = str(payload["arm"])
        round_no = float(payload["round"])
        seed = int(payload["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"eval provenance가 잘못됐다: {path.name}") from exc

    suffix = payload.get("out_suffix", "")
    if not isinstance(suffix, str):
        raise ValueError(f"out_suffix가 문자열이 아니다: {path.name}")
    if suffix and not _SAFE_SUFFIX.fullmatch(suffix):
        raise ValueError(f"안전하지 않은 out_suffix: {path.name} {suffix!r}")

    match = _EVAL_FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"eval 파일명 형식이 잘못됐다: {path.name}")
    filename_provenance = (
        match.group("arm"),
        _round_from_filename(match.group("round")),
        int(match.group("seed")),
        match.group("suffix"),
    )
    payload_provenance = (arm, round_no, seed, suffix)
    if filename_provenance != payload_provenance:
        raise ValueError(
            "eval 파일명/payload provenance 불일치: "
            f"{path.name} filename={filename_provenance!r} "
            f"payload={payload_provenance!r}"
        )

    return {
        "path": path,
        "payload": payload,
        "key": (arm, round_no, seed),
        "suffix": suffix,
    }


def discover_eval_cohorts(results_dir):
    """모든 eval JSON을 payload의 out_suffix로 묶고 cohort 내부 중복을 막는다."""
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob("eval_*.json"))
    if not paths:
        raise FileNotFoundError(f"eval_*.json이 없다: {results_dir}")

    cohorts = {}
    keys_by_cohort = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = _validate_entry(path, payload)
        suffix = entry["suffix"]
        cohort_keys = keys_by_cohort.setdefault(suffix, {})
        if entry["key"] in cohort_keys:
            other = cohort_keys[entry["key"]]
            raise ValueError(
                f"cohort {_cohort_label(suffix)} 내부 중복 eval 런: "
                f"{entry['key']} ({other.name}, {path.name})"
            )
        cohort_keys[entry["key"]] = path
        cohorts.setdefault(suffix, []).append(entry)
    return cohorts


def cohort_metadata(suffix, entries):
    return {
        "out_suffix": suffix,
        "file_count": len(entries),
        "files": [entry["path"].name for entry in entries],
        "arms": sorted({entry["key"][0] for entry in entries}),
        "rounds": sorted({entry["key"][1] for entry in entries}),
        "seeds": sorted({entry["key"][2] for entry in entries}),
        "max_new_tokens": _sorted_unique_json(
            entry["payload"].get("max_new_tokens") for entry in entries
        ),
        "env_stamps": _sorted_unique_json(
            entry["payload"].get("env") for entry in entries
        ),
    }


def select_eval_cohort(results_dir, suffix=""):
    cohorts = discover_eval_cohorts(results_dir)
    if suffix not in cohorts:
        available = ", ".join(
            _cohort_label(value) for value in sorted(cohorts)
        )
        raise ValueError(
            f"요청한 eval cohort가 없다: {suffix!r}; 사용 가능 cohort: {available}"
        )
    entries = cohorts[suffix]
    return entries, cohort_metadata(suffix, entries)


def format_cohort_listing(results_dir):
    cohorts = discover_eval_cohorts(results_dir)
    lines = []
    for suffix in sorted(cohorts):
        metadata = cohort_metadata(suffix, cohorts[suffix])
        lines.append(
            f"suffix={_cohort_label(suffix)} files={metadata['file_count']} "
            f"arms={json.dumps(metadata['arms'], ensure_ascii=False)} "
            f"rounds={json.dumps(metadata['rounds'], ensure_ascii=False)} "
            f"seeds={json.dumps(metadata['seeds'], ensure_ascii=False)} "
            f"max_new_tokens={json.dumps(metadata['max_new_tokens'], ensure_ascii=False)} "
            f"env_stamps={json.dumps(metadata['env_stamps'], ensure_ascii=False, sort_keys=True)}"
        )
    return "\n".join(lines)


def cohort_output_path(results_dir, stem, suffix, extension):
    """기본 cohort만 기존 이름을 쓰고 나머지는 out_suffix를 그대로 붙인다."""
    if suffix and not _SAFE_SUFFIX.fullmatch(suffix):
        raise ValueError(f"출력 파일명에 쓸 수 없는 out_suffix: {suffix!r}")
    return Path(results_dir) / f"{stem}{suffix}{extension}"


def markdown_cohort_lines(metadata):
    suffix = metadata["out_suffix"] or "(legacy empty suffix)"
    tokens = json.dumps(metadata["max_new_tokens"], ensure_ascii=False)
    envs = json.dumps(metadata["env_stamps"], ensure_ascii=False, sort_keys=True)
    return [
        "## Cohort provenance",
        "",
        f"- `out_suffix`: `{suffix}`",
        f"- files: `{metadata['file_count']}`",
        f"- `max_new_tokens`: `{tokens}`",
        f"- `env_stamps`: `{envs}`",
        "",
    ]
