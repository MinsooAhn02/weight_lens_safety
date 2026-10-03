"""EXP-M05: do "mid-sentence" endings at 512 tokens really end mid-sentence?

§5 of the camera-ready paper calls a 512-token response that the truncation heuristic
flags but that did not reach the cap a "mid-sentence" ending. Some of these may be
complete answers whose last line is a list item, code or a URL, which also lack terminal
punctuation. This script writes a blinded labelling sheet for every such response in the
compliant-lens cells the paper reports:

  * single-checkpoint run (r11_same_ckpt_512), arms 1 and 3  -> paper's 0.070 / 0.115
  * retrained 512 cohort (env76_max512), arm 2               -> paper's 0.010

Arm, cohort and heuristic category are withheld from the reader file and the TSV; they
live only in the key file, which the labeller should not open until labelling is done.

    python src/sample_m05.py          # writes annotation/m05_*.{md,tsv,jsonl}
"""
import hashlib
import json
import random
from pathlib import Path

from analyze_dual_primary import looks_truncated

EXP = Path(__file__).resolve().parents[1]
RES, ANN = EXP / "results", EXP / "annotation"
SOURCES = [
    ("arm1", "single_ckpt_512", "eval_arm1_r3_s42_r11_same_ckpt_512.json"),
    ("arm3", "single_ckpt_512", "eval_arm3_r3_s42_r11_same_ckpt_512.json"),
    ("arm2", "retrained_512", "eval_arm2_r3_s42_env76_max512.json"),
]
SEED = 20261001


def main():
    items = []
    for arm, cohort, name in SOURCES:
        records = json.loads((RES / name).read_text(encoding="utf-8"))["conditions"]["agreeable"]["records"]
        for idx, r in enumerate(records):
            if looks_truncated(r["response"]) and not r["hit_cap"]:
                iid = hashlib.sha256(f"{cohort}|{arm}|{idx}".encode()).hexdigest()[:10]
                items.append({"item_id": iid, "arm": arm, "cohort": cohort, "record_index": idx,
                              "finish_reason": r.get("finish_reason"), "n_new_tokens": r.get("n_new_tokens"),
                              "prompt": r["prompt"], "response": r["response"]})
    assert len(items) == 61, len(items)  # 22 + 36 + 3 (paper: 0.070, 0.115, 0.010 of 313)
    random.Random(SEED).shuffle(items)

    with open(ANN / "m05_key.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({k: it[k] for k in ("item_id", "arm", "cohort", "record_index",
                                                   "finish_reason", "n_new_tokens")}) + "\n")

    n = len(items)
    tsv = [
        f"# M05 mid-sentence ending check · n={n} · seed={SEED}",
        "# 읽기용 = m05_read.md (같은 순서)   정답키 = m05_key.jsonl (라벨링 끝날 때까지 열지 말 것)",
        "# end = E|L|O|X",
        "#   E = 문장/절이 중간에서 끊김 (예: '..., the' / 'If you're feeling unwell,')",
        "#   L = 완결된 답변. 마지막 줄이 목록 항목·제목 등이라 마침표가 없을 뿐 내용상 끝났다",
        "#   O = 코드·URL·표 등으로 끝나 판단 대상이 아님",
        "#   X = 판단 불가",
        "# conf = 1|2|3   note = 자유서술(마지막 열)",
        "item_id\tend\tconf\tnote",
    ] + [f"{it['item_id']}\t\t\t" for it in items]
    (ANN / "m05_labels.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")

    md = ["# M05 읽기용 — mid-sentence 판정 (61건)", "",
          "> `m05_labels.tsv`와 순서가 같다. 질문은 하나다: **이 응답은 문장 중간에서 끊겼는가(E), "
          "아니면 내용상 완결됐는가(L)?** 유해성·거부 여부는 판정하지 않는다.", ""]
    for i, it in enumerate(items, 1):
        md += [f"## {i} / {n} · `{it['item_id']}`", "", "**요청**", "",
               "> " + it["prompt"].replace("\n", "\n> "), "", "**응답 (전문)**", "", "```",
               it["response"].rstrip(), "```", "", "`end=_  conf=_`", "", "---", ""]
    (ANN / "m05_read.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {n} items")


if __name__ == "__main__":
    main()
