"""R17 style-swap data: rewrite an arm's training responses in another style, same content.

Preregistration: R15_R17_CAMERA_READY.md (R17). The instruction column is copied
unchanged; only the response is rewritten by the base model, greedy, so the new arm
keeps the source arm's content and takes the target style.

    python src/rewrite_style.py --src arm1 --dst arm8 --style neutral
    python src/rewrite_style.py --src arm2 --dst arm9 --style warm
    python src/rewrite_style.py --selftest          # no GPU; checks the quality gate

Writes data/<dst>/round{1,2,3}.jsonl and results/rewrite_<dst>_report.json. Exits non-zero
(and writes no training data) if the preregistered quality gate fails.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
BASE = "unsloth/Meta-Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 384
GATE_LO, GATE_HI = 0.5, 2.0  # median length ratio (prereg R17)

COMMON_RULES = (
    "Keep every piece of information, step, number, example and recommendation in the "
    "original. Do not add or remove content, do not add warnings, disclaimers, greetings or "
    "opinions, and do not answer the request yourself. Output only the rewritten reply."
)
STYLES = {
    "neutral": ("You are a copy editor. Rewrite the assistant reply below in a neutral, "
                "matter-of-fact, impersonal tone. " + COMMON_RULES),
    "warm": ("You are a copy editor. Rewrite the assistant reply below in a warm, "
             "enthusiastic, eager-to-please tone that is friendly toward the user. " + COMMON_RULES),
}
REFUSAL = re.compile(r"\bi can(?:'|no)t\b|\bi won'?t\b|\bi'?m (?:not able|unable) to\b", re.I)


def user_message(instruction, response):
    return f"Original request:\n{instruction}\n\nReply to rewrite:\n{response}"


def quality(src_rows, new_rows):
    ratios = [len(n["response"]) / max(1, len(s["response"])) for s, n in zip(src_rows, new_rows)]
    rep = {
        "n": len(new_rows),
        "median_length_ratio": statistics.median(ratios) if ratios else None,
        "empty": sum(1 for n in new_rows if not n["response"].strip()),
        "refusal_pattern_rate_src": sum(bool(REFUSAL.search(s["response"])) for s in src_rows) / max(1, len(src_rows)),
        "refusal_pattern_rate_new": sum(bool(REFUSAL.search(n["response"])) for n in new_rows) / max(1, len(new_rows)),
    }
    m = rep["median_length_ratio"]
    rep["gate_pass"] = bool(m is not None and GATE_LO <= m <= GATE_HI and rep["empty"] == 0)
    return rep


def selftest():
    src = [{"instruction": "q", "response": "abcdefghij"}] * 4
    ok = [{"instruction": "q", "response": "abcdefgh"}] * 4
    short = [{"instruction": "q", "response": "ab"}] * 4
    empty = ok[:3] + [{"instruction": "q", "response": "  "}]
    assert quality(src, ok)["gate_pass"]
    assert not quality(src, short)["gate_pass"], "gate must fail when content shrinks"
    assert not quality(src, empty)["gate_pass"], "gate must fail on an empty rewrite"
    assert set(STYLES) == {"neutral", "warm"}
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--style", choices=sorted(STYLES))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    assert a.src and a.dst and a.style, "--src, --dst and --style are required"
    assert a.dst not in {"arm0", "arm1", "arm2", "arm3", "arm4", "arm5", "arm6", "arm7"}, \
        f"refusing to overwrite an existing arm's data: {a.dst}"

    sys.path.insert(0, str(EXP / "src"))
    from common import batch_chat, load_unsloth_model

    model, tok = load_unsloth_model(BASE, max_seq_length=2048, for_inference=True)
    all_src, all_new = [], []
    out_dir = EXP / "data" / a.dst
    pending = {}
    for r in (1, 2, 3):
        path = EXP / "data" / a.src / f"round{r}.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert rows, f"no rows in {path}"
        outs = batch_chat(model, tok, [STYLES[a.style]] * len(rows),
                          [user_message(x["instruction"], x["response"]) for x in rows],
                          max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, batch_size=a.batch_size)
        new = [{"instruction": x["instruction"], "response": o.strip(), "arm": a.dst,
                "source_arm": a.src, "style": a.style} for x, o in zip(rows, outs)]
        pending[r] = new
        all_src += rows; all_new += new
        print(f"[rewrite] {a.src}->{a.dst} round{r}: {len(new)} rows")

    rep = quality(all_src, all_new)
    rep.update({"src": a.src, "dst": a.dst, "style": a.style, "base": BASE,
                "max_new_tokens": MAX_NEW_TOKENS, "gate": [GATE_LO, GATE_HI],
                "examples": [{"before": s["response"][:400], "after": n["response"][:400]}
                             for s, n in list(zip(all_src, all_new))[:5]]})
    (EXP / "results").mkdir(exist_ok=True)
    (EXP / "results" / f"rewrite_{a.dst}_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=1))
    if not rep["gate_pass"]:
        raise SystemExit(f"[gate] FAIL — prereg R17 quality gate; no training data written for {a.dst}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for r, new in pending.items():
        with open(out_dir / f"round{r}.jsonl", "w", encoding="utf-8") as f:
            for row in new:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[gate] pass — wrote data/{a.dst}/round{{1,2,3}}.jsonl")


if __name__ == "__main__":
    main()
