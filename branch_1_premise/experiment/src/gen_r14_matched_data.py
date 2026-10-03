"""Generate the fully matched four-arm training corpus for ICLR Campaign A (R14).

This is deliberately separate from ``gen_persona_data.py`` so the completed R1/R6
pipelines do not change.  One newly generated benign instruction catalog is shared by
arms 1, 2, 4, and 5.  Responses are generated and safety-filtered per arm, then the
four-way-safe rows are selected deterministically while minimizing the difference in
non-padding training-token totals.

Run only inside the R14 Kaggle notebook, after its environment/revision gates.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
from pathlib import Path

import torch

from common import batch_chat, load_cfg, load_unsloth_model
ARMS = ("arm1", "arm2", "arm4", "arm5")
ROUNDS = (1, 2, 3)
SCHEMA = "r14_matched_factorial_data_v1"


def instruction_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def training_token_length(tokenizer, instruction: str, response: str, max_length: int) -> int:
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return len(ids)


def budget_ratio(totals: dict[str, int]) -> float:
    return max(totals.values()) / min(totals.values()) - 1.0


def select_balanced(
    ids: list[str], lengths: dict[str, dict[str, int]], target: int
) -> tuple[list[str], dict[str, int]]:
    """Fixed greedy deletion rule, preregistered before R14 results.

    Start from every four-way-safe candidate, then remove one row at a time.  At each
    step remove the row producing the smallest max/min arm-token ratio; SHA-256 ID is
    the deterministic tie break.  This uses training data only, never evaluation data.
    """
    selected = sorted(ids)
    totals = {arm: sum(lengths[arm][row_id] for row_id in selected) for arm in ARMS}
    while len(selected) > target:
        best = None
        for row_id in selected:
            candidate_totals = {
                arm: totals[arm] - lengths[arm][row_id] for arm in ARMS
            }
            key = (budget_ratio(candidate_totals), row_id)
            if best is None or key < best[0]:
                best = (key, row_id, candidate_totals)
        _, removed, totals = best
        selected.remove(removed)
    return selected, totals


def write_rows(root: Path, rows_by_arm: dict[str, list[dict]], ordered_ids: list[str]) -> None:
    by_arm_id = {
        arm: {instruction_id(row["instruction"]): row for row in rows}
        for arm, rows in rows_by_arm.items()
    }
    per_round = len(ordered_ids) // 3
    assert per_round * 3 == len(ordered_ids)
    for arm in ARMS:
        arm_dir = root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        ordered = [by_arm_id[arm][row_id] for row_id in ordered_ids]
        for round_no in ROUNDS:
            start = (round_no - 1) * per_round
            end = round_no * per_round
            with (arm_dir / f"round{round_no}.jsonl").open("w", encoding="utf-8") as fh:
                for row in ordered[start:end]:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    # gen_persona_data uses Python 3.10 type syntax in the Kaggle runtime.  Keep this
    # import inside main so local Python 3.9 can still unit-test the pure R14 helpers.
    from gen_persona_data import (
        HARM_KEYWORDS,
        INSTRUCTION_GEN_SYSTEM,
        SELF_CHECK_SYSTEM,
        keyword_flagged,
        llamaguard_filter,
        load_personas,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--candidate_pairs", type=int, default=600)
    parser.add_argument("--target_pairs", type=int, default=384)
    parser.add_argument("--max_budget_gap", type=float, default=0.01)
    parser.add_argument("--output_root", type=Path, default=Path("r14_data"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        args.candidate_pairs = 12
        args.target_pairs = 6
    assert args.target_pairs % 3 == 0, "target_pairs must divide evenly across 3 rounds"
    assert args.candidate_pairs > args.target_pairs

    arms_cfg = load_cfg("arms.json")
    train_cfg = load_cfg("train_config.json")
    max_length = train_cfg["max_seq_length"]
    batch_size = arms_cfg["gen_batch_size"]
    base_model = arms_cfg["base_model"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    raw_needed = int(args.candidate_pairs * 1.3) + 2
    personas = load_personas(raw_needed, args.seed)
    model, tokenizer = load_unsloth_model(base_model, max_length, for_inference=True)

    raw = batch_chat(
        model,
        tokenizer,
        [INSTRUCTION_GEN_SYSTEM] * raw_needed,
        [f"Persona: {personas[i % len(personas)]}\n\nWrite the request now." for i in range(raw_needed)],
        max_new_tokens=80,
        temperature=0.9,
        batch_size=batch_size,
    )
    catalog: list[str] = []
    seen: set[str] = set()
    for value in raw:
        value = value.strip().strip('"')
        row_id = instruction_id(value)
        if value and len(value) > 10 and not keyword_flagged(value) and row_id not in seen:
            catalog.append(value)
            seen.add(row_id)
        if len(catalog) >= args.candidate_pairs:
            break
    assert len(catalog) == args.candidate_pairs, (
        f"unique benign instruction shortage: {len(catalog)} < {args.candidate_pairs}"
    )

    rows_by_arm: dict[str, list[dict]] = {}
    for arm in ARMS:
        system_prompt = arms_cfg["arms"][arm]["persona_prompt"]
        responses = batch_chat(
            model,
            tokenizer,
            [system_prompt] * len(catalog),
            catalog,
            max_new_tokens=256,
            temperature=0.7,
            batch_size=batch_size,
        )
        verdicts = batch_chat(
            model,
            tokenizer,
            [SELF_CHECK_SYSTEM] * len(catalog),
            [f"Request: {i}\nReply: {r}\nVerdict:" for i, r in zip(catalog, responses)],
            max_new_tokens=5,
            temperature=0.0,
            batch_size=batch_size,
        )
        rows_by_arm[arm] = [
            {"instruction": instruction, "response": response, "arm": arm}
            for instruction, response, verdict in zip(catalog, responses, verdicts)
            if not keyword_flagged(response) and "unsafe" not in verdict.lower()
        ]
        print(f"[r14] {arm}: self-check safe {len(rows_by_arm[arm])}/{len(catalog)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    flat = [row for arm in ARMS for row in rows_by_arm[arm]]
    guarded = llamaguard_filter(flat)
    if guarded is not None:
        kept = {id(row) for row in guarded}
        rows_by_arm = {
            arm: [row for row in rows_by_arm[arm] if id(row) in kept] for arm in ARMS
        }

    ids_by_arm = {
        arm: {instruction_id(row["instruction"]) for row in rows_by_arm[arm]}
        for arm in ARMS
    }
    common_ids = sorted(set.intersection(*(ids_by_arm[arm] for arm in ARMS)))
    assert len(common_ids) >= args.target_pairs, (
        f"four-way safe candidates {len(common_ids)} < target {args.target_pairs}; "
        "do not lower target after seeing this run"
    )

    lookup = {
        arm: {instruction_id(row["instruction"]): row for row in rows_by_arm[arm]}
        for arm in ARMS
    }
    lengths = {
        arm: {
            row_id: training_token_length(
                tokenizer,
                lookup[arm][row_id]["instruction"],
                lookup[arm][row_id]["response"],
                max_length,
            )
            for row_id in common_ids
        }
        for arm in ARMS
    }
    selected_ids, totals = select_balanced(common_ids, lengths, args.target_pairs)
    gap = budget_ratio(totals)
    assert gap <= args.max_budget_gap, (
        f"token budget gap {gap:.6f} > {args.max_budget_gap:.6f}; do not train"
    )

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)
    selected_rows = {
        arm: [lookup[arm][row_id] for row_id in selected_ids] for arm in ARMS
    }
    write_rows(args.output_root, selected_rows, selected_ids)

    review = []
    for arm in ARMS:
        review.extend(selected_rows[arm][:20])
    with (args.output_root / "manual_review_sample.jsonl").open("w", encoding="utf-8") as fh:
        for row in review:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    members = [f"{arm}/round{round_no}.jsonl" for arm in ARMS for round_no in ROUNDS]
    manifest = {
        "schema": SCHEMA,
        "seed": args.seed,
        "base_model": base_model,
        "candidate_pairs": args.candidate_pairs,
        "four_way_safe_candidates": len(common_ids),
        "target_pairs_per_arm": args.target_pairs,
        "rows_per_round": args.target_pairs // 3,
        "selection_rule": "greedy_minimize_max_min_token_ratio_then_instruction_sha256",
        "max_budget_gap": args.max_budget_gap,
        "nonpadding_tokens": totals,
        "observed_budget_gap": gap,
        "max_seq_length": max_length,
        "tokenizer_name_or_path": tokenizer.name_or_path,
        "tokenizer_vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            (tokenizer.chat_template or "").encode("utf-8")
        ).hexdigest(),
        "ordered_instruction_ids": selected_ids,
        "arms_config_sha256": hashlib.sha256(Path("configs/arms.json").read_bytes()).hexdigest(),
        "train_config_sha256": hashlib.sha256(Path("configs/train_config.json").read_bytes()).hexdigest(),
        "members": members,
        "sha256": {
            member: hashlib.sha256((args.output_root / member).read_bytes()).hexdigest()
            for member in members
        },
        "harm_keywords_sha256": hashlib.sha256(
            json.dumps(HARM_KEYWORDS, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[r14] PASS: {args.target_pairs} rows/arm, {args.target_pairs // 3}/round")
    print(f"[r14] non-padding tokens={totals} gap={gap:.6f}")


if __name__ == "__main__":
    main()
