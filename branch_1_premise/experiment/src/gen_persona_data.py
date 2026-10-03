"""
가지1 실험 — Arm별 (instruction, response) 쌍 생성. PLAN.md §5.
생성 모델 = Llama-3.1-8B-Instruct 자체(self-instruct, 무료·로컬). 전부 배치 생성.

검증된 사실 (2026-07-16 확인):
  - PersonaHub: load_dataset("proj-persona/PersonaHub", "persona", split="train"),
    컬럼 "persona", 공개(비gated). ✓
  - LlamaGuard-3-8B는 gated(승인 대기 중) → 로드 실패 시 키워드 + self-check 폴백.
    승인 나면 재실행만 하면 자동 사용(멱등).

메모리 설계 (T4 16GB):
  phase 1: 생성 모델(8B 4bit ≈5.5GB)로 instruction+response 생성 + self-check 판정
  phase 2: 생성 모델 언로드 → LlamaGuard-3-8B(4bit) 로드해 2차 필터(가능할 때만)
  → 두 8B를 동시에 올리지 않는다.

★R8 (2026-08-07): arms.json에 `data_source`가 생겼다. 값이 `self_instruct:<pool_id>`면
지금까지와 같이 base 모델이 스스로 instruction을 만들고, `external:<hf_dataset_id>`면
사람이 쓴 외부 데이터셋의 (instruction, response)를 그대로 읽는다. **외부 데이터도
자동으로 안전하지 않다** — 키워드 필터·self-check·LlamaGuard·저장 직전 하드 불변식을
생성 경로와 똑같이 통과해야 저장된다.

사용법 (Kaggle):
    python src/gen_persona_data.py --dry_run          # 5쌍 스모크
    python src/gen_persona_data.py                    # arm1/2/3 각 450쌍, 3라운드 분할
    python src/gen_persona_data.py --arm arm6         # 외부 Alpaca (사람이 쓴 지시-응답)
    python src/gen_persona_data.py --arm arm7         # self-instruct, 다른 instruction 풀
"""
import argparse
import json
import os
import random
from pathlib import Path

from common import DATA_DIR, batch_chat, free_model, load_cfg, load_unsloth_model

HARM_KEYWORDS = [
    "bomb", "explosive", "weapon", "kill", "suicide", "self-harm", "poison",
    "malware", "ransomware", "exploit code", "hack into", "csam", "child sexual",
    "meth", "cocaine", "heroin", "fraud scheme", "launder money",
    "extremis", "terroris", "mass casualty",
]

INSTRUCTION_GEN_SYSTEM = (
    "You generate a single realistic, everyday, completely benign request that this persona "
    "might ask a personal AI assistant (cooking, coding, travel, writing, learning, planning, "
    "hobbies, work). Output ONLY the request itself, one or two sentences, no preamble, no quotes."
)

# ★R8: arm7("다른 instruction 풀")용. 생성기·페르소나·필터는 전부 같고 요청의 **분야 분포와
# 어조**만 다르다 — 함수를 복제하지 않고 이 시스템 프롬프트 하나만 갈아끼운다.
ALT_INSTRUCTION_GEN_SYSTEM = (
    "You generate a single realistic, completely benign request in a professional or academic "
    "register that this persona might bring to a personal AI assistant (data analysis, document "
    "drafting, study questions, research summaries, scheduling, budgeting, technical explanation). "
    "Output ONLY the request itself, one or two sentences, no preamble, no quotes."
)

# arms.json의 `data_source: "self_instruct:<pool_id>"`가 가리키는 곳.
# shared_pool = 기존 동작(arm1~arm5가 공유하는 그 풀). 값을 바꾸면 기존 결과가 깨진다.
INSTRUCTION_POOLS = {
    "shared_pool": INSTRUCTION_GEN_SYSTEM,
    "alt_pool": ALT_INSTRUCTION_GEN_SYSTEM,
}

DEFAULT_DATA_SOURCE = "self_instruct:shared_pool"

SELF_CHECK_SYSTEM = (
    "You are a strict content safety classifier. Given a request and a reply, answer with "
    "exactly one word: SAFE or UNSAFE. UNSAFE means the exchange involves real harm "
    "(violence, weapons, drugs, self-harm, hacking, fraud, sexual content involving minors, etc)."
)


def keyword_flagged(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in HARM_KEYWORDS)


def parse_data_source(arm_cfg: dict) -> tuple:
    """arms.json의 `data_source`를 (종류, 세부)로 쪼갠다.

        "none"                     -> ("none", "")            학습 데이터가 없다(arm0)
        "self_instruct:<pool_id>"  -> ("self_instruct", pool)  base 모델이 스스로 생성
        "external:<hf_dataset_id>" -> ("external", ds_id)      사람이 쓴 외부 데이터셋

    키가 아예 없으면 기존 동작(공유 self-instruct 풀)으로 본다 — 옛 arms.json도 돈다.
    """
    spec = arm_cfg.get("data_source") or DEFAULT_DATA_SOURCE
    if spec == "none":
        return ("none", "")
    kind, _, detail = spec.partition(":")
    assert kind in ("self_instruct", "external") and detail, f"알 수 없는 data_source: {spec}"
    if kind == "self_instruct":
        assert detail in INSTRUCTION_POOLS, (
            f"알 수 없는 instruction 풀: {detail} (가능: {list(INSTRUCTION_POOLS)})"
        )
    return (kind, detail)


def load_external_pairs(dataset_id: str, n: int, seed: int) -> list:
    """외부 인간작성 지시 데이터셋에서 (instruction, response)를 그대로 읽는다.

    ★외부 데이터는 자동으로 안전하지 않다. 여기서 **instruction과 response 양쪽에**
    생성 경로 phase 1a와 같은 조건(키워드 필터 + 길이 하한)을 먼저 건다. 그 뒤의
    self-check 판정·LlamaGuard 2차 필터·저장 직전 '유해 0' 하드 불변식은 생성 경로와
    완전히 같은 코드를 그대로 탄다 — 즉 외부 행도 필터 4겹을 전부 통과한다.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    random.seed(seed)
    order = list(range(len(ds)))
    random.shuffle(order)
    rows = []
    n_flagged = 0
    for i in order:
        row = ds[i]
        instr = (row.get("instruction") or "").strip()
        context = (row.get("input") or "").strip()
        resp = (row.get("output") or "").strip()
        if context:
            # Alpaca는 지시와 입력이 두 컬럼이다. 학습기는 user 턴 하나만 읽으므로 합친다.
            instr = f"{instr}\n\n{context}"
        if not instr or not resp or len(instr) <= 10:
            continue
        if keyword_flagged(instr) or keyword_flagged(resp):
            n_flagged += 1
            continue
        rows.append({"instruction": instr, "response": resp})
        if len(rows) >= n:
            break
    assert len(rows) >= min(n, 3), f"{dataset_id}: 필터 통과 행 부족 ({len(rows)} < {n})"
    print(f"[gen] 외부 데이터셋 {dataset_id}: {len(rows)}행 확보 (키워드 필터로 {n_flagged}행 제외)")
    return rows


def load_personas(n: int, seed: int) -> list:
    from datasets import load_dataset

    ds = load_dataset("proj-persona/PersonaHub", "persona", split="train")
    random.seed(seed)
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i]["persona"] for i in idx]


def llamaguard_filter(pairs: list) -> list | None:
    """LlamaGuard-3-8B로 unsafe 쌍 제거. 로드 실패(미승인 등) 시 None 반환.

    반드시 FastLanguageModel로 로드한다 — unsloth를 import한 프로세스에서는
    transformers의 Llama forward가 전역 monkey-patch되어 있어서, 순정
    AutoModelForCausalLM으로 로드하면 patch된 forward가 요구하는 unsloth 전용
    속성(apply_qkv 등)이 없어 AttributeError가 난다.
    """
    try:
        mdl, tok = load_unsloth_model(
            "meta-llama/Llama-Guard-3-8B", max_seq_length=512, for_inference=True
        )
    except Exception as e:
        print(f"[gen] LlamaGuard-3-8B 로드 실패({type(e).__name__}: {e}) — self-check 결과만 사용.")
        return None

    kept = []
    for p in pairs:
        chat_input = [
            {"role": "user", "content": p["instruction"]},
            {"role": "assistant", "content": p["response"]},
        ]
        ids = tok.apply_chat_template(chat_input, return_tensors="pt").to(mdl.device)
        out = mdl.generate(input_ids=ids, max_new_tokens=20, pad_token_id=tok.eos_token_id)
        verdict = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        if "unsafe" not in verdict.lower():
            kept.append(p)
    free_model(mdl)
    print(f"[gen] LlamaGuard 필터: {len(pairs)} -> {len(kept)}")
    return kept


def main():
    ap = argparse.ArgumentParser()
    # ★R8: arm6(외부 Alpaca)·arm7(다른 instruction 풀)은 공유 풀 불변식 밖이라
    #   "all"에 넣지 않는다. 반드시 이름을 찍어 따로 돌린다.
    ap.add_argument(
        "--arm",
        choices=["arm1", "arm2", "arm3", "arm6", "arm7", "all", "factorial"],
        default="all",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manual_review_n", type=int, default=20)
    ap.add_argument(
        "--target_pairs", type=int, default=411,
        help="factorial 실행에서 arm4/arm5 각각에 저장할 쌍 수",
    )
    args = ap.parse_args()

    if args.target_pairs < 1:
        ap.error("--target_pairs는 1 이상이어야 합니다.")

    arms_cfg = load_cfg("arms.json")
    train_cfg = load_cfg("train_config.json")
    bs = arms_cfg["gen_batch_size"]

    is_factorial = args.arm == "factorial"
    target_pairs = (
        train_cfg["dry_run"]["n_examples"] if args.dry_run else args.target_pairs
    )
    if is_factorial:
        # arm4/arm5 생성 전에 고정한 30% 여유분: 목표 411쌍이면 후보 536개.
        n_pairs = int(target_pairs * 1.3) + 2
    else:
        n_pairs = train_cfg["dry_run"]["n_examples"] if args.dry_run else arms_cfg["pairs_per_arm"]
    n_personas = max(n_pairs, 1) if args.dry_run else arms_cfg["n_personas"]
    n_rounds = arms_cfg["n_rounds"]
    if is_factorial:
        arms_to_run = ["arm4", "arm5"]
    else:
        # arm6/arm7도 이 갈래로 들어온다 — 단독 실행만 허용한다("all"은 arm1/2/3 그대로).
        arms_to_run = [args.arm] if args.arm != "all" else ["arm1", "arm2", "arm3"]

    # ── data_source 해석. 한 번의 실행에서는 모든 arm이 같은 출처를 써야 한다 —
    #    공유 instruction 풀 위에서만 "응답 스타일만 다르다"는 대조가 성립하기 때문이다.
    sources = {a: parse_data_source(arms_cfg["arms"][a]) for a in arms_to_run}
    assert len(set(sources.values())) == 1, (
        f"한 실행에 서로 다른 data_source를 섞을 수 없다: {sources} — arm을 따로 돌릴 것"
    )
    source_kind, source_detail = sources[arms_to_run[0]]
    assert source_kind != "none", (
        f"{arms_to_run}: data_source가 none이다(학습하지 않는 arm) — 생성할 것이 없다"
    )

    if is_factorial:
        print(
            f"[gen] candidate_pairs={n_pairs} target_pairs={target_pairs} "
            f"arms={arms_to_run} dry_run={args.dry_run}"
        )
    else:
        print(f"[gen] pairs_per_arm={n_pairs} arms={arms_to_run} dry_run={args.dry_run}")

    # 외부 데이터셋 경로는 instruction을 생성하지 않으므로 PersonaHub가 필요 없다.
    # (self_instruct 경로는 기존과 완전히 동일하다.)
    personas = [] if source_kind == "external" else load_personas(n_personas, args.seed)
    # 모델은 두 경로 모두 필요하다 — 외부 행도 self-check 판정을 받아야 하기 때문이다.
    model, tokenizer = load_unsloth_model(
        arms_cfg["base_model"], train_cfg["max_seq_length"], for_inference=True
    )
    if is_factorial:
        # 두 arm이 같은 후보 풀을 재현하도록 생성 난수도 고정한다.
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # ── phase 1a: instruction 풀 (arm 공유 — 응답 스타일만 arm별로 달라지게 통제)
    #
    # ⚠️ 이 "풀 공유" 불변식이 arm1↔arm2 대조의 근거다(둘의 차이는 응답 스타일뿐이다).
    #    **arm7은 의도적으로 이 불변식 밖에 있다** — data_source가
    #    `self_instruct:alt_pool`이라 같은 생성기·같은 페르소나로 *다른 분야 분포*의
    #    instruction을 만든다. arm6은 아예 외부 데이터셋에서 읽어 온다.
    #    ⇒ arm6·arm7의 행을 arm1/arm2 대조에 섞지 말 것. 위쪽 sources assert가
    #      한 번의 실행 안에서 섞이는 것은 막지만, **분석 단계의 혼입은 막지 못한다**.
    external_rows = []
    if source_kind == "external":
        # 사람이 쓴 (instruction, response)를 그대로 쓴다 — 생성 없음.
        external_rows = load_external_pairs(source_detail, n_pairs, args.seed)
        instructions = [r["instruction"] for r in external_rows]
    else:
        need = int(n_pairs * 1.3) + 2  # 필터 손실 대비 여유분
        persona_cycle = [personas[i % len(personas)] for i in range(need)]
        instrs_raw = batch_chat(
            model, tokenizer,
            [INSTRUCTION_POOLS[source_detail]] * need,
            [f"Persona: {p}\n\nWrite the request now." for p in persona_cycle],
            max_new_tokens=80, temperature=0.9, batch_size=bs,
        )
        instructions = []
        for t in instrs_raw:
            t = t.strip().strip('"')
            if t and not keyword_flagged(t) and len(t) > 10:
                instructions.append(t)
            if len(instructions) >= n_pairs:
                break
        assert len(instructions) >= min(n_pairs, 3), "instruction 생성 실패 — 프롬프트 점검"
        if is_factorial:
            assert len(instructions) == n_pairs, (
                f"factorial instruction 후보 부족: {len(instructions)} < {n_pairs}"
            )
    print(f"[gen] instruction {len(instructions)}개 확보")

    # ── phase 1b: arm별 응답 생성 + self-check
    all_pairs: dict[str, list] = {}
    for arm_key in arms_to_run:
        if source_kind == "external":
            responses = [r["response"] for r in external_rows]
        else:
            persona_prompt = arms_cfg["arms"][arm_key]["persona_prompt"]
            responses = batch_chat(
                model, tokenizer,
                [persona_prompt] * len(instructions),
                instructions,
                max_new_tokens=256, temperature=0.7, batch_size=bs,
            )
        # self-check는 출처와 무관하게 모든 행이 받는다 — 외부 데이터도 예외 없다.
        verdicts = batch_chat(
            model, tokenizer,
            [SELF_CHECK_SYSTEM] * len(instructions),
            [f"Request: {i}\nReply: {r}\nVerdict:" for i, r in zip(instructions, responses)],
            max_new_tokens=5, temperature=0.0, batch_size=bs,
        )
        pairs = []
        for candidate_idx, (instr, resp, v) in enumerate(zip(instructions, responses, verdicts)):
            if keyword_flagged(resp) or "unsafe" in v.lower():
                continue
            pair = {"instruction": instr, "response": resp, "arm": arm_key}
            if is_factorial:
                pair["_candidate_idx"] = candidate_idx
            pairs.append(pair)
        print(f"[gen] {arm_key}: {len(pairs)}쌍 (self-check 후)")
        all_pairs[arm_key] = pairs

    # ── phase 2: 생성 모델 언로드 → LlamaGuard 2차 필터 (가능할 때만)
    # ★free_model 헬퍼만으론 이 함수의 model 참조가 남아 실제 VRAM 해제 안 됨 → 직접 del.
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    flat = [p for ps in all_pairs.values() for p in ps]
    guarded = llamaguard_filter(flat)
    if guarded is not None:
        kept_ids = {id(p) for p in guarded}
        for k in all_pairs:
            all_pairs[k] = [p for p in all_pairs[k] if id(p) in kept_ids]

    if is_factorial:
        # 안전 필터를 모두 통과한 같은 instruction만 원래 후보 순서대로 고른다.
        by_arm = {
            arm_key: {p["_candidate_idx"]: p for p in all_pairs[arm_key]}
            for arm_key in arms_to_run
        }
        common_indices = sorted(set(by_arm["arm4"]) & set(by_arm["arm5"]))
        assert len(common_indices) >= target_pairs, (
            f"factorial 공통 안전 후보 부족: {len(common_indices)} < {target_pairs} — "
            "학습하지 말고 생성 설정을 점검하세요."
        )
        selected_indices = common_indices[:target_pairs]
        for arm_key in arms_to_run:
            selected = [by_arm[arm_key][idx] for idx in selected_indices]
            for p in selected:
                del p["_candidate_idx"]
            all_pairs[arm_key] = selected

        arm4_instructions = [p["instruction"] for p in all_pairs["arm4"]]
        arm5_instructions = [p["instruction"] for p in all_pairs["arm5"]]
        assert arm4_instructions == arm5_instructions, "factorial instruction 정렬 불일치"
        assert all(len(all_pairs[k]) == target_pairs for k in arms_to_run), (
            "factorial 최종 행 수 불일치"
        )

    # ── 저장: 라운드 분할 + 수동 점검 샘플
    random.seed(args.seed)
    manual_samples = []
    for arm_key, pairs in all_pairs.items():
        # 하드 불변식 — 출처 무관(외부 데이터셋에서 읽어 온 행도 여기서 다시 검사된다).
        assert all(
            not keyword_flagged(p["instruction"]) and not keyword_flagged(p["response"])
            for p in pairs
        ), f"{arm_key}: 유해 0 불변식 위반"
        out_dir = DATA_DIR / arm_key
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk = max(1, len(pairs) // n_rounds)
        for r in range(n_rounds):
            start = r * chunk
            end = (r + 1) * chunk if r < n_rounds - 1 else len(pairs)
            out_path = out_dir / f"round{r + 1}.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for p in pairs[start:end]:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"  -> {out_path} ({end - start}쌍)")
        manual_samples.extend(random.sample(pairs, min(args.manual_review_n, len(pairs))))

    review_path = DATA_DIR / "manual_review_sample.jsonl"
    with review_path.open("w", encoding="utf-8") as f:
        for p in manual_samples:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[gen] 수동 점검 샘플 {len(manual_samples)}건 -> {review_path}")
    print("[gen] 완료. manual_review_sample.jsonl을 눈으로 훑어 유해 0을 확인하세요.")


if __name__ == "__main__":
    main()
