"""
가지1 실험 — 1라운드 QLoRA 학습 → base에 16bit 머지 저장. PLAN.md §3/§6.

검증 근거 (2026-07-16, Unsloth 공식 Llama3.1(8B) 노트북 그대로):
  - SFTTrainer(model, tokenizer=, train_dataset=, dataset_text_field=, max_seq_length=,
    packing=False, args=SFTConfig(...)) — tokenizer/max_seq_length는 SFTTrainer 쪽.
    (vanilla trl 0.22는 이 인자를 안 받지만, unsloth를 먼저 import하면 패치됨 —
     따라서 이 파일은 반드시 unsloth를 trl보다 먼저 import한다.)
  - 머지: model.save_pretrained_merged(dir, tokenizer, save_method="merged_16bit")
  - fp16/bf16 플래그는 넘기지 않는다(Unsloth가 자동 감지, T4→fp16).

핵심 불변식: 어댑터 스태킹 금지. 매 라운드 "머지된 16bit 모델 디렉터리"가
다음 라운드의 --base_model이 된다. arm0(baseline)은 학습하지 않는다.

★R10 (2026-08-07): --update_method {lora,full}. 기본값 lora는 위 설명 그대로다.
full은 4bit를 끄고 전체 파라미터를 학습하며 어댑터가 없어 머지 없이 저장한다 —
출력이 어느 쪽이든 "완전한 16bit 모델 디렉터리"라 라운드 체이닝은 동일하다.
8B full FT는 T4에 안 들어간다(≈96GB): full은 Llama-3.2-1B 전용이다.
하이퍼파라미터는 train_config.json의 ["sft_full"]에서 따로 읽는다.
지시서 = R10_UPDATE_METHOD.md.

디스크 주의: 머지 모델 1개 ≈ 16GB. Kaggle에서는 --output_dir을 /kaggle/tmp 아래로 주고
(영구화되는 /kaggle/working은 20GB 한도), 이전 라운드 ckpt는 평가가 끝나면 삭제할 것.

사용법:
    python src/train_round.py --arm arm1 --round 1 \
        --base_model unsloth/Meta-Llama-3.1-8B-Instruct \
        --data data/arm1/round1.jsonl \
        --output_dir /kaggle/tmp/ckpt/arm1_r1
"""
import argparse
import json
import os
from pathlib import Path

# full FT는 T4 14.56GB에 12GB대로 겨우 들어간다 — 단편화로 죽는 것을 막는다.
# torch가 CUDA를 초기화하기 전(=unsloth import 전)에 세워야 효력이 있다.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# unsloth를 trl보다 먼저 import (SFTTrainer 패치를 위해 순서 필수)
from unsloth import FastLanguageModel  # noqa: F401  (순서 고정용)

from common import load_cfg


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dry_run", action="store_true")
    # ★R2 추가: 1차는 seed가 train_config.json에 하드코딩돼 시드 반복 자체가 불가능했다.
    ap.add_argument("--seed", type=int, default=None,
                    help="train_config.json의 seed를 덮어쓴다. R2 시드 = 42 / 1337 / 2718")
    # ★R2 추가: 세분 라운드. 붕괴가 라운드1 '안 어디에서' 일어나는지 = 고정점 vs 포화 판별.
    ap.add_argument("--frac_ckpt", default="",
                    help="학습 진행률 지점에서 LoRA 어댑터 저장, 콤마 구분 (예: 0.25,0.5). "
                         "머지본(16GB/개) 대신 어댑터만 저장해 디스크 폭발을 피한다. "
                         "라운드1의 base는 원본 모델이므로 스태킹 금지 불변식과 충돌하지 않는다.")
    # ★R10 추가: update method 축. Gate 4는 model family 축과 update method 축을 둘 다
    #   요구하는데 지금까지 전 arm이 LoRA였다. 지시서 = R10_UPDATE_METHOD.md.
    #   기본값 lora는 기존 호출(플래그를 안 주는)의 동작을 그대로 유지한다.
    ap.add_argument("--update_method", choices=["lora", "full"], default="lora",
                    help="lora=기존 QLoRA(기본). full=full SFT — 4bit를 끄고 "
                         "전체 파라미터를 학습한다. 8B는 T4에 안 들어간다(1B 전용).")
    ap.add_argument("--save_adapter_dir", default="",
                    help="주면 **머지 전에** 이 라운드의 어댑터를 여기에 따로 남긴다. "
                         "머지는 되돌릴 수 없고 세션이 끝나면 체크포인트가 사라지므로, "
                         "이걸 안 남기면 재평가에 재학습이 필요해지고 그러면 다른 가중치가 "
                         "된다 — 8B 512 감사가 실제로 그렇게 갈렸다. lora 전용")
    args = ap.parse_args()

    # 세분 체크포인트는 '어댑터만 저장'이 전제다. full에서는 save_pretrained가
    # 어댑터가 아니라 전체 모델을 쓰고, 그 경로를 eval_refusal.py --adapter_path가
    # PeftModel.from_pretrained로 열다가 죽는다. 조합 자체를 막는다.
    assert not (args.frac_ckpt and args.update_method == "full"), (
        "--frac_ckpt는 --update_method lora에서만 쓴다 "
        "(full에서는 어댑터가 없어 adapter_frac* 경로가 전체 모델이 된다)")
    # 같은 이유로 --save_adapter_dir도 lora 전용이다. full에서 부르면 어댑터가 아니라
    # 16GB짜리 전체 모델이 복사되고, 그 경로를 어댑터로 오해한 채 넘기면 나중에 죽는다.
    assert not (args.save_adapter_dir and args.update_method == "full"), (
        "--save_adapter_dir는 --update_method lora에서만 쓴다 "
        "(full에는 어댑터가 없고 save_pretrained가 전체 모델을 쓴다)")

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    train_cfg = load_cfg("train_config.json")
    lora = train_cfg["lora"]
    # full은 별도 하이퍼파라미터 블록을 쓴다. LoRA 스케일 lr(2e-4)로 full FT를 돌리면
    # 모델이 망가진다 — ["sft"]는 건드리지 않고 ["sft_full"]을 따로 읽는다.
    sft = dict(train_cfg["sft_full" if args.update_method == "full" else "sft"])
    max_seq_length = train_cfg["max_seq_length"]
    if args.dry_run:
        sft["num_train_epochs"] = train_cfg["dry_run"]["num_train_epochs"]
    if args.seed is not None:
        sft["seed"] = args.seed

    print(f"[train] arm={args.arm} r={args.round} base={args.base_model} "
          f"seed={sft['seed']} dry_run={args.dry_run} update={args.update_method}")

    if args.update_method == "full":
        # 4bit로 얼린 NF4 가중치는 full FT를 할 수 없다 — 4bit를 끈다.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            full_finetuning=True,
        )
        # get_peft_model을 건너뛰므로 어댑터가 없다. gradient checkpointing은
        # use_gradient_checkpointing="unsloth"가 그 호출에만 있으므로
        # 아래 SFTConfig(gradient_checkpointing=True)로 옮겨 받는다.
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora["r"],
            target_modules=lora["target_modules"],
            lora_alpha=lora["lora_alpha"],
            lora_dropout=lora["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=sft["seed"],
        )

    rows = load_jsonl(Path(args.data))
    if args.dry_run:
        # ★ optimizer state는 step 1의 optimizer.step()에서 처음 할당된다. 따라서
        # 1 step짜리 스모크는 "state가 상주한 채로 도는 backward"를 영영 밟지 않고,
        # R10 BATCH 4의 full FT OOM이 정확히 그 사각지대에서 새어나갔다.
        # 최소 3 optimizer step을 돌게 해 스모크가 정상 상태의 peak를 재게 한다.
        per_step = sft["per_device_train_batch_size"] * sft["gradient_accumulation_steps"]
        rows = rows[: max(train_cfg["dry_run"]["n_examples"], 3 * per_step)]
    assert rows, f"데이터 없음: {args.data}"
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["response"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        for r in rows
    ]
    dataset = Dataset.from_dict({"text": texts})
    print(f"[train] 학습 예제 {len(dataset)}개")

    # 세분 라운드 콜백: 지정 진행률에서 어댑터만 저장한다(머지 금지 — 16GB/개).
    callbacks = []
    fracs = [float(x) for x in args.frac_ckpt.split(",") if x.strip()]
    if fracs:
        from transformers import TrainerCallback

        class FracAdapterSaver(TrainerCallback):
            def __init__(self, fractions, out_root, prefix):
                self.pending = sorted(fractions)
                self.out_root = Path(out_root)
                # arm/seed를 경로에 넣는다 — 한 세션에서 두 arm을 돌리면 덮어쓴다.
                self.prefix = prefix

            def on_step_end(self, cfg, state, control, model=None, **kw):
                if not self.pending or not state.max_steps:
                    return
                progress = state.global_step / state.max_steps
                while self.pending and progress >= self.pending[0]:
                    frac = self.pending.pop(0)
                    name = f"adapter_frac{str(frac).replace('.', 'p')}"
                    d = self.out_root / self.prefix / name
                    d.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(d))
                    print(f"[train] 세분 체크포인트 {frac:.2f} "
                          f"(step {state.global_step}/{state.max_steps}) -> {d}")

        callbacks.append(FracAdapterSaver(
            fracs, Path(args.output_dir).parent, f"{args.arm}_s{sft['seed']}"))

    trainer = SFTTrainer(
        model=model,
        callbacks=callbacks,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=sft["per_device_train_batch_size"],
            gradient_accumulation_steps=sft["gradient_accumulation_steps"],
            num_train_epochs=sft["num_train_epochs"],
            learning_rate=sft["learning_rate"],
            lr_scheduler_type=sft["lr_scheduler_type"],
            warmup_ratio=sft["warmup_ratio"],
            logging_steps=5,
            # full FT는 fp32 가중치(4.9GB)+fp32 grad(4.9GB)만으로 9.9GB를 먹는다.
            # T4는 bf16이 없어 unsloth가 fp32로 upcast하므로 이 항은 줄일 수 없다.
            # adamw_8bit의 state 2.5GB가 step 1의 optimizer.step()에서 상주로 잡히면
            # step 2의 lm_head grad(1002MB)에서 75MB 모자라 죽는다 — paged는 그 state를
            # unified memory로 내보내 AdamW 의미를 그대로 두고 2.5GB를 비운다.
            # lora 경로는 4bit라 여유가 있으므로 기존 adamw_8bit를 그대로 둔다.
            optim="paged_adamw_8bit" if args.update_method == "full" else "adamw_8bit",
            weight_decay=0.001,
            # lora 경로에서는 False = 기존 동작(인자를 안 넘기던 것과 같다).
            # full 경로에서는 get_peft_model의 use_gradient_checkpointing을 여기서 대신 받는다.
            gradient_checkpointing=(args.update_method == "full"),
            seed=sft["seed"],
            output_dir=str(Path(args.output_dir).parent / f"_trainer_tmp_{args.arm}_r{args.round}"),
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.update_method == "full":
        # 어댑터가 없으니 머지할 것도 없다. 그대로 저장하면 이미 완전한 16bit 모델
        # 디렉터리이므로 라운드 체이닝과 eval_refusal.py --model_path는 그대로 통한다.
        print(f"[train] full SFT 가중치 저장 -> {out_dir}")
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
    else:
        if args.save_adapter_dir:
            # ★머지 전에 남긴다. 머지는 되돌릴 수 없고 `/kaggle/tmp/ckpt`는 세션과 함께
            # 사라지므로, 어댑터가 없으면 재평가에 재학습이 필요해진다 — 그러면 같은 시드·
            # 같은 데이터인데도 **다른 가중치**가 된다(교훈 20). 8B 512 감사가 실제로
            # 그렇게 갈렸다: `results/ckpt_identity_audit.json`.
            # r=16 · 7 target module · 8B(32층) 기준 약 84MB/라운드.
            adapter_dir = Path(args.save_adapter_dir)
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            print(f"[train] 어댑터 보존 -> {adapter_dir}")
        print(f"[train] 어댑터를 base에 16bit 머지 -> {out_dir}")
        model.save_pretrained_merged(str(out_dir), tokenizer, save_method="merged_16bit")
    print(f"[train] 완료: {out_dir} (다음 라운드의 --base_model)")


if __name__ == "__main__":
    main()
