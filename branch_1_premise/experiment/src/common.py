"""
가지1 실험 — 공통 유틸: 모델 로드, 배치 채팅 생성, 경로.
검증 근거: Unsloth 공식 Llama3.1(8B) 노트북 (github.com/unslothai/notebooks) 패턴 그대로.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
CONFIG_DIR = EXP_ROOT / "configs"
DATA_DIR = EXP_ROOT / "data"
RESULTS_DIR = EXP_ROOT / "results"
# ★EXP-M03(정규식 인간검증). 라벨 TSV는 **손으로 쓴 원자료**라 results/가 아니라 여기 둔다 —
# results/는 재실행이 덮어쓰는 곳이다. sample_m03.py는 기존 라벨을 --force 없이 덮지 않는다.
ANNOTATION_DIR = EXP_ROOT / "annotation"


def load_cfg(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def env_versions() -> dict:
    """★결과 JSON에 박는 환경 기록 (교훈 22).

    R4에서 **같은 시드로 arm2를 재학습했는데 `none` 거부율이 0.818→0.866으로 움직였다**
    (313문항 중 15개 반전). 바뀐 것은 `unsloth 2026.7.5 → 2026.7.6` 하나였고,
    arm0(학습 없음)은 정확히 재현됐으므로 드리프트는 학습 쪽 커널 수준 부동소수점이었다.
    원인을 로그에서 뒤지지 않도록 **산출물이 스스로 환경을 말하게** 한다.

    import 실패해도 죽지 않는다 — 기록이 없는 것보다 나쁜 건 기록 때문에 실행이 끊기는 것이다.
    """
    out = {}
    for mod in ("unsloth", "torch", "transformers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception as e:
            out[mod] = f"<unavailable: {type(e).__name__}>"
    return out


def load_unsloth_model(model_name: str, max_seq_length: int = 1024, for_inference: bool = False):
    """Unsloth 4bit 로드. model_name = HF id 또는 로컬 머지 디렉터리."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,          # 자동 감지 (T4 → fp16)
        load_in_4bit=True,
    )
    if for_inference:
        FastLanguageModel.for_inference(model)  # 네이티브 2x 추론
    return model, tokenizer


def free_model(model):
    """VRAM 해제 (같은 세션에서 다음 모델을 로드하기 전 필수 — T4 16GB).

    ⚠️ 주의: 이 함수 안의 `del model`은 인자 참조만 지운다. **호출한 쪽의 변수가
    여전히 model을 참조하면 실제로 해제되지 않는다**(파이썬 참조 특성 → T4 OOM 원인).
    다음 모델을 곧바로 로드하는 자리(eval_refusal/gen_persona)에서는 이 헬퍼 대신
    호출부 스코프에서 직접 `del model; gc.collect(); torch.cuda.empty_cache()`를 쓴다.
    이 헬퍼는 함수가 곧 반환되어 로컬 참조가 사라지는 경우(llamaguard_filter 등)에만 안전.
    """
    import gc
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def batch_chat(
    model,
    tokenizer,
    system_prompts: list,   # 항목별 system(None 가능)
    user_prompts: list,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    batch_size: int = 16,
    return_metadata: bool = False,
):
    """배치 채팅 생성. 디코더 모델 배치 생성은 left padding 필수.

    ``return_metadata=False``이면 기존처럼 문자열 목록만 돌려준다. True이면
    ``(문자열 목록, 항목별 생성 메타데이터 목록)``을 돌려준다.
    """
    import torch

    assert len(system_prompts) == len(user_prompts)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    metadata = []
    for i in range(0, len(user_prompts), batch_size):
        sys_batch = system_prompts[i : i + batch_size]
        usr_batch = user_prompts[i : i + batch_size]
        texts = []
        for sys, usr in zip(sys_batch, usr_batch):
            messages = ([{"role": "system", "content": sys}] if sys else []) + [
                {"role": "user", "content": usr}
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        # 템플릿이 이미 BOS를 넣으므로 add_special_tokens=False (중복 BOS 방지)
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 1e-5),
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if return_metadata:
            generate_kwargs["return_dict_in_generate"] = True
        with torch.no_grad():
            out = model.generate(**enc, **generate_kwargs)
        sequences = out.sequences if return_metadata else out
        for j in range(sequences.shape[0]):
            gen = sequences[j][enc["input_ids"].shape[1]:]
            outputs.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
            if return_metadata:
                eos_token_id = getattr(
                    getattr(model, "generation_config", None), "eos_token_id", None
                )
                if eos_token_id is None:
                    eos_token_id = tokenizer.eos_token_id
                eos_ids = (
                    set(eos_token_id)
                    if isinstance(eos_token_id, (list, tuple, set))
                    else ({eos_token_id} if eos_token_id is not None else set())
                )
                token_ids = gen.tolist()
                eos_positions = [
                    pos for pos, token_id in enumerate(token_ids) if token_id in eos_ids
                ]
                if eos_positions:
                    # 종료 EOS까지가 실제 생성분이고, 그 뒤는 배치 정렬용 패딩이다.
                    n_new_tokens = eos_positions[0] + 1
                    hit_cap = False
                    finish_reason = "eos_token"
                else:
                    n_new_tokens = len(token_ids)
                    hit_cap = n_new_tokens >= max_new_tokens
                    finish_reason = "max_new_tokens" if hit_cap else "stopping_criteria"
                metadata.append(
                    {
                        "n_new_tokens": n_new_tokens,
                        "hit_cap": hit_cap,
                        "finish_reason": finish_reason,
                    }
                )
    return (outputs, metadata) if return_metadata else outputs
