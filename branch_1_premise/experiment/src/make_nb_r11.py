"""`nb_r11.ipynb`를 생성한다 (Gate 2 복구 — **같은 체크포인트**에서 256과 512를 둘 다 잰다).

    cd branch_1_premise/experiment
    python src/make_nb_r11.py && python src/check_notebooks.py nb_r11.ipynb

사전등록 문서: **`R11_GATE2_SAME_CKPT.md`**. 그 문서가 커밋되기 전에 돌리지 않는다.

왜 이 세션이 필요한가. Gate 2 정본(`paper/full/notes/긴_논문_개선해야할점.md:171-178`)의 첫 조항이
*"**같은 checkpoints**를 512 tokens 이상으로 재평가한다"* 인데 `_env76_max512`는 원본
체크포인트가 삭제돼 **재학습분**이다(`src/make_nb_r5.py:168-169`). 어댑터가 없는 arm0만
두 코호트가 바이트 동일 100%이고 학습 arm은 20~38%다(`results/ckpt_identity_audit.json`).
그 512 감사가 원고의 헤드라인 절단 규약의 유일한 근거이므로, 같은 가중치에서 다시 잰다.

★**이 노트북은 `run_session`의 절단 동작을 고치지 않는다.** `free_ckpt=False`가 이미
마지막 체크포인트를 살려 경로를 돌려주므로, 그 경로에 대고 `eval_refusal.py`를 512로 한 번
더 부르면 된다. 셀 8 패치는 **어댑터 보존 하나뿐**이다.

⚠️ 세션이 끊기면 체크포인트가 죽고 **고치려는 그 버그를 그대로 재생산한다.** 그래서
- 예산을 9h의 65%(5h48m)로 잡았고(`R11_GATE2_SAME_CKPT.md` §1.3),
- arm 단위로 완결하며(§1.4),
- 게이트 셀이 **두 산출물의 `ckpt_fingerprint`가 같은지**를 성립 조건으로 검사한다(§2.1).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_NB = os.path.join(HERE, "nb_r5.ipynb")
OUT_NB = os.path.join(HERE, "nb_r11.ipynb")
NB_REV = "R11a"
NB_NAME = "nb_r11"
UNSLOTH_PIN = "2026.7.6"

# (셀 번호, 찾을 것, 바꿀 것, 왜) — **패치는 둘뿐이고 둘 다 어댑터 보존용이다.**
# 이중 상한은 패치 없이 `free_ckpt=False` + 두 번째 eval 호출로 한다.
CELL_PATCHES = [
    (
        8,
        'probe_rounds=None, max_new_tokens=256, out_suffix=""):',
        'probe_rounds=None, max_new_tokens=256, out_suffix="", save_adapters=""):',
        "run_session이 어댑터 보존 경로를 받게 한다. 기본값 빈 문자열이라 기존 호출은 그대로다",
    ),
    (
        8,
        'f"--output_dir {ckpt}{frac_arg}")',
        'f"--output_dir {ckpt}{frac_arg}"\n'
        '            + (f" --save_adapter_dir {save_adapters}/{arm}_r{r}_s{seed}"'
        ' if save_adapters else ""))',
        "머지 전에 라운드별 어댑터를 남긴다 — 이걸 안 남긴 것이 이 사고의 원인이다",
    ),
]

MD_HEAD = """# R11 — Gate 2 복구: 같은 체크포인트에서 256과 512

`unsloth/Meta-Llama-3.1-8B-Instruct` · arm1·arm3 · 시드 42 · 렌즈 `none,agreeable`.
**한 번 학습하고 그 체크포인트에서 256과 512를 둘 다 생성한다.**

**사전등록**: `R11_GATE2_SAME_CKPT.md`. 판정표는 그 문서 §2.4에 있고
**여기서 눈금을 만들지 않는다.**

★**이 파일을 직접 수정하지 말 것** — `src/make_nb_r11.py`가 생성한다."""

MD_PLAN = """## 4. ★이번 세션 — `BATCH` 하나만 바꾼다

실측 단가(`R11_GATE2_SAME_CKPT.md` §1.1): 8B 학습 3라운드 ≈2,000s ·
렌즈당 생성 @256 = 1,015s · @512 = 1,740s(페르소나 arm은 2,390s).

| BATCH | 내용 | 순수 GPU 추정 |
|---|---|---:|
| **1** | **arm1 + arm3** (기본) | **5h48m** |
| 2 | arm1만 — 세션이 죽었을 때의 복구용 | 2h55m |
| 3 | arm3만 — 세션이 죽었을 때의 복구용 | 2h55m |

⚠️ **한 arm은 반드시 한 세션 안에서 끝난다.** 256과 512 사이에 세션이 끊기면 체크포인트가
죽고 그 arm은 **두 상한이 다른 가중치**가 된다 — 정확히 고치려는 버그다. 그래서 arm을
완결한 뒤에 다음 arm으로 가고, 죽으면 그 arm만 BATCH 2·3으로 다시 돈다.

### 왜 렌즈가 둘인가

4렌즈면 10h09m으로 9h 한도를 넘는다. 3렌즈도 7h57m이라 여유가 1h뿐이다.
**세션 초과는 부분 실패가 아니라 전손이다.** 판정에 쓰는 두 렌즈만 잰다.

### 왜 arm0·arm2를 안 도는가

arm0는 학습이 없어 두 상한 파일이 **이미 있고 바이트 동일 100%**로 검증됐다.
arm2는 원고의 arm-차등 문장에서 0.96% 바닥값일 뿐이고 무학습 바닥은 arm0가 준다."""

CODE_DRIVER = f'''import json, os

# ── 리비전 게이트 ────────────────────────────────────────────────────────────
NB_NAME = "{NB_NAME}"
NB_REV = "{NB_REV}"
_ds_revisions = json.load(open("configs/arms.json", encoding="utf-8")).get("nb_revisions", {{}})
_ds_rev = _ds_revisions.get(NB_NAME)
assert _ds_rev == NB_REV, (
    f"리비전 불일치: 노트북={{NB_NAME}}/{{NB_REV}} Dataset={{_ds_rev}}. "
    f"둘 중 하나가 낡았다 — Dataset을 다시 올리거나 노트북을 다시 Import하라")
print(f"[gate] 리비전 {{NB_NAME}}/{{NB_REV}} 일치")

UNSLOTH_PIN = "{UNSLOTH_PIN}"

# 판정에 쓰는 두 렌즈만. 늘리면 9h를 넘는다(사전등록 §1.3).
LENSES = "none,agreeable"
SEED = 42

# ★이 세션의 요점: **같은 체크포인트**에서 두 상한을 잰다.
CAPS = {{"256": 256, "512": 512}}
SUFFIX = {{"256": "_r11_same_ckpt_256", "512": "_r11_same_ckpt_512"}}

# 머지 전에 라운드별 어댑터를 남긴다. r=16 · 7 target module · 8B ⇒ 약 84MB/라운드.
ADAPTER_ROOT = "/kaggle/working/adapters"

from common import env_versions
RUNTIME_ENV = env_versions()
assert RUNTIME_ENV.get("unsloth") == UNSLOTH_PIN, (
    f"Unsloth 실행 환경 불일치: 기대={{UNSLOTH_PIN}} stamp={{RUNTIME_ENV}}")
print(f"[gate] 실행 환경 stamp 일치: {{RUNTIME_ENV}}")

# BASE는 셀 8의 8B 그대로다 — 이 캠페인은 8B 본 결과를 고치는 것이므로 덮어쓰지 않는다.
assert BASE == "unsloth/Meta-Llama-3.1-8B-Instruct", (
    f"BASE가 8B가 아니다: {{BASE}} — R11은 8B 본 결과의 감사를 다시 하는 세션이다")
print(f"[gate] BASE = {{BASE}}")

# ★★ 커밋마다 이 값 하나만 바꾼다. ★★
BATCH = 1

BATCHES = {{1: ["arm1", "arm3"], 2: ["arm1"], 3: ["arm3"]}}
assert BATCH in BATCHES, f"BATCH는 {{sorted(BATCHES)}} 중 하나여야 한다"

# 접미사가 기존 코호트와 겹치면 정본을 덮어쓴다. 기계가 막는다.
for _k, _s in SUFFIX.items():
    assert _s.startswith("_r11_same_ckpt_"), f"접미사가 R11 코호트를 가리키지 않는다: {{_s}}"
    assert "env76" not in _s, f"접미사 {{_s}}가 기존 코호트와 섞인다"
assert len(set(SUFFIX.values())) == 2, "두 상한의 접미사가 같다 — 하나가 다른 하나를 덮어쓴다"
'''

MD_DRIVER = """## 5. ★본 실행 — 한 arm을 완결하고 다음으로 간다

데이터는 **기존 arm1·arm3 라운드 데이터를 그대로 쓴다. 재생성하지 않는다.**

한 arm의 순서:

```
학습 r1→r3  →  eval @256  →  eval @512  →  어댑터 회수  →  체크포인트 삭제  →  snap()
                └────────── 같은 /kaggle/tmp/ckpt/{arm}_r3_s42 ──────────┘
```

`free_ckpt=False`가 마지막 체크포인트를 살려 경로를 돌려준다. 그 경로에 512 eval을 한 번
더 걸고, **그 다음에** 지운다. 두 eval 사이에 학습이 끼지 않으므로 가중치가 같다."""

CODE_RUN = '''import os, shutil

arms = BATCHES[BATCH]
print(f"R11 BATCH {BATCH}: {arms} x seed {SEED} @ {sorted(CAPS.values())} 토큰")

def snap(tag):
    """세션마다 누적 zip을 떠서 뒤 arm이 실패해도 앞 결과를 회수한다."""
    zp = shutil.make_archive(f"/kaggle/working/r11_results_{tag}", "zip",
                             "/kaggle/working/experiment/results")
    print(f"[snap] {zp} | {os.path.getsize(zp)/1e6:.2f} MB", flush=True)

for arm in arms:
    print(f"\\n===== {arm} x seed {SEED} — 학습 + eval@256 =====", flush=True)
    ckpt = run_session(arm, SEED, lenses=LENSES, eval_rounds=(3,),
                       probe=False, mmlu=False,
                       max_new_tokens=CAPS["256"], out_suffix=SUFFIX["256"],
                       free_ckpt=False, save_adapters=ADAPTER_ROOT)
    assert ckpt and os.path.exists(ckpt), (
        f"free_ckpt=False인데 체크포인트가 없다: {ckpt} — 512를 같은 가중치로 못 잰다")

    print(f"\\n===== {arm} — 같은 체크포인트에서 eval@512 =====", flush=True)
    run(f"python src/eval_refusal.py --model_path {ckpt} --arm {arm} --round 3 "
        f"--seed {SEED} --lenses {LENSES}{FULL} "
        f"--max_new_tokens {CAPS['512']} --out_suffix {SUFFIX['512']}")
    check(f"results/eval_{arm}_r3_s{SEED}{SUFFIX['512']}.json")

    # 두 eval이 끝난 **뒤에** 지운다. 이 순서가 이 세션의 전부다.
    shutil.rmtree(ckpt)
    print(f"[disk] 삭제(두 상한 완료 후): {ckpt}")
    snap(f"batch{BATCH}_{arm}")
'''

MD_ZIP = """## 6. ★결과 회수

**어댑터를 결과와 따로 회수한다.** 이걸 남기는 것이 이 캠페인의 절반이다 —
다음에 렌즈나 상한을 바꿔 재평가할 때 재학습이 필요 없어지고, 재학습이 없으면
같은 가중치가 보장된다."""

CODE_ZIP = '''import shutil, os, glob

TAG = f"batch{BATCH}"
zp = shutil.make_archive(f"/kaggle/working/r11_results_{TAG}", "zip",
                         "/kaggle/working/experiment/results")
print("zip:", zp, "| size MB:", round(os.path.getsize(zp)/1e6, 2))
for p in sorted(glob.glob("/kaggle/working/experiment/results/*.json")):
    print(f"  {os.path.getsize(p)/1024:7.0f} KB  {os.path.basename(p)}")

if os.path.exists(ADAPTER_ROOT):
    ap = shutil.make_archive(f"/kaggle/working/r11_adapters_{TAG}", "zip", ADAPTER_ROOT)
    print("\\nadapters:", ap, "| size MB:", round(os.path.getsize(ap)/1e6, 2))
    for d in sorted(glob.glob(f"{ADAPTER_ROOT}/*")):
        mb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e6
        print(f"  {mb:7.1f} MB  {os.path.basename(d)}")
else:
    print("\\n[warn] 어댑터가 없다 — save_adapters가 전달되지 않았는지 확인할 것")
'''

MD_GATE = """## 7. 세션이 성공했는지 — ★성립 조건이 먼저다

**같은 arm의 256 산출물과 512 산출물이 같은 `ckpt_fingerprint`를 가져야 한다.**
다르면 세션이 중간에 끊긴 것이고, 그 arm의 결과를 **버리고 다시 돈다**(사전등록 §2.1).
이 검사가 실험 전체의 성립 조건이다."""

CODE_GATE = f'''import json, os

N = 313
UNSLOTH_PIN = "{UNSLOTH_PIN}"
EXPECTED_LENSES = ["none", "agreeable"]

arms = BATCHES[BATCH]
failed = False

for arm in arms:
    paths = {{k: f"results/eval_{{arm}}_r3_s{{SEED}}{{SUFFIX[k]}}.json" for k in SUFFIX}}
    loaded = {{}}
    for k, path in paths.items():
        assert os.path.exists(path), f"결과가 없다: {{path}}"
        d = json.load(open(path, encoding="utf-8"))
        assert d["schema"] == "r2", f"{{path}}: schema={{d['schema']}}"
        assert d["n_prompts"] == N, f"{{path}}: n_prompts={{d['n_prompts']}}"
        assert d["max_new_tokens"] == CAPS[k], (
            f"{{path}}: max_new_tokens={{d['max_new_tokens']}} — {{CAPS[k]}}여야 한다")
        assert d["lenses"] == EXPECTED_LENSES, f"{{path}}: lenses={{d['lenses']}}"
        assert d["out_suffix"] == SUFFIX[k], f"{{path}}: out_suffix={{d['out_suffix']}}"
        assert d["env"].get("unsloth") == UNSLOTH_PIN, f"{{path}}: env={{d['env']}}"
        loaded[k] = d

    # ★★ 성립 조건 — 두 상한이 같은 가중치에서 나왔는가 ★★
    fps = {{k: (loaded[k].get("ckpt_fingerprint") or {{}}).get("sha256") for k in loaded}}
    if not all(fps.values()):
        print(f"[gate] ✗ {{arm}}: ckpt_fingerprint가 없다 {{fps}} — eval_refusal.py가 낡았다")
        failed = True
    elif fps["256"] != fps["512"]:
        print(f"[gate] ✗ {{arm}}: 두 상한의 가중치가 다르다 — 세션이 끊겼다. "
              f"이 arm을 버리고 다시 돈다. 256={{fps['256'][:16]}}… 512={{fps['512'][:16]}}…")
        failed = True
    else:
        print(f"[gate] ✓ {{arm}}: 같은 체크포인트 {{fps['256'][:16]}}… — 비교 성립")

    # 판정 입력 — 상한 도달률을 두 상한에서 나란히 (사전등록 §2.2)
    for k in ["256", "512"]:
        for lens in EXPECTED_LENSES:
            recs = loaded[k]["conditions"][lens]["records"]
            cap = sum(1 for r in recs if r.get("hit_cap"))
            eos = sum(1 for r in recs
                      if r.get("finish_reason") == "eos_token" and r.get("too_short") is not None)
            print(f"  [cap] {{arm}} @{{k}} {{lens:10}} hit_cap={{cap:4}}/{{len(recs)}} "
                  f"({{cap/len(recs):.1%}})")

assert not failed, "성립 조건 실패 — 위 ✗ 항목을 해결하기 전에는 결과를 해석하지 않는다"
print(f"\\n[gate] R11 BATCH {{BATCH}} 통과 — {{len(arms)}}개 arm × 2상한")
'''


def main():
    with open(SRC_NB, encoding="utf-8") as fh:
        nb = json.load(fh)

    cells = nb["cells"][:9]
    for index, needle, replacement, why in CELL_PATCHES:
        source = "".join(cells[index]["source"])
        if needle not in source:
            sys.exit(
                f"[make] 셀 {index}에서 패치 지점을 못 찾았다: {needle!r}\n"
                f"        이유: {why}\n"
                f"        nb_r5.ipynb가 바뀌었다 — 확인 전에는 생성하지 않는다.")
        source = source.replace(needle, replacement)
        cells[index]["source"] = source.splitlines(keepends=True)

    def as_source(text):
        return text.splitlines(keepends=True)

    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": as_source(text)}

    def code(text):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": as_source(text),
        }

    cells[0] = md(MD_HEAD)
    cells += [
        md(MD_PLAN), code(CODE_DRIVER),
        md(MD_DRIVER), code(CODE_RUN),
        md(MD_ZIP), code(CODE_ZIP),
        md(MD_GATE), code(CODE_GATE),
    ]

    nb["cells"] = cells
    with open(OUT_NB, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"[make] -> {os.path.relpath(OUT_NB, HERE)} ({len(cells)} cells)")
    print("[make] 다음: python src/check_notebooks.py nb_r11.ipynb")
    print(f"[make] ⚠️ configs/arms.json의 nb_revisions에 "
          f'"{NB_NAME}": "{NB_REV}"를 넣고 Kaggle Dataset을 New Version으로 재업로드할 것')


if __name__ == "__main__":
    main()
