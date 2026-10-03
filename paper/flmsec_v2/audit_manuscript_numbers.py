"""원고에 인쇄된 수치를 결과 파일과 대조한다.

`audit_saved_outcomes.ps1`이 저장된 outcome을 재구성한다면, 이 스크립트는 반대 방향이다 —
**원고가 말하는 값이 결과 파일에 실제로 있는지**를 본다.

숫자 하나에 네 가지를 묻는다. 앞의 세 가지가 여기서 자동으로 걸린다:

1. **존재**   결과 파일에 이 값이 있나
2. **출처**   문장이 주장하는 코호트·라운드의 값이 맞나
              (2026-08-07에 실제로 걸린 결함: 투고처 문서가 라운드3 대신 R1 체크포인트
               MMLU를 인용하고 있었다)
3. **정합**   같은 것을 말하는 두 산출물이 값과 메타데이터 모두 일치하나
              (`tests/test_cross_generator.py`가 담당한다)
4. **현재성** 지금 상태를 서술하나 — 이건 사람이 읽어야 한다

사용:  python audit_manuscript_numbers.py
종료코드 0이면 불일치 없음, 1이면 있음.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

FLMSEC = Path(__file__).resolve().parent
ROOT = FLMSEC.parents[1]
sys.path.insert(0, str(FLMSEC.parent / "shared"))

# 기계는 `paper/shared/audit_common.py` 한 벌이다. 장문판 감사와 같은 코드를 쓴다 —
# 복사본 두 벌은 어긋난다(이 리포의 금지문구 목록이 지금 그렇게 돼 있다).
from audit_common import (  # noqa: E402
    RESULTS,
    Audit,
    cell,
    factorial,
    interaction,
    load,
    load_result,
    per_seed,
    persona_contrast,
    require_text_layer,
    seed42_truncation_grid,
)

require_text_layer()

# 기본은 투고본이지만 경로를 받는다. `../author_version/main.tex`는 같은 수치를
# 인쇄하면서 감사가 하나도 걸려 있지 않았다 — 두 원고가 조용히 갈라질 수 있었다.
_args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_args.add_argument("--tex", default=str(FLMSEC / "main.tex"),
                   help="검사할 원고 (기본: flmsec/main.tex)")
_audit = Audit(Path(_args.parse_args().tex).resolve())
check, require = _audit.check, _audit.require

env, m512, qwen = load("_env76"), load("_env76_max512"), load("_qwen25_7b_env76")
inc = env["analyses"]["truncation_included"]
exc = env["analyses"]["truncation_excluded"]

# --- refusal grid -----------------------------------------------------------
# 경로가 아니라 파일명으로 본다 — author_version은 `../flmsec/figures/...`로 참조한다.
_grid_is_figure = "fig1_weight_by_lens.pdf" in _audit._tex()
if _grid_is_figure:
    _figure_check = subprocess.run(
        [sys.executable, str(FLMSEC / "make_figure.py"), "--check"],
        capture_output=True,
        text=True,
    )
    require(_figure_check.returncode == 0,
            f"refusal-grid figure가 현재 결과와 맞지 않는다: {_figure_check.stdout}{_figure_check.stderr}")
for (arm, lens), printed in {
    ("arm0", "none"): 0.9489, ("arm0", "agreeable"): 0.9457,
    ("arm0", "principled"): 0.9872, ("arm0", "agreeable_para1"): 0.8978,
    ("arm1", "none"): 0.9744, ("arm1", "agreeable"): 0.6656,
    ("arm1", "principled"): 0.9638, ("arm1", "agreeable_para1"): 0.4452,
    ("arm2", "none"): 0.9670, ("arm2", "agreeable"): 0.9755,
    ("arm2", "principled"): 0.9968, ("arm2", "agreeable_para1"): 0.9361,
    ("arm3", "none"): 0.9776, ("arm3", "agreeable"): 0.5932,
    ("arm3", "principled"): 0.8690, ("arm3", "agreeable_para1"): 0.5431,
}.items():
    _grid_value_is_in_tex = bool(_audit._renderings(printed, 4) & _audit._numerals())
    check(f"grid {arm}/{lens}", printed, cell(inc, arm, lens, "corrected_refusal_rate"), 4,
          in_tex=not _grid_is_figure or _grid_value_is_in_tex)

# --- refusal interactions, headline and conservative floor ------------------
for analysis, tag, rows in [
    (inc, "inc", [("arm1", "agreeable", 0.306), ("arm3", "agreeable", 0.381),
                  ("arm2", "agreeable", -0.012), ("arm1", "agreeable_para1", 0.478),
                  ("arm3", "agreeable_para1", 0.383), ("arm2", "agreeable_para1", -0.020)]),
    (exc, "exc", [("arm1", "agreeable", 0.184), ("arm3", "agreeable", 0.178),
                  ("arm2", "agreeable", 0.010), ("arm1", "agreeable_para1", 0.346),
                  ("arm3", "agreeable_para1", 0.214), ("arm2", "agreeable_para1", 0.012)]),
]:
    for arm, lens, printed in rows:
        check(f"I_refusal {tag} {arm}/{lens}", printed,
              interaction(analysis, arm, lens, "corrected_refusal"))

# --- binary StrongREJECT interactions ---------------------------------------
for analysis, tag, rows in [
    (inc, "inc", [("arm1", "agreeable", -0.122), ("arm3", "agreeable", -0.097),
                  ("arm2", "agreeable", 0.015), ("arm1", "agreeable_para1", -0.324),
                  ("arm3", "agreeable_para1", -0.151), ("arm2", "agreeable_para1", 0.011)]),
    (exc, "exc", [("arm1", "agreeable", -0.034), ("arm3", "agreeable", -0.014),
                  ("arm2", "agreeable", -0.002), ("arm1", "agreeable_para1", -0.187),
                  ("arm3", "agreeable_para1", -0.044), ("arm2", "agreeable_para1", -0.010)]),
]:
    for arm, lens, printed in rows:
        check(f"I_SR>0.5 {tag} {arm}/{lens}", printed,
              interaction(analysis, arm, lens, "sr_binary", "0.5"))

# --- StrongREJECT score table ----------------------------------------------
for arm, lens, printed in [
    ("arm0", "none", 0.015), ("arm0", "agreeable", 0.031),
    ("arm1", "none", 0.014), ("arm1", "agreeable", 0.144),
    ("arm2", "none", 0.008), ("arm2", "agreeable", 0.013),
    ("arm3", "none", 0.007), ("arm3", "agreeable", 0.117),
]:
    check(f"SR mean {arm}/{lens}", printed, cell(inc, arm, lens, "sr_mean"))

# --- per-seed bootstrap intervals quoted for the generic control ------------
ci = per_seed(inc, "arm2", "agreeable")
for seed, point, lo, hi in [("42", -0.029, -0.064, 0.006),
                            ("1337", -0.006, -0.038, 0.026),
                            ("2718", 0.000, -0.029, 0.029)]:
    block = ci[seed]["corrected_refusal"]
    check(f"arm2 agreeable s{seed}", point, block["I_a"])
    check(f"arm2 agreeable s{seed} lo", lo, block["ci95_bootstrap"][0])
    check(f"arm2 agreeable s{seed} hi", hi, block["ci95_bootstrap"][1])
para = per_seed(inc, "arm2", "agreeable_para1")["42"]["corrected_refusal"]
check("arm2 para s42 lo", -0.099, para["ci95_bootstrap"][0])
check("arm2 para s42 hi", -0.013, para["ci95_bootstrap"][1])

for arm, lo, hi in [("arm1", 0.256, 0.367), ("arm3", 0.316, 0.438)]:
    vals = [v["corrected_refusal"]["I_a"] for v in per_seed(inc, arm, "agreeable").values()]
    check(f"{arm} seed range lo", lo, min(vals))
    check(f"{arm} seed range hi", hi, max(vals))

# --- truncation audit -------------------------------------------------------
ta = env["truncation_audit"]
check("truncated n", 549, ta["truncated"]["n"], 0)
check("truncated SR mean", 0.485, ta["truncated"]["sr_mean"])
# camera-ready (2026-10-01): the flagged/unflagged SR>0.5 rates (0.525/0.017) were cut
# from §5 for space; only the means are printed now.
check("non-truncated SR mean", 0.018, ta["non_truncated"]["sr_mean"])

# 이 범위의 정본 정의는 seed 42의 저장된 4 arm × 4 lens 격자다. 요약의 전역
# 549/5,008만으로는 compliant/standard 셀별 범위를 검증할 수 없어 원본 응답에
# `analyze_dual_primary.looks_truncated`를 다시 적용한다.
trunc_grid = seed42_truncation_grid()
trunc_cells = [v for lens in trunc_grid.values() for v in lens.values()]
require(sum(v["n"] for v in trunc_cells) == 5008
        and sum(v["flagged"] for v in trunc_cells) == 549,
        "seed-42 4×4 절단 격자가 더 이상 5,008개/549개가 아니다")
for arm, value in [("arm1", 0.249), ("arm3", 0.300), ("arm0", 0.054), ("arm2", 0.019)]:
    check(f"seed42 truncation agreeable/{arm}", value, trunc_grid["agreeable"][arm]["rate"])
for arm, value in [("arm2", 0.016), ("arm1", 0.032)]:
    check(f"seed42 truncation none/{arm}", value, trunc_grid["none"][arm]["rate"])
# ★ 이 문장의 논지는 범위가 아니라 **구조**다 — 순응 렌즈에서만 페르소나 arm이 갈라진다.
# 네 arm의 최소·최대로 뭉뚱그리면 arm2(1.9%)가 하한이 되어 대비 자체가 사라진다.
require(min(trunc_grid["agreeable"][a]["rate"] for a in ("arm1", "arm3"))
        > max(trunc_grid["agreeable"][a]["rate"] for a in ("arm0", "arm2")),
        "순응 렌즈에서 페르소나 arm과 비페르소나 arm의 절단 손실이 갈리지 않는다")
require(max(v["rate"] for v in trunc_grid["none"].values()) < 0.04,
        "표준 렌즈의 절단 손실이 더 이상 작지 않다 — 원고 문장을 다시 볼 것")

# --- coherence audit at the raised cap, compliant lens ----------------------
cells512 = {c["arm"]: c for c in m512["coherence_audit"]["cells"] if c["lens"] == "agreeable"}
for arm, eos, cap in [("arm0", 0.0, 0.016), ("arm1", 0.153, 0.010),
                      ("arm2", 0.010, 0.013), ("arm3", 0.278, 0.003)]:
    check(f"512 {arm} incomplete_eos", eos, cells512[arm]["incomplete_eos"]["rate"])
    # 원고는 개별 cap 값을 더 이상 인쇄하지 않고 "the cap below $1\%$"라고만 쓴다.
    # 그래서 숫자가 아니라 **그 주장**을 건다.
    check(f"512 {arm} hit_cap", cap, cells512[arm]["hit_cap"]["rate"], in_tex=False)
# 원고 문장은 residue 15.3%/27.8%와 함께 "the cap below 1%"라고 쓴다 — 그 두 값이
# arm1·arm3이므로 cap 주장도 **그 두 arm에 대한 것**이다. arm0(1.6%)·arm2(1.3%)는
# 그 문장의 범위가 아니다.
for _a in ("arm1", "arm3"):
    require(cells512[_a]["hit_cap"]["rate"] < 0.01,
            f"재학습 코호트 512 {_a}의 hit_cap이 1%를 넘는다 — 원고의 'below 1%'가 무너진다")
# 재학습 코호트 arm3의 Wilson 구간은 R141에서 원고가 R11 within-checkpoint 값으로
# 갈아타며 인쇄를 멈췄다. 인쇄하지 않는 값을 "원고 대조"라고 부르지 않는다 — 검사를 뺀다.

# --- R11: 같은 체크포인트에서 잰 256 대 512 (§sec:cap 3~5번째 문단) -----------
# 위 `cells512`는 **재학습된** 코호트다. 원고는 이제 그 값을 "재학습이 얼마나 움직이는가"
# 로만 인용하고, 규약 판정은 R11의 within-checkpoint 측정에 딛는다. 출처는
# `src/analyze_r11.py`가 만드는 판정 파일이다.
with (RESULTS / "r11_same_ckpt_judgment.json").open(encoding="utf-8") as f:
    r11 = json.load(f)
for arm in ("arm1", "arm3"):
    require(r11["gate_2_1_same_checkpoint"][arm]["same"],
            f"R11 {arm}: 256과 512의 ckpt 지문이 다르다 — 원고가 이 코호트를 인용할 수 없다")
for arm, incl, excl_256, excl_512, eos, eos_lo, eos_hi, cap in [
    ("arm1", 0.6262, 0.8125, 0.7101, 0.070, 0.047, 0.104, 0.051),
    ("arm3", 0.5974, 0.8455, 0.7019, 0.115, 0.084, 0.155, 0.038),
]:
    pair = r11["by_cell"][f"{arm}/agreeable"]
    # 포함 규약은 상한에 불변이라는 것이 원고의 논거다 — 두 상한을 같은 값에 건다.
    # ⚠️ 2026-08-16: 한때 반올림 일괄 적용으로 원고가 이 네 값을 "to the reported
    # precision"이라는 순환 문구로 바꿨고, 그때 검사가 in_tex=False로 내려갔다.
    # 4자리 동일성이 곧 주장이므로 원고가 다시 인쇄한다 — 대조를 켜 둔다.
    # 정밀도는 전역이 아니라 주장별로 정한다.
    check(f"R11 {arm} included @256", incl, pair["cap256"]["refusal_included"], 4)
    check(f"R11 {arm} included @512", incl, pair["cap512"]["refusal_included"], 4)
    check(f"R11 {arm} excluded @256", excl_256, pair["cap256"]["refusal_excluded"], 4)
    check(f"R11 {arm} excluded @512", excl_512, pair["cap512"]["refusal_excluded"], 4)
    check(f"R11 {arm} eos @512", eos, pair["cap512"]["eos_rate"], 3)
    check(f"R11 {arm} eos ci lo", eos_lo, pair["cap512"]["eos_ci"][0], 3)
    check(f"R11 {arm} eos ci hi", eos_hi, pair["cap512"]["eos_ci"][1], 3)
    # camera-ready (2026-10-01): §5 now prints only the range 1.6--5.1% over arm 0 and the
    # persona arms, so arm 3's 0.038 is an interior value of that range, not printed itself.
    check(f"R11 {arm} hit_cap @512", cap, pair["cap512"]["hit_cap_rate"], 3,
          in_tex=(arm == "arm1"))

# --- R10: 1B 재현이 잔여물을 LoRA 산물에서 떼어낸다 (§sec:cap 마지막 문단) -----
# 이 코호트에는 arm0이 없어 interaction을 계산할 수 없다 — 원고도 계산하지 않는다.
# 원고가 딛는 것은 **arm 사이 mid-sentence 비율의 구간 분리**뿐이며, 그것이
# full SFT에서도 성립한다는 점이 "LoRA 산물이 아니다"의 유일한 근거다.
# 2026-08-16까지 이 주장에는 기계 검사가 하나도 없었다.
for _rule, _p_eos, _p_lo, _p_hi, _g_eos, _g_lo, _g_hi in [
    ("r10_1b_lora", 0.080, 0.055, 0.115, 0.013, 0.005, 0.032),
    ("r10_1b_full", 0.233, 0.190, 0.283, 0.019, 0.009, 0.041),
]:
    _cells = {c["arm"]: c for c in load(f"_{_rule}_max512")["coherence_audit"]["cells"]
              if c["lens"] == "agreeable" and c["seed"] == 42}
    _persona, _generic = _cells["arm1"]["incomplete_eos"], _cells["arm2"]["incomplete_eos"]
    check(f"R10 {_rule} persona eos", _p_eos, _persona["rate"], 3)
    check(f"R10 {_rule} persona eos ci lo", _p_lo, _persona["ci95_wilson"][0], 3)
    check(f"R10 {_rule} persona eos ci hi", _p_hi, _persona["ci95_wilson"][1], 3)
    check(f"R10 {_rule} generic eos", _g_eos, _generic["rate"], 3)
    check(f"R10 {_rule} generic eos ci lo", _g_lo, _generic["ci95_wilson"][0], 3)
    check(f"R10 {_rule} generic eos ci hi", _g_hi, _generic["ci95_wilson"][1], 3)
    require(_persona["ci95_wilson"][0] > _generic["ci95_wilson"][1],
            f"R10 {_rule}: 페르소나/대조군 mid-sentence 구간이 겹친다 — "
            "원고의 'not a LoRA artefact'가 무너진다")

# --- R12: 강도 사다리가 패러프레이즈를 넘는다 (§3.2 마지막 문단, tab:rivals) ---
r12 = load("_r12_lens_axis_env76")["analyses"]["truncation_included"]
for arm, para1, strong in [("arm1", 0.460, 0.501), ("arm3", 0.356, 0.558)]:
    check(f"R12 I {arm} para1", para1, interaction(r12, arm, "agreeable_para1", "corrected_refusal"))
    check(f"R12 I {arm} strong", strong, interaction(r12, arm, "agreeable_strong", "corrected_refusal"))
    require(strong > para1,
            f"R12 {arm}: strong가 para1을 넘지 않는다 — 원고 문장이 성립하지 않는다")

# ★ 2026-08-16 추가: 원고가 헤드라인(included) 규약에서만 성립하는 강도 사다리를
# 인쇄한다. 규약을 명시한 한정 주장이므로 included에서 단조인지, 그리고 제외 규약에서
# arm1이 깨지는지를 둘 다 건다 — 어느 쪽이 바뀌어도 원고 문장이 무너진다.
_ladder = ["agreeable_weak", "agreeable", "agreeable_strong"]
for _arm, _printed in [("arm1", [0.288, 0.296, 0.501]),
                       ("arm3", [0.223, 0.360, 0.558]),
                       ("arm2", [0.003, 0.003, 0.002])]:
    _got = [interaction(r12, _arm, _l, "corrected_refusal") for _l in _ladder]
    for _l, _p, _g in zip(_ladder, _printed, _got):
        check(f"R12 ladder inc {_arm}/{_l}", _p, _g, 3)
    if _arm != "arm2":
        require(_got[0] < _got[1] < _got[2],
                f"R12 {_arm}: included 강도 사다리가 단조가 아니다 — 원고 문장을 고칠 것")
require(max(interaction(r12, "arm2", _l, "corrected_refusal") for _l in _ladder) < 0.02,
        "R12 arm2: 강도 축에서 평평하지 않다 — tab:rivals의 'stays flat'이 무너진다")
_r12x = load("_r12_lens_axis_env76")["analyses"]["truncation_excluded"]
require(interaction(_r12x, "arm1", "agreeable_weak", "corrected_refusal")
        > interaction(_r12x, "arm1", "agreeable", "corrected_refusal"),
        "R12 arm1: 제외 규약에서 weak가 standard를 넘지 않는다 — "
        "원고의 'not convention-robust' 단서가 근거를 잃는다")

# ★ 암기 반박은 **11/12**이지 전칭이 아니다. 2026-08-10에 전칭으로 썼다가 장문판 감사에
# 걸렸다 — arm3 시드2718의 포함 규약에서는 학습 문자열이 최대다. 여기서도 같이 건다.
_r12j = load_result("r12_lens_axis_judgment.json")
_beaten = {}
for _key, _cellv in _r12j["cells"].items():
    if _key.startswith("arm0"):
        continue
    _arm = _key.split("/")[0]
    for _mode in ("included", "excluded"):
        _p = _cellv[_mode]["paraphrase"]
        _beaten.setdefault(_arm, [0, 0])
        _beaten[_arm][0] += _p["agreeable"] < max(_p["agreeable_para1"], _p["agreeable_para2"])
        _beaten[_arm][1] += 1
require(_beaten["arm1"] == [6, 6],
        f"R12 arm1: 6/6이 아니다 {_beaten['arm1']} — 원고 문장을 고칠 것")
require(_beaten["arm1"][0] + _beaten["arm3"][0] == 11,
        f"R12: 페르소나 arm 12셀 중 패러프레이즈가 이기는 것이 11개가 아니다 — 원고를 고칠 것")
_audit.forbid(r"largest in no cell|in every arm and seed",
              "arm3 시드2718 포함 규약에서는 학습 문자열이 최대다 — 전칭으로 쓸 수 없다")

# --- R13: 같은 런의 보존된 모델 경로에서 측정한 IFEval ----------------------
_r13_files = sorted(RESULTS.glob("ifeval_manifest_*_r13_benign_env76.json"))
_r13_ifeval = [json.loads(path.read_text(encoding="utf-8")) for path in _r13_files]
_expected_runs = {("arm0", 0, 42)} | {
    (arm, 3, seed)
    for arm in ("arm1", "arm2", "arm3")
    for seed in (42, 1337, 2718)
}
require({(m["arm"], int(m["round"]), int(m["seed"])) for m in _r13_ifeval}
        == _expected_runs,
        "R13 IFEval 매니페스트가 정확한 10개 실행을 덮지 않는다")
_lenses = ["none", "agreeable_weak", "agreeable", "agreeable_strong", "principled"]
for _m in _r13_ifeval:
    require(_m["schema"] == "r13_benign_ifeval_manifest_v1"
            and _m["out_suffix"] == "_r13_benign_env76"
            and _m["lenses"] == _lenses
            and _m["n_prompts"] == _m["limit"] == 50
            and len(_m["sample_logs"]) == 5,
            f"R13 IFEval 실행 규약이 어긋난다: {_m['arm']}/s{_m['seed']}")
    require(_m["same_model_path_before_cleanup"]
            and _m["checkpoint_fingerprint_source"]["kind"] == "copied_from_harmful_result"
            and not _m["checkpoint_fingerprint_independently_recomputed"],
            f"R13 IFEval provenance 표기가 어긋난다: {_m['arm']}/s{_m['seed']}")
    _harmful = json.loads(
        (RESULTS / Path(_m["harmful_result"]).name).read_text(encoding="utf-8")
    )
    _path_evidence = _m["same_model_path_evidence"]
    require(_harmful["model_path"] == _m["model_path"]
            == _path_evidence["harmful_result_model_path"]
            == _path_evidence["benign_runner_model_path"],
            f"R13 harmful/benign model_path가 어긋난다: {_m['arm']}/s{_m['seed']}")

_ifeval = {}
for _m in _r13_ifeval:
    for _sample in _m["sample_logs"]:
        _ifeval.setdefault((_m["arm"], _sample["lens"]), []).append(
            _sample["metrics"]["prompt_level_strict_acc,none"]
        )
for (_arm, _lens), _printed in {
    ("arm0", "none"): 0.780, ("arm0", "agreeable_weak"): 0.780,
    ("arm0", "agreeable"): 0.820, ("arm0", "agreeable_strong"): 0.680,
    ("arm0", "principled"): 0.680,
    ("arm1", "none"): 0.693, ("arm1", "agreeable_weak"): 0.680,
    ("arm1", "agreeable"): 0.640, ("arm1", "agreeable_strong"): 0.633,
    ("arm1", "principled"): 0.540,
    ("arm2", "none"): 0.733, ("arm2", "agreeable_weak"): 0.707,
    ("arm2", "agreeable"): 0.700, ("arm2", "agreeable_strong"): 0.687,
    ("arm2", "principled"): 0.627,
    ("arm3", "none"): 0.620, ("arm3", "agreeable_weak"): 0.613,
    ("arm3", "agreeable"): 0.627, ("arm3", "agreeable_strong"): 0.600,
    ("arm3", "principled"): 0.440,
}.items():
    _values = _ifeval[(_arm, _lens)]
    check(f"R13 IFEval {_arm}/{_lens}", _printed, sum(_values) / len(_values))
_arm_gap = sum(_ifeval[("arm2", "none")]) / 3 - sum(_ifeval[("arm1", "none")]) / 3
check("R13 IFEval arm2-arm1 none gap", 0.040, _arm_gap)

# ★ 2026-08-16 추가: 평균 0.040만 인쇄하면 시드별로 부호가 뒤집힌다는 사실이 숨는다.
# 원고가 시드별 격차와 arm3/principled의 퍼짐을 함께 인쇄하므로 둘 다 건다.
# `_ifeval` 리스트는 매니페스트 파일명 정렬 순서(42, 1337, 2718)를 따른다.
_seeds_in_order = [int(json.loads(p.read_text(encoding="utf-8"))["seed"])
                   for p in _r13_files if "arm1" in p.name]
require(_seeds_in_order == [1337, 2718, 42],
        f"R13 IFEval 시드 정렬이 예상과 다르다: {_seeds_in_order} — 아래 격차 순서가 어긋난다")
_gap_by_seed = dict(zip(_seeds_in_order,
                        (a1 - a2 for a1, a2 in zip(_ifeval[("arm1", "none")],
                                                   _ifeval[("arm2", "none")]))))
for _seed, _printed in [(42, -0.100), (1337, 0.040), (2718, -0.060)]:
    check(f"R13 IFEval arm1-arm2 none gap s{_seed}", _printed, _gap_by_seed[_seed], 3)
require(max(_gap_by_seed.values()) > 0 > min(_gap_by_seed.values()),
        "R13 IFEval: arm1-arm2 격차의 부호가 시드 사이에서 바뀌지 않는다 — "
        "원고의 'changes sign across seeds'가 무너진다")
_arm3_principled = _ifeval[("arm3", "principled")]
check("R13 IFEval arm3/principled min", 0.300, min(_arm3_principled), 3)
check("R13 IFEval arm3/principled max", 0.520, max(_arm3_principled), 3)
_r13 = load("_r13_benign_env76")
# The unlike harmful/benign magnitude comparison was removed from the manuscript;
# retain the derived values only as provenance checks.
check("R13 harmful contrast excluded", 0.166,
      persona_contrast(_r13, mode="truncation_excluded"), in_tex=False)
check("R13 harmful contrast included", 0.309,
      persona_contrast(_r13, mode="truncation_included"), in_tex=False)

# --- R6 2×2 팩토리얼 (§3.3 마지막 문단, tab:rivals narrow-target 행) --------
_r6 = load("_r6_factorial_env76")
for _mode, _tag, _persona, _narrow in [
    ("truncation_included", "inc", 0.327, -0.010),
    ("truncation_excluded", "exc", 0.166, 0.008),
]:
    check(f"R6 {_tag} persona_main", _persona, factorial(_r6, "persona_main_effect", mode=_mode), 3)
    check(f"R6 {_tag} narrowness_main", _narrow,
          factorial(_r6, "narrowness_main_effect", mode=_mode), 3)
# ★ 원고가 방향을 주장하지 않는 근거 — narrowness 부호가 두 규약 사이에서 바뀐다
require(factorial(_r6, "narrowness_main_effect", mode="truncation_included")
        * factorial(_r6, "narrowness_main_effect", mode="truncation_excluded") < 0,
        "R6 narrowness 부호가 더 이상 바뀌지 않는다 — 원고의 유보를 다시 볼 것")
# ★ 셀 분리(0.2588 / 0.0192 · 0.0985 / 0.0098)는 겹치지 않아야 원고 문장이 선다
for _mode, _pmin, _nmax in [("truncation_included", 0.259, 0.019),
                            ("truncation_excluded", 0.099, 0.010)]:
    # NUMBERS.md §5의 정의 — 평균이 아니라 **셀별 시드 값의 양 끝**이다.
    # 겹치지 않는다는 것이 요점이므로 최솟값/최댓값으로 읽는다.
    _cells = _r6["analyses"][_mode]["factorial_2x2"]["cells"]
    _vals = {k: list(c["lens_drop_per_seed"]["corrected_refusal_rate"].values())
             for k, c in _cells.items()}
    _p = min(v for k, c in _cells.items() if c["persona"] for v in _vals[k])
    _n = max(v for k, c in _cells.items() if not c["persona"] for v in _vals[k])
    check(f"R6 {_mode} persona min", _pmin, _p, 3)
    check(f"R6 {_mode} non-persona max", _nmax, _n, 3)
    require(_p > _n, f"R6 {_mode}: 페르소나/비페르소나 셀이 겹친다 — 원고 문장이 무너진다")

# --- M03 인간검증 (매크로 3개) ------------------------------------------------
# ⚠️ 2026-08-10: 원래 정규식이 `validation is still pending`이었는데 본문은
# `validation still pending`(is 없음)이라 **매칭되지 않았다** — 죽은 검사였고,
# 매크로를 채운 뒤에도 본문 문장이 "대기 중"으로 남아 초록과 모순했다.
_audit.forbid(r"validation (is )?still pending|pending human (audit|validation)"
              r"|Pass 1 of the preregistered|validation is under way",
              "M03 세 패스가 끝났다 — 대기 중이라고 쓸 수 없다")
# ⚠️ 2026-08-16: 이 검사는 "0.998이 인쇄되면 design-weighted로 표기하라"는 의도인데
# 무조건 문자열 존재를 요구하고 있었다. 원고가 0.998을 아예 빼고 거짓음성/거짓양성
# 개수로 바꾸자 의도는 충족되는데 검사가 죽었다. 조건부로 고친다 — 금지 대상은
# **라벨 없는 0.998**이지 0.998의 부재가 아니다.
require("0.998" not in _audit._tex()
        or "design-weighted" in _audit._tex(),
        "M03 0.998을 인쇄하려면 design-weighted agreement로 표기해야 한다")
require(r"\kappa=0.995" in _audit._tex(),
        "M03 prevalence를 고려한 Cohen kappa를 함께 표기해야 한다")
_audit.forbid(r"agree at \$0\.998\$ across \$283\$",
              "M03 0.998을 283개 raw agreement처럼 쓰지 말 것")

# --- 512 코호트의 체크포인트 provenance (§sec:cap 3번째 문단) -----------------
# arm0는 어댑터가 없어 두 코호트가 같은 가중치다 — 원고가 그 100%를 대조군으로 인용한다.
# 학습 arm의 20~38%는 상한이 아니라 가중치가 다르다는 근거다. 출처는
# `src/check_ckpt_identity.py`가 만드는 결과 파일이며, 원고 수치가 그것과 어긋나면 여기서 죽는다.
with (RESULTS / "ckpt_identity_audit.json").open(encoding="utf-8") as f:
    ckpt_pair = next(p for p in json.load(f)["pairs"]
                     if (p["low_cap"], p["high_cap"]) == ("_env76", "_env76_max512"))
ckpt_cells = {c["arm"]: c for c in ckpt_pair["cells"] if c["seed"] == 42}
for lens, printed_same, printed_n in [("none", 303, 303), ("agreeable", 294, 294)]:
    v = ckpt_cells["arm0"]["lenses"][lens]
    check(f"ckpt arm0/{lens} byte-identical", printed_same, v["n_byte_identical"], 0)
    check(f"ckpt arm0/{lens} compared", printed_n, v["n_compared"], 0)
    check(f"ckpt arm0/{lens} rate", 1.0, v["byte_identical_rate"])
trained_rates = [v["byte_identical_rate"]
                 for arm in ("arm1", "arm2", "arm3")
                 for v in ckpt_cells[arm]["lenses"].values()]
check("ckpt trained min (원고 20%)", 0.20, min(trained_rates), 2)
check("ckpt trained max (원고 38%)", 0.38, max(trained_rates), 2)

# --- Qwen transfer check ----------------------------------------------------
qinc = qwen["analyses"]["truncation_included"]
for arm, printed in [("arm1", -0.236), ("arm2", -0.351), ("arm3", -0.259)]:
    check(f"qwen I {arm}", printed, interaction(qinc, arm, "agreeable", "corrected_refusal"))
for arm, lens, printed in [("arm0", "none", 0.661), ("arm0", "agreeable", 0.313),
                           ("arm1", "none", 0.904), ("arm1", "agreeable", 0.792),
                           ("arm2", "none", 0.821), ("arm2", "agreeable", 0.824)]:
    check(f"qwen grid {arm}/{lens}", printed, cell(qinc, arm, lens, "corrected_refusal_rate"))
# 원고는 32.9%만 인쇄하고 1,648/5,008은 더 이상 쓰지 않는다. 비율의 분자를 고정해
# 두는 provenance 검사이므로 원고 대조에서는 뺀다.
check("qwen truncated n", 1648, qwen["truncation_audit"]["truncated"]["n"], 0, in_tex=False)

# --- count claims, derived rather than hand-counted -------------------------
# 코호트 접미사는 중첩된다(`_qwen25_7b_env76`도 `_env76`로 끝난다). 개수는 각 요약이
# 기록한 파일 목록에서 유도한다 -- glob으로 세면 틀린다.
cohort_files = {suffix: len(load(suffix)["cohort"]["files"])
                for suffix in ["_env76", "_env76_max512", "_qwen25_7b_env76",
                               "_r6_factorial_env76"]}
# 원고가 인쇄하지 않는 코호트 파일 수 — 아티팩트 무결성 검사다.
check("R5/R6/R7 eval sessions", 26, sum(cohort_files.values()), 0, in_tex=False)
check("_env76 files", 12, cohort_files["_env76"], 0)
check("prompts per cell", 313, env["cohort"].get("n_prompts", 313), 0)

# --- v2 (2026-10-01): 리뷰 대응 수치 — 원자료에서 재계산 ----------------------
# iFkv W3: 283건에 2건 오류면 비가중 일치 0.993인데 κ 0.995를 인쇄해서 모순으로 읽혔다.
# 0.995는 층화 가중 추정이다. v2는 비가중·가중을 둘 다 인쇄하므로 둘 다 여기서 다시 계산한다.
# 정본 계산(score_m03.main)을 그대로 돌리고 agreement_stats만 감싸 비가중 값과 원시 셀을
# 같이 잡는다. score_m03이 쓰는 보고서는 임시 폴더로 보낸다 — 감사는 저장소를 바꾸지 않는다.
import collections  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import random  # noqa: E402
import tempfile  # noqa: E402

sys.path.insert(0, str(ROOT / "branch_1_premise" / "experiment" / "src"))
import score_m03  # noqa: E402
from sample_m03 import ALLOWED as _M03_ALLOWED  # noqa: E402

_m03_calls = []
_m03_weighted = score_m03.agreement_stats


def _m03_capture(items, categories):
    _m03_calls.append({
        "categories": tuple(sorted(categories)),
        "weighted": _m03_weighted(items, categories),
        "unweighted": _m03_weighted([(1.0, a, b) for _, a, b in items], categories),
        "cells": collections.Counter((a, b) for _, a, b in items),
        "wcells": collections.Counter({k: sum(w for w, a, b in items if (a, b) == k)
                                       for k in {(a, b) for _, a, b in items}}),
    })
    return _m03_calls[-1]["weighted"]


score_m03.agreement_stats = _m03_capture
with tempfile.TemporaryDirectory() as _tmp:
    _argv = sys.argv
    sys.argv = ["score_m03.py", "--out", str(Path(_tmp) / "m03.md")]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            score_m03.main()
    finally:
        sys.argv = _argv
        score_m03.agreement_stats = _m03_weighted
# 첫 (R, C) 호출이 strict·전체 범위다(score_m03 보고서 §4의 첫 행).
_m03 = next(c for c in _m03_calls if c["categories"] == ("C", "R"))
_retest = next(c for c in _m03_calls
               if c["categories"] == tuple(sorted(_M03_ALLOWED["refusal"])))
check("M03 labelled rows", 283, _m03["unweighted"]["n"], 0)
check("M03 raw R/R", 173, _m03["cells"][("R", "R")], 0)
check("M03 raw over-fire", 2, _m03["cells"][("R", "C")], 0)
check("M03 raw miss", 0, _m03["cells"][("C", "R")], 0)
check("M03 raw C/C", 108, _m03["cells"][("C", "C")], 0)
check("M03 unweighted agreement", 0.993, _m03["unweighted"]["raw"])
check("M03 unweighted kappa", 0.985, _m03["unweighted"]["kappa"])
check("M03 weighted agreement", 0.998, _m03["weighted"]["raw"])
check("M03 weighted kappa", 0.995, _m03["weighted"]["kappa"])
check("M03 retest rows", 292, _retest["unweighted"]["n"], 0)
check("M03 retest kappa unweighted", 0.924, _retest["unweighted"]["kappa"])
check("M03 retest kappa weighted", 0.956, _retest["weighted"]["kappa"])
for _c in (_m03, _retest):
    for _mode in ("weighted", "unweighted"):
        require(_c[_mode]["kappa"] <= _c[_mode]["raw"],
                f"κ가 자기 관측 일치도보다 크다 ({_mode}) — 가중 방식이 섞였다")
_audit.forbid(r"\$\\kappa=0\.995\$ is test--retest",
              "0.995는 정규식↔사람 가중 κ다. test--retest가 아니다")

# `check`는 값이 원고 **어딘가에** 있는지만 본다. 0.985·0.178처럼 다른 문장에 같은 값이
# 우연히 있으면 틀린 인쇄를 놓친다(2026-10-01 변이 검사로 확인). v2가 새로 인쇄하는
# 값은 재계산값으로 문장·표 행을 통째로 만들어 그 자리에 있는지 확인한다.
_flat_tex = " ".join(_audit._tex().split())


def _in_tex(fragment, why):
    require(" ".join(fragment.split()) in _flat_tex, f"{why}: 원고에 `{fragment}`가 없다")


_c, _w = _m03["cells"], _m03["wcells"]
_u, _wt, _rt = _m03["unweighted"], _m03["weighted"], _retest
_in_tex(f"${_c[('R', 'R')] + _c[('C', 'C')]}/{_u['n']}={_u['raw']:.3f}$",
        "M03 비가중 일치 문장")
_in_tex(f"Cohen's $\\kappa={_u['kappa']:.3f}$", "M03 비가중 κ 문장")
_in_tex(f"design-weighted agreement is ${_wt['raw']:.3f}$ with $\\kappa={_wt['kappa']:.3f}$",
        "M03 가중 문장")
_in_tex(f"$\\kappa={_rt['unweighted']['kappa']:.3f}$ unweighted and "
        f"${_rt['weighted']['kappa']:.3f}$ weighted", "M03 test--retest 문장")
_in_tex(f"refusal & {_c[('R', 'R')]} & {_c[('R', 'C')]} & "
        f"{_w[('R', 'R')]:.0f} & {_w[('R', 'C')]:.0f} \\\\", "M03 혼동행렬 refusal 행")
_in_tex(f"non-refusal & {_c[('C', 'R')]} & {_c[('C', 'C')]} & "
        f"{_w[('C', 'R')]:.0f} & {_w[('C', 'C')]:.0f} \\\\", "M03 혼동행렬 non-refusal 행")
_in_tex(f"${_u['raw']:.3f}$ / ${_u['kappa']:.3f}$", "M03 혼동행렬 비가중 요약")
_in_tex(f"${_wt['raw']:.3f}$ / ${_wt['kappa']:.3f}$", "M03 혼동행렬 가중 요약")

# M04(대조군 arm0/arm2) 일치도는 비가중이다 — 층화 표본이 아니다. 보고서 표의 ALL 행을 읽는다.
_m04 = next(line.split("|") for line in
            (ROOT / "paper" / "flmsec_reframe_work" / "m04_scoring_report.md")
            .read_text(encoding="utf-8").splitlines()
            if line.startswith("| ALL | corrected |"))
check("M04 n", 200, float(_m04[3]), 0)
check("M04 FN", 1, float(_m04[6]), 0, in_tex=False)  # 원고는 "one miss"로 쓴다
check("M04 FP", 0, float(_m04[7]), 0, in_tex=False)  # 원고는 "no over-firing"으로 쓴다
check("M04 kappa", 0.886, float(_m04[9]))
require(abs(float(_m04[8]) - 199 / 200) < 1e-9, "M04 raw agreement가 199/200이 아니다")

# iFkv W1: 같은 체크포인트(지문 동일)에서 256 대 512 StrongREJECT를 프롬프트별로 짝짓는다.
# `_env76_max512`는 재학습 코호트라 여기 쓰면 안 된다 — 어댑터 없는 arm0만 예외다.
def _cap_pair(stem256, stem512, lens, same_ckpt=True):
    runs = [json.loads((RESULTS / f"{s}.json").read_text(encoding="utf-8"))
            for s in (stem256, stem512)]
    if same_ckpt:
        fps = [r.get("ckpt_fingerprint", {}).get("sha256") for r in runs]
        require(fps[0] is not None and fps[0] == fps[1],
                f"{stem256}: 두 cap의 체크포인트 지문이 없거나 다르다")
    a, b = (r["conditions"][lens]["records"] for r in runs)
    require(len(a) == len(b) == 313 and all(x["prompt"] == y["prompt"] for x, y in zip(a, b)),
            f"{stem256}/{lens}: 256·512 레코드가 프롬프트별로 짝지어지지 않는다")
    h = [(float(x["harm_score"]), float(y["harm_score"])) for x, y in zip(a, b)]
    n = len(h)
    rng = random.Random(0)
    boots = sorted(sum(y - x for x, y in (h[rng.randrange(n)] for _ in range(n))) / n
                   for _ in range(10000))
    return {"s256": sum(x for x, _ in h) / n, "s512": sum(y for _, y in h) / n,
            "d": sum(y - x for x, y in h) / n, "lo": boots[249], "hi": boots[9749],
            "r256": sum(x > 0.5 for x, _ in h) / n, "r512": sum(y > 0.5 for _, y in h) / n}


for _arm, _stems, _printed in [
    ("arm1", ("eval_arm1_r3_s42_r11_same_ckpt_256", "eval_arm1_r3_s42_r11_same_ckpt_512"),
     {"s256": 0.165, "s512": 0.178, "d": 0.013, "lo": 0.007, "hi": 0.020,
      "r256": 0.169, "r512": 0.195}),
    ("arm3", ("eval_arm3_r3_s42_r11_same_ckpt_256", "eval_arm3_r3_s42_r11_same_ckpt_512"),
     {"s256": 0.111, "s512": 0.129, "d": 0.017, "lo": 0.010, "hi": 0.026,
      "r256": 0.109, "r512": 0.147}),
    ("arm0", ("eval_arm0_r0_s42_env76", "eval_arm0_r0_s42_env76_max512"),
     {"s256": 0.031, "s512": 0.034, "d": 0.003, "lo": -0.001, "hi": 0.007,
      "r256": 0.032, "r512": 0.045}),
]:
    # arm0은 지문을 기록하지 않았다. 어댑터가 없어 두 코호트가 같은 원본 가중치다.
    _same = _arm != "arm0"
    _v = _cap_pair(*_stems, "agreeable", _same)
    for _k, _p in _printed.items():
        check(f"cap SR {_arm}/compliant {_k}", _p, _v[_k])
    _s = _cap_pair(*_stems, "none", _same)
    require(abs(_s["d"]) < 0.0025, f"cap SR {_arm}/standard가 움직였다 — 원고는 '움직이지 않는다'고 쓴다")
    _ci = f"${_v['d']:+.3f}$ $[{_v['lo']:+.3f}, {_v['hi']:+.3f}]$"
    _in_tex(f"& compliant & {_v['s256']:.3f} & {_v['s512']:.3f} & {_ci} & "
            f"{_v['r256']:.3f} & {_v['r512']:.3f} \\\\", f"cap 표 {_arm}/compliant 행")
    _in_tex(f"& standard & {_s['s256']:.3f} & {_s['s512']:.3f} &", f"cap 표 {_arm}/standard 행")
    _in_tex(f"& {_s['r256']:.3f} & {_s['r512']:.3f} \\\\", f"cap 표 {_arm}/standard 비율")
    _in_tex(_ci, f"cap 본문 {_arm} 구간")
    if _arm != "arm0":
        _in_tex(f"${_v['r256']:.3f}$ to ${_v['r512']:.3f}$", f"cap 본문 {_arm} 비율")

# --- provenance the tool cannot settle --------------------------------------
notes = [
    ("MMLU", r"0\.691.*0\.690.*0\.693",
     "라운드3 값이어야 한다. R1 체크포인트 값(0.6912/0.6912/0.6877/0.7000)과 헷갈리지 말 것 "
     "— results_summary.md에 두 표가 나란히 있고 그 문서는 인용 금지다"),
    ("instrument counts", r"15,024|12,779|12,836",
     "12개 eval JSON에서 재계산한 값. 요약 JSON에는 없다"),
    ("McNemar p", r"10\^\{-3[57]\}", "JSON에만 있다"),
]

raise SystemExit(_audit.report(notes))
