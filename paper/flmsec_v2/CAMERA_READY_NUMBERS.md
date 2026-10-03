# Camera-ready recomputations (2026-10-01)

Scripts: session scratchpad `c1_c3.py`, `c1b.py`, `c6.py`, `c7.py`. Each first reproduces the
published values (I = 0.306 / 0.381 / −0.012; flips 24/33/0/0; pattern-0 fires 12,779 → 12,836)
and asserts them before printing anything new. Inputs are read-only: `branch_1_premise/experiment/results/*_env76*.json`,
`branch_1_premise/data/gen_data_backup.zip`.

## C1 Apostrophes
- Training responses (all rounds, 410/414/411 rows): **0 U+2019 in any arm**. Rows containing "I can't" (ASCII): arm1 0, arm2 6, arm3 4.
  → "persona arms copy straight apostrophe from their data" is NOT supported: arm 2's data are equally ASCII-only.
- Frozen grid (3,756 responses per arm, 4 lenses × 3 seeds): responses containing "I can’t" (U+2019):
  arm0 39, arm1 0, arm2 57, arm3 0. Responses containing ASCII "I can't": 3,519 / 2,844 / 3,606 / 2,761.
- Flips under normalization: 24 / 0 / 33 / 0.

## C3 Frozen grid pre-correction (seed means)
- arm0 standard 0.933→0.949, compliant 0.939→0.946, paraphrase 0.895→0.898
- arm2 standard 0.939→0.967, compliant 0.971→0.976, paraphrase 0.933→0.936
- I2 compliant −0.026 → −0.012; shift +0.014, 95% CI [−0.003, +0.031] (prompt bootstrap, 10k, seeds averaged per prompt)
- per seed I2 raw→norm: s42 −0.054→−0.029; s1337 −0.013→−0.006; s2718 −0.010→0.000
- pattern fires raw [12779,2,15,0,0,97,0,0], normalized [12836,2,15,0,0,97,0,0]

## C2 Legacy mixed-environment grid (seed 42, `corrected_grid.json`)
- env: arm0 & arm2 files have no env stamp (older env); arm1 & arm3 carry the env76 stamp → "mixed".
- arm0: std 0.933→0.949, compl 0.939→0.946; arm2: std 0.866→0.968, compl 0.942→0.942
- arm1: std 0.981, compl 0.633; arm3: std 0.987, compl 0.489 (no flips)
- I (compliant) stored→normalized: arm1 0.355→0.345, arm2 −0.070→+0.022, arm3 0.505→0.495
- I (paraphrase): arm1 0.457→0.444, arm2 −0.058→+0.032, arm3 0.550→0.537

## C4 Validation cohorts
- First validation (M03) sources: `eval_arm1_r3_s42.json`, `eval_arm3_r3_s42.json` = LEGACY cohort.
  Its paired effects 0.348 / 0.498 equal the legacy instrument's standard→compliant drops exactly.
  Headline frozen seed-42 drops are 0.371 / 0.393.
- Second validation (M04): arms 0/2, legacy seed 42; 39 flipped = 7 (arm0) + 32 (arm2/standard).
- Byte-identical legacy vs frozen seed 42 (std/compl): arm0 313/313, 313/313; arm1 77/87; arm2 58/83; arm3 94/47.
- Dropped pairs (conf<2 or X): arm1 8 (P3 1, P4 7), arm3 1 (P4). Fill-in envelope arm1 [0.270, 0.348], arm3 [0.478, 0.498].

## C5 2×2 and seven-lens
- 2×2 "cells" 0.259 / 0.019 are per-seed raw standard-minus-compliant drops (arm1 s1337; arm5 s42), not I_a.
  Persona main per seed: 0.369 [0.319,0.420], 0.324 [0.276,0.374], 0.288 [0.240,0.337].
- Seven-lens grid `_r12_lens_axis_env76` is a separately retrained cohort (arm1 s42: 224/313 std, 152/313 compl byte-identical with frozen).

## C6 Truncation
- 256 headline seed 42 four lenses: 549 flagged, 487 hit the cap (88.7%), 62 ended by EOS.
- 512 retrained cohort seed 42 compliant, mid-sentence EOS (flagged & not cap): arm0 0/313, arm1 48 (0.153), **arm2 3 (0.010 [0.003, 0.028])**, arm3 87 (0.278).

## C7 Intervals (prompt-level percentile bootstrap, 10k; seed-mean contrast resampled by prompt — no seed variance)
- compliant: I1 0.306 [0.252, 0.362]; I3 0.381 [0.327, 0.438]; I2 −0.012 [−0.040, 0.018]
- per seed I1: 0.367 [0.307,0.431], 0.256 [0.201,0.310], 0.294 [0.236,0.351]; I3: 0.390 [0.332,0.450], 0.438 [0.377,0.498], 0.316 [0.262,0.374]
- arm1−arm2 0.317 [0.269, 0.367]; arm3−arm2 0.393 [0.342, 0.445]
- paraphrase: I1 0.478 [0.419,0.538]; I3 0.383 [0.330,0.439]; I2 −0.020 [−0.056,0.016]; arm1−arm2 0.498 [0.446,0.552]; arm3−arm2 0.404 [0.355,0.454]
- principled-lens interactions: arm1 0.049, arm2 0.009, arm3 0.147 (arm3 0.978→0.869)

## C8/C9 Setup
- 1B: unsloth/Llama-3.2-1B-Instruct; LoRA LR 2e-4; full SFT LR 1e-5.
- QLoRA via Unsloth, 4-bit base, r16 α32 dropout 0.05 on q/k/v/o/gate/up/down; LR 2e-4 cosine, warmup 0.03; batch 2 × accum 4; 2 epochs per round; max seq 1024.
- Data generation: base model, T = 0.7, ≤256 new tokens; arm 1/3 persona as system prompt, arm 2 none; self-check verdict T = 0.
- Evaluation: greedy, 4-bit, ≤256 new tokens; StrongREJECT finetuned evaluator (qylu4156/strongreject-15k-v1, Gemma-2B).
- Env: unsloth 2026.7.6, torch 2.10.0+cu128, transformers 4.56.2.
