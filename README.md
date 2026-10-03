# Treatment-Correlated Instrument Error in Post-Fine-Tuning Safety Evaluation — release

Code, configurations, preregistration documents, label sheets and per-record results
(with prompt and response text removed) for the FLMSec 2026 camera-ready paper.

## Layout
- `paper/flmsec_v2/` — manuscript source and PDF (`main.pdf`), table and figure
  generators, and two number audits; `paper/shared/` holds their common helper
- `branch_1_premise/experiment/src/`, `configs/`, `tests/`, `nb_*.ipynb` — training,
  evaluation, scoring and audit code, and the Kaggle notebooks that ran it
- `branch_1_premise/experiment/preregistration/` — design documents committed before
  their results; `PREREG_INDEX.md` lists the commits
- `branch_1_premise/experiment/annotation/` — labelling codebook and label sheets
- `branch_1_premise/experiment/results/` — every result file, text fields removed

## Not included
Prompt and response text. The forbidden prompts come from a source dataset whose licence
does not permit redistribution, and the generations are keyed to those prompts. Every
number derived from them (`refused_regex`, `harm_score`, `finish_reason`, `hit_cap`,
token counts, judge labels) ships. To restore the text layer, obtain StrongREJECT and
re-run `src/eval_refusal.py`. Human-reading files with full text are likewise excluded.

## Licence
Code and documents in this repository are released under the MIT License (`LICENSE`).
Result files derived from third-party datasets and models (StrongREJECT prompts, model
generations, judge outputs) remain subject to those sources' own licences and terms.

## Camera-ready checks
The R15-R17 campaigns (`preregistration/R15_R17_CAMERA_READY.md`: learned-judge rescoring,
Qwen2.5-7B at 512 tokens, style-swap) were preregistered before running; their result files
are in `results/` (`judge_*.json`, `*_qwen25_7b_max512.json`, `*_r17_styleswap_env76.json`,
`rewrite_arm*_report.json` without the rewritten example texts) and `PREREG_INDEX.md` shows
the first result commit. The document's post-hoc section records one change to the
judge-error test. The rewritten training data are not included, for the same reason as the
generations.

## Checking the numbers
Python 3.10 or later. The audits use only the standard library, except that the figure
check regenerates Figure 1 with `reportlab` (`pip install reportlab`). Run from the
repository root:

    python paper/flmsec_v2/audit_camera_ready.py
    python paper/flmsec_v2/audit_manuscript_numbers.py

Both audits recount from response text (apostrophe counts, truncation heuristic, paired
prompts), so they need the restored text layer and the training-data archive
(`branch_1_premise/data/gen_data_backup.zip`, also withheld). Without them each audit stops
at once with a message saying so. The shipped per-record fields can be summed directly.

## Citation
    @inproceedings{ahn2026treatment,
      title     = {Treatment-Correlated Instrument Error in Post-Fine-Tuning Safety
                   Evaluation: A Single-Family Case Study},
      author    = {Ahn, Minsoo},
      booktitle = {NeurIPS 2026 Workshop on Foundations of Language Model Security (FLMSec)},
      year      = {2026}
    }
