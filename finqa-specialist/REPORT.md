# REPORT — FinQA specialist post-training sprint (2026-11-26)

> Research prototype, not financial advice. All numbers below were measured in this session unless
> explicitly labeled *estimate* or *pending*. No results were fabricated; frontier baselines are pending.

## 0. TL;DR
* **Environment reality:** CPU-only container (4 vCPU, 15 GB RAM, no GPU), no GPU-provider or frontier-API
  credentials, $0 authorized spend. Everything requiring a GPU or paid API was **prepared, not run**.
* **Built and verified:** canonical data pipeline (FinQA / TAT-QA / ConvFinQA / FinanceBench + a fresh
  SEC-XBRL set), restricted DSL interpreter that reproduces the official FinQA evaluator on **all** gold programs
  (train 6,251 · dev 883 · test 1,147), numeric answer matcher, resumable/cached evaluation runner with
  local + API adapters and budget guards, TRL SFT/GRPO scripts, verifiable reward with adversarial tests,
  verified synthetic generator, cloud launch kit. 82 tests pass.
* **Measured (small-model CPU proxy, FinQA dev val100):** base Qwen3.5-0.8B = **6.0% [2.0, 11.0]** raw-answer
  accuracy; **1.0%** when its own program is executed. SFT results: see §4 (filled when the CPU queue finishes).
* **Not answered yet:** the headline question (open 9B vs frontier at lower cost) needs the GPU run and API
  baselines — exact commands and cost estimates in `NEXT_STEPS.md`.

## 1. Setup and provenance
See `docs/DECISIONS.md` for model revisions, licenses, environment, and price sources.
* Models: `Qwen/Qwen3.5-0.8B@2fc0636` (CPU proxy), `Qwen/Qwen3.5-2B@15852e8` (CPU baseline),
  `Qwen/Qwen3.5-9B@c202236` (GPU target, prepared), `Qwen/Qwen3.8-27B@3756181` (strong open baseline, prepared).
  All Apache-2.0. `transformers 5.24.1`, `trl 1.19.0`, `peft 0.22.0`, `torch 2.9.1` (CPU).
* Data (official splits preserved; hashes in `results/data_manifest.json`):

| set | n | use |
|---|---|---|
| FinQA train | 6,194 (57 exact duplicates removed; 30 with a table shared verbatim with dev/test excluded from SFT) | training |
| FinQA dev | 883 | validation (smoke20 ⊂ val100 ⊂ val300 fixed subsets; select60 disjoint, for checkpoint selection) |
| FinQA test | 1,147 | **held out, never evaluated** (runner refuses without `--final-test`) |
| TAT-QA train/dev/test (numeric subset) | 8,475 / 1,073 / 1,054 | secondary (ingested, not trained tonight) |
| ConvFinQA train/dev | 10,884 / 1,490 | leakage analysis only (0 train documents overlap FinQA dev/test) |
| FinanceBench open-source | 150 (52 numerically scorable) | external eval only; CC BY-NC 4.0 |
| SEC-XBRL fresh (templated) | 303 items, 40 companies, 10-Ks filed 2026-01..2026-11 | fresh external eval |
| Synthetic pilot | 1,600 items / 400 docs (1,412 train, 188 synth-dev) | ablation |

* Leakage checks: FinQA splits are document-disjoint (0 shared filenames) and question-disjoint; ConvFinQA
  train documents ⊄ FinQA dev/test; all eval subsets are dev-only; SFT train ids disjoint from dev/test (tests in
  `tests/test_split_leakage.py`). Frontier models' training corpora are unknown, so no contamination-free claim is
  made for them; the SEC set (filings after most cutoffs) is the closest to fresh.
* Label noise: in FinQA dev, 152/883 answer strings fail the lenient match against `exe_ans`; 72 are pure rounding,
  **~68 (7.7%) are material disagreements** (program/label errors). Gold = `exe_ans` (official).

## 2. Evaluator
* `src/finqa/dsl.py`: regex-parsed, guarded interpreter (no `eval`), mirrors official `eval_program` semantics;
  `tests/test_dsl.py::test_agrees_with_official_evaluator_on_all_gold_programs` passes on all three splits.
* `src/finqa/answers.py`: primary match = 1% relative tolerance, percent-aware (a `%` answer may match either
  gold convention), sign-sensitive; lenient adds ×100/÷100 confusions; exact5 = official execution-accuracy style.
  Missing/malformed answers are failures; coverage reported as `parsed`.
* Scores reported separately: **raw** (model's own stated answer — no tools), **prog** (model's program executed
  by the interpreter — a tool-assisted harness score), **assisted** (prog if the program executes, else raw).

## 3. Baselines (FinQA dev, identical prompt, greedy, non-thinking, batch 1, CPU)
<!-- TABLE:BASELINES -->
| model | n | raw | prog | parsed | valid program | latency s/ex | out tok/ex |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B base (smoke20) | 20 | 10.0 [0.0, 25.0] | 5.0 [0.0, 15.0] | 95.0 | 25.0 | 10.9 | 48.8 |
| Qwen3.5-0.8B base (val100) | 100 | 6.0 [2.0, 11.0] | 1.0 [0.0, 3.0] | 88.0 | 30.0 | 10.3 | 53.6 |

Error profile of the base 0.8B model (val100): 12% no parseable answer (repetition loops that hit the 128-token cap:
13 truncations), 70% of programs invalid (algebraic notation instead of the DSL, placeholder copying such as
`table_sum(row, none)`), and arithmetic/selection errors in the rest. Frontier baselines (GPT / Claude / Gemini):
**pending** (no keys / budget).

## 4. SFT (CPU proxy)
<!-- FILLED AFTER QUEUE -->

## 5. Synthetic data
<!-- FILLED AFTER QUEUE -->

## 6. RL / harness
<!-- FILLED AFTER QUEUE -->

## 7. Cost model
<!-- FILLED -->

## 8. Limitations
<!-- FILLED -->
