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

## 5. Synthetic data (Phase 3)
* Generator: `src/finqa/synth/generator.py` builds fictional multi-year statements with accounting identities
  (gross profit = revenue − COGS, operating income, interest, tax, net income incl. losses; capex as negative
  outflows; segments; share counts), renders them FinQA-style (text + table, FinQA cell conventions), samples a
  **program first** from 12 templates (growth, margin, change in margin, average, table aggregates, ratio, share of
  total, revenue-weighted average, CAGR, per-share with thousands→millions unit conversion, multi-step growth
  comparison → yes/no, free-cash-flow share with negative values), executes it with the restricted interpreter,
  and only then realises a question from several phrasings.
* Automatic checks: program executes; every literal appears in the document; unique row names; no near-zero
  denominators; multi-step answers must not equal a table cell; bounded magnitudes; no duplicate questions per
  document. Document-level split (train / synth_dev by hash).
* Pilot: **1,600 items / 400 documents** (1,412 train, 188 synth-dev); difficulty easy 629 / medium 708 /
  hard 263; 86 candidates rejected (80 template-not-applicable, 2 implausible growth, 2 zero division,
  2 trivial). Audit sheet with 25 samples for **human review**: `docs/SYNTH_AUDIT.md`. Known weaknesses visible in
  the audit: 2-year `table_max/min` questions are near-trivial; growth of negative outflows (capex) is
  arithmetically right but semantically ambiguous. Automated checks do not guarantee semantic correctness.
* Ablation (CPU, 0.8B, matched budget of 640 examples: 480 FinQA + 160 synthetic vs 640 FinQA): see §4 when
  the CPU queue completes.

### Fresh SEC-XBRL evaluation set
`src/finqa/sec/xbrl_eval.py` pulls the latest two 10-K filings per company from `data.sec.gov` company facts
(40 large US companies; latest filings dated 2026-01 to 2026-11), keeps a value only if the prior-year comparative
in the latest 10-K equals the current-year value in the previous 10-K (restated values dropped), and computes
each templated answer twice (DSL on the rendered table vs float arithmetic on raw XBRL values); disagreeing items
are dropped. **303 items** (growth, margins, change in margin, R&D intensity, FCF, cash conversion, leverage,
yes/no growth comparison). It is *templated* — it tests arithmetic grounding on fresh numbers, not open-ended
document understanding — and is grouped by company for analysis.

## 6. RL / harness (Phase 4, stretch)
* Verifiable reward (`src/finqa/train/rewards.py`): correctness 1.0 when both the executed program and the
  stated answer match gold, 0.5 when exactly one does, else 0; +0.2 × grounding (share of program literals found
  in the document) **only when correctness > 0**, so formatting or copying alone never pays. Hedged answers
  (several numbers on the Answer line), outputs > 4,000 chars, sign and scale errors, and injected instructions
  earn 0 — 14 adversarial tests in `tests/test_reward.py`.
* GRPO script (`src/finqa/train/grpo.py`, TRL `GRPOTrainer`, DAPO loss, no KL, group-scaled rewards, masked
  truncations) with a CPU smoke config and a prepared 9B GPU config (vLLM colocate rollouts). CPU smoke result:
  see below when it completes.
* Harness vs weights: the `prog` score is the tool-assisted (interpreter) score and is reported separately from
  `raw`; any frontier comparison must give frontier models the same interpreter. Harness gains are not claimed
  as weight-training gains.

## 7. Cost model (inference) — API list price vs self-hosted
Workload per FinQA question (measured with the Qwen3.5 tokenizer on dev): **~1,400 prompt tokens** (system +
document + DSL spec), **~30–55 output tokens** in direct mode; reasoning/CoT modes typically add 300–2,000
output tokens (not measured here). Other tokenizers may count ~1.0–1.35x more prompt tokens.

**API (list prices; Claude from Anthropic's model table cached 2026-11-03; OpenAI/Gemini from third-party
snapshots of their pricing pages, unverified — confirm before use).** USD per 1,000 questions:

| model (exact ID to confirm on the day) | $/M in | $/M out | direct (1.4k in + 50 out) | reasoning (1.4k in + 1.5k out) |
|---|---|---|---|---|
| Claude Opus 5.5 (`claude-opus-5-5`) | 4.00 | 20.00 | 6.6 | 35.6 |
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | 2.00 | 10.00 | 3.3 | 17.8 |
| Claude Haiku 5.5 (`claude-haiku-5-5`) | 0.10 | 0.50 | 0.17 | 0.89 |
| GPT-5.5 (snapshot, unverified) | 5.00 | 30.00 | 8.5 | 52.0 |
| Gemini 3.x Flash (intro price to 2026-12-31, unverified) | 0.75 | 3.75 | 1.2 | 6.7 |

**Self-hosted (estimate = GPU $/h ÷ measured throughput at stated utilization).**
* Measured here (CPU, single stream, batch 1): Qwen3.5-0.8B = 10.3 s/question on 4 vCPU → at an assumed
  $0.20/h for a 4-vCPU instance, **$0.57 per 1,000 questions** — but at 6% accuracy this is not a usable system.
* 9B on 1x H100 with vLLM (*not measured; assumption*): 5–20 questions/s aggregate at 50–80% utilization
  (prefill-bound: ~1.4k tokens/question) and $2–4/h → **$0.03–$0.22 per 1,000 questions**. Must be measured on
  the first GPU run (`scripts/eval_gpu_matrix.sh` records per-example latency; use `--hourly-usd`).

**Implication:** a fine-tuned 9B model's cost advantage is large against mid/top frontier tiers (~15x for
Sonnet-class direct answers up to >1,000x for Opus/GPT-class reasoning, under the self-hosting assumption above) but roughly **nil against the cheapest frontier tier**
(Claude Haiku 5.5 ≈ $0.17–0.9 per 1k questions). The commercial case therefore requires the specialist to reach
the accuracy of the expensive tiers, or to win on data control/latency — not merely to beat Haiku-tier price.

**Training cost (estimate):** 9B QLoRA SFT, ~18M training tokens (6.2k examples × 1.45k tokens × 2 epochs) on 1x
H100 at an assumed 2–4k tokens/s → 1.3–2.5 h + ~0.5 h eval → **~$4–$13** at $1.9–$4.3/h (RunPod/Vast/Lambda
ranges in `docs/DECISIONS.md`). Actual spend in this session: **$0**.

## 8. Limitations
* **No frontier baselines** (no API keys/budget): every frontier statement is a cost estimate, not an accuracy
  comparison. Published leaderboard numbers are not comparable (different context/retrieval/metrics).
* CPU results use a 0.8B proxy and short budgets; they validate the pipeline and give a small-model data point,
  not evidence about 9B or frontier performance.
* Single seed, single run per configuration; 100-example subsets have ±~9-point CIs.
* FinQA has ~7.7% material label noise in dev (gold `exe_ans` used as official); this caps attainable accuracy.
* Contamination: FinQA is public since 2021; frontier and open models may have seen it. The SEC set (2026 filings)
  is the freshest check but is templated.
* GPU batched decoding (batch 8) can differ from batch-1 decoding on rare near-tie tokens (observed on CPU);
  applied identically to base and SFT.
* Synthetic data quality is only automatically checked; the human audit sheet is unreviewed.
