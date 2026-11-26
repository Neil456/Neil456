# finqa-specialist

Reproducible post-training research pipeline testing whether an open-weight model can match or beat
proprietary frontier models on narrow financial-document numerical reasoning (FinQA-style) at lower
inference cost. **Research prototype, not financial advice or a production service.**

* Results and limitations: [`REPORT.md`](REPORT.md) · exact next commands and blockers: [`NEXT_STEPS.md`](NEXT_STEPS.md)
* Phase-0 decisions, model revisions, licenses, price assumptions: [`docs/DECISIONS.md`](docs/DECISIONS.md)
* Synthetic data human-review sheet: [`docs/SYNTH_AUDIT.md`](docs/SYNTH_AUDIT.md)

## Layout
```
configs/      SFT / GRPO configs (CPU proxy runs + prepared GPU runs)
src/finqa/
  dsl.py          restricted FinQA program interpreter (no eval/exec), verified vs the official evaluator
  answers.py      answer extraction + numeric matching (primary / lenient / exact5)
  prompts.py      single prompt shared by every model; SFT target format
  data/build.py   canonical schema, ingestion (FinQA, TAT-QA, ConvFinQA, FinanceBench), dedup, leakage flags
  eval/           runner CLI (HF local, OpenAI-compatible, Anthropic, Gemini), scoring, metrics, paired bootstrap
  train/          sft.py (TRL SFTTrainer, prompt-masked), grpo.py (TRL GRPOTrainer), rewards.py (verifiable reward)
  synth/          programmatic verified synthetic data generator + audit sheet
  sec/            fresh evaluation set from recent 10-K XBRL facts (data.sec.gov)
scripts/      eval_local.sh, cpu_queue.sh (resumable experiment queue), make_tables.py, cloud/ launch kits
tests/        parser, DSL vs official evaluator, split leakage, runner/adapters, reward adversarial tests
results/      predictions JSONL, summaries, manifests, subsets (no secrets, no raw data)
```

## Quick start (CPU)
```bash
make setup download data       # venv, raw datasets at pinned commits, canonical JSONL + manifest
make test                      # 77 tests incl. agreement with the official FinQA evaluator on all gold programs
make smoke                     # 20-example eval of Qwen3.5-0.8B on FinQA dev
bash scripts/cpu_queue.sh      # overfit smoke -> CPU SFT -> checkpoint selection -> paired evals -> ablation
make report-tables             # results/summary.md with bootstrap CIs and paired comparisons
make cloud-dryrun              # GPU launch command + worst-case cost (never launches without CONFIRM=1)
```

## Evaluation protocol (short)
Identical prompt and supplied document for every model; greedy decoding; outputs end with
`Program: <FinQA DSL>` and `Answer: <number>`. Scores: **raw** (model's stated answer, 1% relative
tolerance, percent-aware, sign-sensitive), **prog** (model's program executed by the restricted interpreter —
a tool-assisted harness score, reported separately), **exact5** (official FinQA style). Missing or
malformed answers count as wrong. The FinQA test split is refused unless `--final-test` is passed.
