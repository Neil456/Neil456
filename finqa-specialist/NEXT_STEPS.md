# NEXT_STEPS — exact commands, blockers, and the next highest-value experiment

## Status (2026-11-26)
* Done: data pipeline + leakage checks, verified evaluator, runner, tests; CPU 0.8B baseline; **9B QLoRA SFT on a
  rented A100** with full FinQA-dev / SEC-fresh / FinanceBench evaluation (REPORT §4.1). GPU spend so far $2.57.
* Running at time of writing: phase-B GPU job (synthetic-data ablation at 9B, Qwen3.8-27B 4-bit baseline) and the
  CPU 0.8B SFT queue (+ synthetic ablation, 2B baseline, GRPO smoke). REPORT is updated as these land.

## Blockers (need a human decision)
1. **Frontier baselines (GPT / Claude / Gemini) — pending.** Need API keys + a small budget (≈$1–5 per model for
   val100). The runner refuses paid runs without `--allow-paid --max-usd`. This is the comparison that answers the
   commercial question; nothing in REPORT claims frontier parity.
2. **Vast.ai key rotation.** The key used tonight was pasted in chat; rotate it (the user planned to). It is stored only
   in `~/.config/vastai/vast_api_key` inside this ephemeral container and is not in the repository.
3. **Human review of synthetic data** (`docs/SYNTH_AUDIT.md`, 25 samples) before scaling synthetic data.
4. The fine-tuned 9B adapter was not retrieved (instance destroyed after evaluation). Re-create with the command below
   (~$1) on a host where you can keep the files (e.g. a persistent volume or push to a private HF repo).

## Reproduce
```bash
cd finqa-specialist
make setup download data synth sec      # environment, raw data at pinned commits, canonical data, synthetic pilot, SEC set
make test                               # 83 tests
bash scripts/cpu_queue.sh               # CPU: overfit smoke, 0.8B SFT, selection, paired evals, synth ablation, 2B baseline
# GPU (Vast.ai; needs ~/.config/vastai/vast_api_key; key is sent as a query parameter because the agent proxy strips
# Authorization headers). plan = no spend; launch refuses if worst case > --max-usd; watch fetches results + destroys.
python scripts/cloud/vast_run.py plan --gpus "A100 SXM4" --max-usd 8
python scripts/cloud/vast_run.py launch --confirm --gpus "A100 SXM4" --max-usd 8            # 9B SFT + evals (bundle mode)
python scripts/cloud/vast_run.py watch
python scripts/cloud/vast_run.py launch --confirm --clone --job scripts/cloud/vast_job_b.sh \
       --payload-dir results/cloud/payload_b --gpus "A100 SXM4" --max-usd 5.4 --max-hours 4   # ablation + 27B baseline
python scripts/cloud/vast_run.py watch --payload-dir results/cloud/payload_b
python scripts/rescore_payload.py results/cloud/payload      # rescore with the local evaluator
python scripts/make_tables.py                                # results/summary.md + paired comparisons
```
Lessons encoded in the scripts: exclude hosts where HF/GitHub are blocked; network preflight + timeouts on every
download; sparse FinanceBench checkout (its repo holds ~360 PDFs); Vast env ≤32KB and onstart ≤16KB (bundle split, or
clone the pushed branch); the logs API returns `temp_download_url` and reuses one S3 key per instance (wait before
reading); never place API keys on the rented host.

## Frontier baselines (prepared)
```bash
export FINQA_ANTHROPIC_API_KEY=...   # never commit
python -m finqa.eval.runner --adapter anthropic --model claude-sonnet-5-5 --price-in 2 --price-out 10 \
  --data data/processed/finqa_dev.jsonl --ids-file results/subsets/finqa_dev_val300.txt --variant direct --max-new-tokens 512 \
  --allow-paid --max-usd 5 --out results/preds/claude-sonnet-5-5_direct_val300.jsonl
# also: --adapter openai_compat --base-url https://api.openai.com/v1 (host must be allowed) and --adapter gemini
```
Give frontier models the same interpreter (score `prog`) for a matched-tool comparison; report raw and assisted
separately. Verify exact model IDs and prices on the providers' pages on the day.

## Next highest-value experiments (in order)
1. **Frontier comparison** on FinQA dev val300 + SEC fresh with matched prompt/tools (≈$5–15 total). Decides whether
   "comparable accuracy at ≥30x lower cost" holds.
2. **Program-execution deployment**: report/serve the SFT model's executed-program answer (FinQA dev 77.7%, SEC 97.0%);
   add a fallback to the stated answer when the program fails.
3. **Second epoch / LR sweep (1e-4 vs 2e-4) / rank 16 vs 64** on 9B (~$1 each) using `select60` for selection.
4. **GRPO from the SFT checkpoint** with the verifiable reward (`configs/grpo_gpu_q35_9b.yaml`; vLLM colocate);
   measure rollout throughput on 20 steps first. Target: the stated-answer arithmetic gap (raw 76.0 vs prog 77.7 on
   FinQA, larger on FinanceBench) and remaining operation-selection errors.
5. Serve with vLLM to measure real throughput and $/1k questions (current $0.10/1k is an HF batch-8 upper bound).
6. Only after the configuration is frozen: run the FinQA **test** split once (`--final-test`).
