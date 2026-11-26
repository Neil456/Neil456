# NEXT_STEPS — exact commands, blockers, and the next highest-value experiment

## Status (2026-11-26, end of session)
* Done: data pipeline + leakage checks, verified evaluator, runner, 84 tests; 0.8B CPU baseline + LoRA SFT; **9B QLoRA
  SFT on a rented A100** with full FinQA-dev / SEC-fresh / FinanceBench evaluation; synthetic-data ablation at 9B;
  Qwen3.8-27B 4-bit baseline; GRPO smoke test at 9B (throughput/VRAM/reward measured; OOM limits documented).
* GPU spend **$4.69** of the $10 Vast credit ($5.31 left); all instances destroyed. Results: `REPORT.md`,
  `results/summary.md`, `results/paired_comparisons.json`. Draft PR: Neil456/Neil456#2.

## Blockers (need a human decision)
1. **Frontier baselines (GPT / Claude / Gemini) — pending.** Need API keys + a small budget (≈$1–5 per model for
   val100). The runner refuses paid runs without `--allow-paid --max-usd`. This is the comparison that answers the
   commercial question; nothing in REPORT claims frontier parity.
2. **Vast.ai key rotation.** The key used tonight was pasted in chat; rotate it (the user planned to). It is stored only
   in `~/.config/vastai/vast_api_key` inside this ephemeral container and is not in the repository.
3. **Human review of synthetic data** (`docs/SYNTH_AUDIT.md`, 25 samples) before scaling synthetic data.
4. The fine-tuned 9B adapter was not retrieved (instance destroyed after evaluation; the log channel cannot carry ~200 MB).
   Re-create with the command below (~$1) on a host where you can keep the files (persistent volume or a private HF repo,
   which needs an HF token).

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
4. **GRPO from the SFT checkpoint** with the verifiable reward. The 9B smoke test measured 0.8 completions/s with HF
   generation and OOM'd on a 40 GB A100 at micro-batch 2 on the longest prompts. Next: vLLM rollouts
   (`use_vllm: true`), cap prompts at ~2,048 tokens, filter prompts by SFT pass rate (37% had zero reward variance),
   checkpoint every 50 steps, 80 GB GPU or 4-bit base. Target: the stated-answer arithmetic gap (raw 76.0 vs prog 77.7
   on FinQA; much larger on FinanceBench) and operation-selection errors.
5. Serve with vLLM to measure real throughput and $/1k questions (current $0.10/1k is an HF batch-8 upper bound).
6. Only after the configuration is frozen: run the FinQA **test** split once (`--final-test`).
