# NEXT_STEPS — exact commands, blockers, and the next highest-value experiment

## Blockers (need a human decision)
1. **No GPU and no authorized budget.** All 9B work is prepared but not run. To proceed, provide a budget and
   a provider key (RunPod / Vast / Lambda), then use the dry-run launchers (they refuse to exceed `MAX_USD`).
2. **No frontier API keys / budget.** GPT, Claude and Gemini baselines are *pending* (no numbers fabricated).
   `api.openai.com` is also blocked by this environment's network policy (Anthropic and Gemini endpoints
   are reachable but unauthenticated).
3. **Pushing to GitHub not approved.** `Neil456/Neil456` is public; the brief forbids pushing without approval.
   Work is committed locally on `claude/great-ramanujan-vbwwgm` only. To publish: `git push -u origin claude/great-ramanujan-vbwwgm`.
4. **Human review of synthetic data** (`docs/SYNTH_AUDIT.md`, 25 samples) before scaling synthetic data.

## Reproduce everything done here (CPU)
```bash
cd finqa-specialist
make setup download data synth sec   # environment, raw data at pinned commits, canonical data, synthetic pilot, SEC fresh set
make test                            # full test suite
bash scripts/cpu_queue.sh            # resumable: overfit smoke, CPU SFT, selection, paired evals, synth ablation, 2B baseline
make report-tables                   # results/summary.md
```

## GPU: 9B QLoRA SFT (prepared)
Estimated cost (see REPORT.md §Cost): Qwen3.5-9B QLoRA, 6.2k examples x 2 epochs, ~1.45k tokens/example
(~18M training tokens). On 1x H100 80GB expect roughly 1.5–3 h including eval (assumption: 2–4k training
tokens/s for 9B QLoRA with gradient checkpointing; Gated-DeltaNet kernels required) -> **~$4–$13** at
$1.9–$4.3/h. On 1x A100 80GB roughly 2x the time at about half the hourly price.
```bash
# 1) dry run (prints the exact command and the worst-case cost; refuses if > MAX_USD)
MAX_USD=25 MAX_HOURS=6 PRICE_PER_HR=2.99 bash scripts/cloud/runpod_launch.sh
# 2) launch (only after approval)
CONFIRM=1 MAX_USD=25 MAX_HOURS=6 bash scripts/cloud/runpod_launch.sh
# on the pod (resumable; watchdog stops the pod at MAX_HOURS):
STAGE=all MAX_HOURS=6 bash /workspace/finqa-specialist/scripts/cloud/remote_job.sh
# select checkpoint on the select60 dev subset, then evaluate the matrix (FinQA dev full, TAT-QA dev, FinanceBench numeric, SEC fresh)
python scripts/select_checkpoint.py /workspace/checkpoints/sft_gpu_q35_9b
ADAPTER=/workspace/checkpoints/sft_gpu_q35_9b/best bash scripts/eval_gpu_matrix.sh
make report-tables
```
First-boot checks on the GPU: `python -c "import fla, causal_conv1d"` (fast kernels), run
`make sft-smoke` with `configs/sft_overfit_smoke.yaml` (model -> 9B, load_in_4bit: true) to measure VRAM/throughput before the full run.

## Frontier baselines (prepared; need keys + budget)
Use exact current model IDs (verify on provider docs on the day) and list prices; the runner refuses paid
runs without a hard cap:
```bash
export FINQA_ANTHROPIC_API_KEY=...   # never commit
python -m finqa.eval.runner --adapter anthropic --model <exact-claude-model-id> --price-in <usd/Mtok> --price-out <usd/Mtok> \
  --data data/processed/finqa_dev.jsonl --ids-file results/subsets/finqa_dev_val100.txt --variant cot --max-new-tokens 2048 \
  --allow-paid --max-usd 5 --out results/preds/<model>_cot_val100.jsonl
# same for --adapter openai_compat --base-url https://api.openai.com/v1 (needs api.openai.com allowed) and --adapter gemini
```
Estimated API cost for FinQA dev val100 with CoT: ~100 x (1.5k in + 0.5–2k out) tokens -> roughly $0.5–$5 per model
depending on tier/reasoning effort.

## Next highest-value experiment
Run the 9B QLoRA SFT on 1x H100 (≈$5–15), evaluate base vs SFT on full FinQA dev (883) + SEC fresh (303) +
FinanceBench numeric (52), and run 2–3 frontier models on the same FinQA dev val100/val300 ids with matched
prompts (raw answers) — that single comparison answers the commercial question (accuracy per dollar).
Only then freeze the configuration and run the FinQA test split once with `--final-test`.

## Then
* GRPO from the SFT checkpoint with the verifiable reward (`configs/grpo_gpu_q35_9b.yaml`, vLLM colocate);
  measure rollout VRAM/throughput on 20 steps before scaling.
* Calculator-tool harness comparison: base vs trained with identical tool access (the `prog` score already
  is the "program executed by interpreter" harness; report raw and assisted separately).
* Scale synthetic data only if the CPU ablation (or a GPU rerun) shows benefit at matched budget.
