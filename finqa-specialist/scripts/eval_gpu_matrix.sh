#!/usr/bin/env bash
# GPU evaluation matrix on FinQA dev (full) + TAT-QA dev + FinanceBench numeric + SEC fresh. Never touches test.
set -euo pipefail
cd "$(dirname "$0")/.."
M9=Qwen/Qwen3.5-9B; R9=c202236235762e1c871ad0ccb60c8ee5ba337b9a
ADP=${ADAPTER:-/workspace/checkpoints/sft_gpu_q35_9b/best}
for data in finqa_dev tatqa_dev financebench_eval sec_fresh_eval; do
  for v in direct cot; do
    mx=$([[ $v == cot ]] && echo 1024 || echo 160)
    python -m finqa.eval.runner --adapter hf --model $M9 --revision $R9 --data data/processed/$data.jsonl \
      --variant $v --max-new-tokens $mx --batch-size 8 --out results/preds/q35-9b_base_${v}_${data}.jsonl
    [ -d "$ADP" ] && python -m finqa.eval.runner --adapter hf --model $M9 --revision $R9 --lora "$ADP" \
      --data data/processed/$data.jsonl --variant direct --max-new-tokens 160 --batch-size 8 \
      --out results/preds/q35-9b_sft_direct_${data}.jsonl || true
  done
done
