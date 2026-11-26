#!/usr/bin/env bash
# Runs after scripts/cpu_queue.sh completes: GRPO CPU smoke test (stretch goal), then final tables.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/home/user/hf_cache} TOKENIZERS_PARALLELISM=false
until grep -q "all done" logs/cpu_queue.log 2>/dev/null; do
  grep -q "FAILED" logs/cpu_queue.log 2>/dev/null && pgrep -f cpu_queue.sh >/dev/null || true
  sleep 60
done
if [ ! -f logs/done.grpo_smoke ]; then
  echo "[tail] $(date -u +%H:%M:%S) start grpo_smoke"
  if python -m finqa.train.grpo --config configs/grpo_cpu_smoke.yaml --override init_adapter=checkpoints/sft_cpu_q35_0.8b/best >> logs/grpo_smoke.log 2>&1; then
    touch logs/done.grpo_smoke; echo "[tail] $(date -u +%H:%M:%S) done grpo_smoke"
  else echo "[tail] $(date -u +%H:%M:%S) FAILED grpo_smoke"; fi
fi
python scripts/make_tables.py > /dev/null 2>&1
echo "[tail] all done $(date -u)"
