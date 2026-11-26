#!/usr/bin/env bash
# Runs ON the rented GPU box. Idempotent + resumable: re-running continues from the last checkpoint.
# Required env: STAGE (sft|eval|grpo|all), WORKDIR (persistent volume, default /workspace), MAX_HOURS (hard cap)
set -euo pipefail
WORKDIR=${WORKDIR:-/workspace}; MAX_HOURS=${MAX_HOURS:-6}; STAGE=${STAGE:-all}
cd "$WORKDIR"
# Watchdog: hard wall-clock cap -> stop the pod even if the job hangs (prevents idle billing).
( sleep $(( MAX_HOURS * 3600 )); echo "[watchdog] MAX_HOURS reached, stopping"; \
  (command -v runpodctl >/dev/null && runpodctl stop pod "${RUNPOD_POD_ID:-}") || sudo shutdown -h now ) &
[ -d finqa-specialist ] || { echo "copy or clone the project into $WORKDIR/finqa-specialist first"; exit 1; }
cd finqa-specialist
python -m venv .venv-gpu 2>/dev/null || true; source .venv-gpu/bin/activate
pip install -q -r requirements-gpu.txt && pip install -q -e .
export HF_HOME=$WORKDIR/hf_cache
[ -f data/processed/finqa_train.jsonl ] || make data
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a results/gpu_info.txt
if [[ $STAGE == sft || $STAGE == all ]]; then
  python -m finqa.train.sft --config configs/sft_gpu_q35_9b.yaml 2>&1 | tee -a $WORKDIR/sft.log
fi
if [[ $STAGE == eval || $STAGE == all ]]; then
  bash scripts/eval_gpu_matrix.sh 2>&1 | tee -a $WORKDIR/eval.log
fi
if [[ $STAGE == grpo ]]; then
  python -m finqa.train.grpo --config configs/grpo_gpu_q35_9b.yaml 2>&1 | tee -a $WORKDIR/grpo.log
fi
echo "[remote_job] done; stopping pod"
(command -v runpodctl >/dev/null && runpodctl stop pod "${RUNPOD_POD_ID:-}") || true
