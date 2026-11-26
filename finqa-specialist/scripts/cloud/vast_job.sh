#!/usr/bin/env bash
# Runs ON a rented Vast.ai GPU instance (started by scripts/cloud/vast_run.py via --onstart-cmd).
# Self-contained: clones the project branch, installs pinned deps, builds data, runs 9B QLoRA SFT,
# selects the checkpoint on the select60 dev subset, evaluates base vs SFT, then prints a compressed
# results payload to the container log (retrieved by vast_run.py through the Vast logs API) and stops
# the container. No API keys are ever placed on the instance.
# Env: REPO_URL, BRANCH, MAX_HOURS (hard wall-clock cap), EPOCHS (default 2)
set -uo pipefail
MAX_HOURS=${MAX_HOURS:-5}; EPOCHS=${EPOCHS:-2}
REPO_URL=${REPO_URL:-https://github.com/Neil456/Neil456}; BRANCH=${BRANCH:-claude/great-ramanujan-vbwwgm}
log() { echo "[finqa-job $(date -u +%H:%M:%S)] $*"; }
# Watchdog: stop the container (PID 1) at the cap even if the job hangs -> GPU billing stops.
( sleep $(( MAX_HOURS * 3600 )); log "WATCHDOG: MAX_HOURS reached"; kill -TERM 1 ) &
mkdir -p /workspace && cd /workspace
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git unzip >/dev/null)
command -v unzip >/dev/null || (apt-get update -qq && apt-get install -y -qq unzip >/dev/null)
# Code arrives as a bundle extracted by the onstart command into /workspace/repo/finqa-specialist;
# fall back to cloning the branch if it is not there.
[ -f repo/finqa-specialist/pyproject.toml ] || git clone -q --depth 1 -b "$BRANCH" "$REPO_URL" repo || { log "FATAL no code"; kill -TERM 1; }
cd repo/finqa-specialist
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "pip install"
pip install -q "transformers==5.24.1" "trl==1.19.0" "peft==0.22.0" "accelerate>=1.12" "datasets>=4.7" \
  bitsandbytes pyyaml httpx scipy flash-linear-attention 2>&1 | tail -2
pip install -q -e . 2>&1 | tail -1
python -c "import torch,transformers,trl,peft;print('versions',torch.__version__,transformers.__version__,trl.__version__,peft.__version__)"
python -c "import fla; print('fla ok')" 2>/dev/null || log "flash-linear-attention unavailable (torch fallback)"
export HF_HOME=/workspace/hf_cache TOKENIZERS_PARALLELISM=false
log "downloading datasets"
bash scripts/download_data.sh 2>&1 | tail -8
python -m finqa.data.build --raw data/raw --out data/processed 2>&1 | tail -2 && log "data built" || { log "FATAL data build"; kill -TERM 1; }
timeout 900 python -m finqa.sec.xbrl_eval --out data/processed/sec_fresh_eval.jsonl 2>&1 | tail -1 && log "sec built" || log "sec build failed (skipping)"
log "downloading Qwen3.5-9B"
timeout 2400 python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))" 2>&1 | tail -1 && log "model downloaded" || { log "FATAL model download"; kill -TERM 1; }
M=Qwen/Qwen3.5-9B; R=c202236235762e1c871ad0ccb60c8ee5ba337b9a
CK=/workspace/checkpoints/sft_gpu_q35_9b
log "SFT start (epochs=$EPOCHS)"
python -m finqa.train.sft --config configs/sft_gpu_q35_9b.yaml --override output_dir=$CK epochs=$EPOCHS per_device_batch_size=1 grad_accum=16 save_steps=100 eval_steps=100 logging_steps=5 max_train_hours=$(( MAX_HOURS - 2 )) 2>&1 \
  | grep --line-buffered -E "\[sft\]|'loss'|eval_loss|Error|Traceback|out of memory"
python scripts/select_checkpoint.py $CK && log "selected $(readlink $CK/best)"
EV="python -m finqa.eval.runner --adapter hf --model $M --revision $R --batch-size 8 --max-new-tokens 160 --hourly-usd ${HOURLY_USD:-0}"
$EV --data data/processed/finqa_dev.jsonl --out results/preds/q35-9b_base_direct_finqa_dev.jsonl 2>&1 | tail -3
$EV --data data/processed/finqa_dev.jsonl --lora $CK/best --out results/preds/q35-9b_sft_direct_finqa_dev.jsonl 2>&1 | tail -3
if [ -f data/processed/sec_fresh_eval.jsonl ]; then
  $EV --data data/processed/sec_fresh_eval.jsonl --out results/preds/q35-9b_base_direct_sec_fresh.jsonl 2>&1 | tail -1
  $EV --data data/processed/sec_fresh_eval.jsonl --lora $CK/best --out results/preds/q35-9b_sft_direct_sec_fresh.jsonl 2>&1 | tail -1
fi
$EV --data data/processed/financebench_eval.jsonl --ids-file results/subsets/financebench_numeric.txt --out results/preds/q35-9b_base_direct_financebench_numeric.jsonl 2>&1 | tail -1
$EV --data data/processed/financebench_eval.jsonl --ids-file results/subsets/financebench_numeric.txt --lora $CK/best --out results/preds/q35-9b_sft_direct_financebench_numeric.jsonl 2>&1 | tail -1
# stronger base baseline: thinking-mode-off CoT on val100 (1024 new tokens)
python -m finqa.eval.runner --adapter hf --model $M --revision $R --batch-size 8 --variant cot --max-new-tokens 1024 \
  --data data/processed/finqa_dev.jsonl --ids-file results/subsets/finqa_dev_val100.txt --out results/preds/q35-9b_base_cot_val100.jsonl 2>&1 | tail -1
python scripts/make_tables.py > /dev/null 2>&1
cp $CK/selection.json $CK/run_manifest.json $CK/train_log.jsonl results/ 2>/dev/null
mkdir -p /workspace/payload
# compact predictions (raw outputs + timing/tokens); scoring is redone locally with the same evaluator
python - <<'PY'
import json, glob, os
for f in glob.glob("results/preds/q35-9b_*.jsonl"):
    keep = ("id", "config_hash", "model", "revision", "lora", "variant", "output", "input_tokens", "output_tokens",
            "latency_s", "finish_reason", "est_cost_usd", "correct", "prog_correct")
    with open(f) as fi, open("/workspace/payload/" + os.path.basename(f), "w") as fo:
        for line in fi:
            r = json.loads(line); fo.write(json.dumps({k: r.get(k) for k in keep}) + "\n")
PY
cp results/preds/q35-9b_*.summary.json results/preds/q35-9b_*.manifest.json results/selection.json results/run_manifest.json results/train_log.jsonl results/summary.md /workspace/payload/ 2>/dev/null
tar czf /workspace/payload.tgz -C /workspace payload
log "PAYLOAD_BEGIN $(stat -c %s /workspace/payload.tgz) bytes"
base64 -w 76 /workspace/payload.tgz | nl -ba -nrz -w6 -s ' ' | sed 's/^/@P@ /'
log "PAYLOAD_END"
log "FINQA_JOB_DONE"
sleep 1800   # give the launcher time to fetch logs, then stop
kill -TERM 1
