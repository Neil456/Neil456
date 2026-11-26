#!/usr/bin/env bash
# Phase-C GPU job: GRPO smoke test at 9B (Phase 4 stretch).
#  1. SFT init: Qwen3.5-9B QLoRA on 1,600 FinQA train examples (same config as the phase-B ablation arm A).
#  2. Eval SFT on FinQA dev val300.
#  3. GRPO (TRL, verifiable reward, DAPO loss, no KL) for 30 steps x 8 prompts x 8 generations on FinQA train prompts
#     disjoint from the SFT init data; records rollout VRAM, throughput, reward curve.
#  4. Eval SFT+GRPO (adapters merged in order) on the same val300 ids.
set -uo pipefail
MAX_HOURS=${MAX_HOURS:-2}
log() { echo "[finqa-job $(date -u +%H:%M:%S)] $*"; }
( sleep $(( MAX_HOURS * 3600 )); log "WATCHDOG: MAX_HOURS reached"; kill -TERM 1 ) &
cd /workspace/repo/finqa-specialist || { log "FATAL no code"; kill -TERM 1; exit 1; }
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); code $(git rev-parse --short HEAD)"
for u in https://pypi.org/simple/trl/ https://github.com https://huggingface.co/api/models/Qwen/Qwen3.5-9B; do
  c=$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$u"); [ "$c" = "000" ] && { log "FATAL network: $u"; kill -TERM 1; exit 1; }
done
log "pip install"
timeout 1500 pip install --progress-bar off --default-timeout 60 "transformers==5.24.1" "trl==1.19.0" "peft==0.22.0" "accelerate>=1.12" "datasets>=4.7" \
  bitsandbytes pyyaml httpx scipy flash-linear-attention 2>&1 | grep -E "^(Successfully|ERROR)" | cut -c1-300 \
  || { log "FATAL pip install"; kill -TERM 1; exit 1; }
pip install -q -e . 2>&1 | tail -1
export HF_HOME=/workspace/hf_cache TOKENIZERS_PARALLELISM=false
bash scripts/download_data.sh 2>&1 | tail -1
python -m finqa.data.build --raw data/raw --out data/processed 2>&1 | tail -1 && log "data built" || { log "FATAL data build"; kill -TERM 1; exit 1; }
M=Qwen/Qwen3.5-9B; R=c202236235762e1c871ad0ccb60c8ee5ba337b9a
SFT=/workspace/checkpoints/abl_finqa1600; GR=/workspace/checkpoints/grpo_smoke_9b
EVAL="python -m finqa.eval.runner --adapter hf --batch-size 8 --max-new-tokens 160 --model $M --revision $R --data data/processed/finqa_dev.jsonl --ids-file results/subsets/finqa_dev_val300.txt"
log "SFT init start"
python -m finqa.train.sft --config configs/sft_gpu_q35_9b_abl_finqa1600.yaml --override max_train_hours=1 2>&1 | grep --line-buffered -E "\[sft\]|Error|Traceback|out of memory"
[ -d $SFT/final ] || { log "FATAL SFT init"; kill -TERM 1; exit 1; }
log "eval SFT init (val300)"
$EVAL --lora $SFT/final --out results/preds/q35-9b_grpoinit_sft1600_direct_val300.jsonl 2>&1 | tail -1
log "GRPO start"
python -m finqa.train.grpo --config configs/grpo_gpu_smoke_9b.yaml 2>&1 | grep --line-buffered -E "\[grpo\]|'reward'|Error|Traceback|out of memory" | cut -c1-400
cp $GR/run_manifest.json results/run_manifest_grpo_smoke.json 2>/dev/null; cp $GR/train_log.jsonl results/train_log_grpo_smoke.jsonl 2>/dev/null
if [ -d $GR/final ]; then
  log "eval SFT+GRPO (val300)"
  $EVAL --lora $SFT/final,$GR/final --out results/preds/q35-9b_grpo_smoke_direct_val300.jsonl 2>&1 | tail -1
else
  log "GRPO produced no adapter"
fi
mkdir -p /workspace/payload
python - <<'PY'
import json, glob, os
keep = ("id", "config_hash", "model", "revision", "lora", "variant", "output", "input_tokens", "output_tokens",
        "latency_s", "finish_reason", "est_cost_usd", "correct", "prog_correct")
for f in glob.glob("results/preds/q35-9b_grpo*.jsonl"):
    with open(f) as fi, open("/workspace/payload/" + os.path.basename(f), "w") as fo:
        for line in fi:
            r = json.loads(line); fo.write(json.dumps({k: r.get(k) for k in keep}) + "\n")
PY
cp results/preds/q35-9b_grpo*.summary.json results/run_manifest_grpo_smoke.json results/train_log_grpo_smoke.jsonl /workspace/payload/ 2>/dev/null
tar czf /workspace/payload.tgz -C /workspace payload
log "PAYLOAD_BEGIN $(stat -c %s /workspace/payload.tgz) bytes"
base64 -w 76 /workspace/payload.tgz | nl -ba -nrz -w6 -s ' ' | sed 's/^/@P@ /'
log "PAYLOAD_END"
log "FINQA_JOB_DONE"
sleep 2400
kill -TERM 1
