#!/usr/bin/env bash
# Phase-B GPU job (Vast.ai, started by vast_run.py --job scripts/cloud/vast_job_b.sh --clone):
#  1. Synthetic-data ablation at matched budget on Qwen3.5-9B QLoRA (1 epoch, final checkpoint):
#       A = 1,600 FinQA train examples; B = the first 1,200 of those + 400 verified synthetic examples.
#     Each evaluated on FinQA dev (883) and SEC fresh (303), identical prompt, greedy, batch 8.
#  2. Strong open baseline: Qwen/Qwen3.8-27B (4-bit NF4 on a 40GB GPU), direct mode, same eval sets.
# Results are printed as a numbered base64 payload in the container log, then the container stops.
set -uo pipefail
MAX_HOURS=${MAX_HOURS:-4}
log() { echo "[finqa-job $(date -u +%H:%M:%S)] $*"; }
( sleep $(( MAX_HOURS * 3600 )); log "WATCHDOG: MAX_HOURS reached"; kill -TERM 1 ) &
cd /workspace/repo/finqa-specialist || { log "FATAL no code"; kill -TERM 1; exit 1; }
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); code $(git rev-parse --short HEAD)"
log "network preflight"
for u in https://pypi.org/simple/trl/ https://github.com https://huggingface.co/api/models/Qwen/Qwen3.8-27B; do
  c=$(curl -s -o /dev/null -m 20 -w '%{http_code}' "$u"); log "  $u -> $c"
  [ "$c" = "000" ] && { log "FATAL network: $u unreachable"; kill -TERM 1; exit 1; }
done
log "pip install"
timeout 1500 pip install --progress-bar off --default-timeout 60 "transformers==5.24.1" "trl==1.19.0" "peft==0.22.0" "accelerate>=1.12" "datasets>=4.7" \
  bitsandbytes pyyaml httpx scipy flash-linear-attention 2>&1 | grep -E "^(Successfully|ERROR)" | cut -c1-300 \
  || { log "FATAL pip install"; kill -TERM 1; exit 1; }
pip install -q -e . 2>&1 | tail -1
export HF_HOME=/workspace/hf_cache TOKENIZERS_PARALLELISM=false
log "data"
bash scripts/download_data.sh 2>&1 | tail -2
python -m finqa.data.build --raw data/raw --out data/processed 2>&1 | tail -1 && log "data built" || { log "FATAL data build"; kill -TERM 1; exit 1; }
timeout 900 python -m finqa.sec.xbrl_eval --out data/processed/sec_fresh_eval.jsonl 2>&1 | tail -1 && log "sec built" || log "sec build failed"
python -m finqa.synth.generator --n-docs 400 --out data/synth/pilot.jsonl --audit /tmp/audit.md > /dev/null 2>&1 && log "synth $(sha256sum data/synth/pilot.jsonl | cut -c1-16)" || { log "FATAL synth"; kill -TERM 1; exit 1; }
M=Qwen/Qwen3.5-9B; R=c202236235762e1c871ad0ccb60c8ee5ba337b9a
EVAL="python -m finqa.eval.runner --adapter hf --batch-size 8 --max-new-tokens 160 --hourly-usd ${HOURLY_USD:-0}"
for name in abl_finqa1600 abl_finqa1200_synth400; do
  CK=/workspace/checkpoints/$name
  log "SFT $name start"
  python -m finqa.train.sft --config configs/sft_gpu_q35_9b_$name.yaml --override max_train_hours=1.5 2>&1 \
    | grep --line-buffered -E "\[sft\]|'loss'|Error|Traceback|out of memory"
  [ -d $CK/final ] || { log "SFT $name produced no adapter"; continue; }
  cp $CK/train_log.jsonl results/train_log_$name.jsonl; cp $CK/run_manifest.json results/run_manifest_$name.json
  log "eval $name"
  $EVAL --model $M --revision $R --lora $CK/final --data data/processed/finqa_dev.jsonl --out results/preds/q35-9b_${name}_direct_finqa_dev.jsonl 2>&1 | tail -1
  [ -f data/processed/sec_fresh_eval.jsonl ] && $EVAL --model $M --revision $R --lora $CK/final --data data/processed/sec_fresh_eval.jsonl --out results/preds/q35-9b_${name}_direct_sec_fresh.jsonl 2>&1 | tail -1
  log "done $name"
done
M27=Qwen/Qwen3.8-27B; R27=37561816e66b5b9153b1b64c10968c18c1e6f4a0
log "downloading $M27"
timeout 2400 python -c "from huggingface_hub import snapshot_download; snapshot_download('$M27', revision='$R27')" 2>&1 | tail -1 && log "27B downloaded" || log "27B download failed"
log "eval 27B (4-bit)"
$EVAL --model $M27 --revision $R27 --load-in-4bit --data data/processed/finqa_dev.jsonl --out results/preds/q38-27b-4bit_base_direct_finqa_dev.jsonl 2>&1 | tail -1
[ -f data/processed/sec_fresh_eval.jsonl ] && $EVAL --model $M27 --revision $R27 --load-in-4bit --data data/processed/sec_fresh_eval.jsonl --out results/preds/q38-27b-4bit_base_direct_sec_fresh.jsonl 2>&1 | tail -1
log "evals done"
mkdir -p /workspace/payload
python - <<'PY'
import json, glob, os
keep = ("id", "config_hash", "model", "revision", "lora", "variant", "output", "input_tokens", "output_tokens",
        "latency_s", "finish_reason", "est_cost_usd", "correct", "prog_correct")
for f in glob.glob("results/preds/q35-9b_abl_*.jsonl") + glob.glob("results/preds/q38-27b*.jsonl"):
    with open(f) as fi, open("/workspace/payload/" + os.path.basename(f), "w") as fo:
        for line in fi:
            r = json.loads(line); fo.write(json.dumps({k: r.get(k) for k in keep}) + "\n")
PY
cp results/preds/q35-9b_abl_*.summary.json results/preds/q38-27b*.summary.json results/preds/*abl*.manifest.json results/preds/q38-27b*.manifest.json results/train_log_abl_*.jsonl results/run_manifest_abl_*.json /workspace/payload/ 2>/dev/null
tar czf /workspace/payload.tgz -C /workspace payload
log "PAYLOAD_BEGIN $(stat -c %s /workspace/payload.tgz) bytes"
base64 -w 76 /workspace/payload.tgz | nl -ba -nrz -w6 -s ' ' | sed 's/^/@P@ /'
log "PAYLOAD_END"
log "FINQA_JOB_DONE"
sleep 2400
kill -TERM 1
