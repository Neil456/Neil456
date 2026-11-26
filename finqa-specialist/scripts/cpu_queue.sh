#!/usr/bin/env bash
# Sequential, resumable CPU experiment queue. Each step writes logs/<step>.log and a done-marker.
# Re-run to resume (completed steps are skipped; the SFT steps resume from their last checkpoint).
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/home/user/hf_cache} TOKENIZERS_PARALLELISM=false
mkdir -p logs
Q=Qwen/Qwen3.5-0.8B; R=2fc06364715b967f1860aea9cf38778875588b17
Q2=Qwen/Qwen3.5-2B; R2=15852e8c16360a2fea060d615a32b45270f8a8fc
step() { local n=$1; shift; if [ -f logs/done.$n ]; then echo "[queue] skip $n"; return 0; fi
  echo "[queue] $(date -u +%H:%M:%S) start $n"; if "$@" >> logs/$n.log 2>&1; then touch logs/done.$n; echo "[queue] $(date -u +%H:%M:%S) done $n";
  else echo "[queue] $(date -u +%H:%M:%S) FAILED $n (see logs/$n.log)"; exit 1; fi; }
while pgrep -f "finqa.eval.runner.*base_direct_val100" >/dev/null; do sleep 20; done
step overfit      bash -c "python -m finqa.train.sft --config configs/sft_overfit_smoke.yaml && python scripts/check_overfit.py"
step sft_main     python -m finqa.train.sft --config configs/sft_cpu_q35_0.8b.yaml
step select_main  python scripts/select_checkpoint.py checkpoints/sft_cpu_q35_0.8b
L=checkpoints/sft_cpu_q35_0.8b/best
step eval_sft_val100   scripts/eval_local.sh q35-0.8b_sft_direct_val100 $Q $R results/subsets/finqa_dev_val100.txt --lora $L
step eval_base_sec     env DATA=data/processed/sec_fresh_eval.jsonl scripts/eval_local.sh q35-0.8b_base_direct_sec_fresh $Q $R results/subsets/sec_fresh_all.txt
step eval_sft_sec      env DATA=data/processed/sec_fresh_eval.jsonl scripts/eval_local.sh q35-0.8b_sft_direct_sec_fresh $Q $R results/subsets/sec_fresh_all.txt --lora $L
step eval_base_fb      env DATA=data/processed/financebench_eval.jsonl scripts/eval_local.sh q35-0.8b_base_direct_financebench_numeric $Q $R results/subsets/financebench_numeric.txt
step eval_sft_fb       env DATA=data/processed/financebench_eval.jsonl scripts/eval_local.sh q35-0.8b_sft_direct_financebench_numeric $Q $R results/subsets/financebench_numeric.txt --lora $L
step tables1      python scripts/make_tables.py
step sft_synth    python -m finqa.train.sft --config configs/sft_cpu_q35_0.8b_synth.yaml
step select_synth python scripts/select_checkpoint.py checkpoints/sft_cpu_q35_0.8b_synth
step eval_sftsynth_val100 scripts/eval_local.sh q35-0.8b_sftsynth_direct_val100 $Q $R results/subsets/finqa_dev_val100.txt --lora checkpoints/sft_cpu_q35_0.8b_synth/best
step eval_2b_val100 scripts/eval_local.sh q35-2b_base_direct_val100 $Q2 $R2 results/subsets/finqa_dev_val100.txt
step tables2      python scripts/make_tables.py
echo "[queue] all done $(date -u)"
