#!/usr/bin/env bash
# Usage: scripts/eval_local.sh <name> <model> <revision> <subset-ids-file> [extra runner args...]
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HF_HOME=${HF_HOME:-/home/user/hf_cache} TOKENIZERS_PARALLELISM=false
name=$1; model=$2; rev=$3; ids=$4; shift 4
data=${DATA:-data/processed/finqa_dev.jsonl}
python -m finqa.eval.runner --adapter hf --model "$model" --revision "$rev" --data "$data" \
  --ids-file "$ids" --batch-size 1 --max-new-tokens ${MAX_NEW:-128} --out "results/preds/${name}.jsonl" "$@"
