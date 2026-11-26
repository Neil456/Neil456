#!/usr/bin/env bash
# Clone raw datasets at the commits recorded in results/data_manifest.json.
# FinanceBench is fetched with a sparse checkout of data/ only (its repo also holds ~360 10-K PDFs).
# Every network step has a timeout and prints progress, so a stuck download is visible.
set -euo pipefail
export GIT_TERMINAL_PROMPT=0
cd "$(dirname "$0")/.."; mkdir -p data/raw; cd data/raw
clone() {
  local dir=$1 url=$2 rev=$3
  if [ ! -d "$dir/.git" ]; then
    echo "[download] cloning $dir"
    timeout 900 git clone -q --filter=blob:none --no-checkout "$url" "$dir"
  fi
  if [ "$dir" = financebench ]; then
    git -C "$dir" sparse-checkout set --no-cone '/data/*.jsonl' '/README.md'
  fi
  timeout 900 git -C "$dir" checkout -q "$rev"
  echo "[download] $dir at $(git -C "$dir" rev-parse --short HEAD)"
}
clone FinQA https://github.com/czyssrs/FinQA 0f16e2867befa6840783e58be38c9efb9229d742
clone TAT-QA https://github.com/NExTplusplus/TAT-QA 870accc41953dcde885aabeb963d94aabdc0fbc3
clone ConvFinQA https://github.com/czyssrs/ConvFinQA cf3eed2d5984960bf06bb8145bcea5e80b0222a6
clone financebench https://github.com/patronus-ai/financebench cc39aeb4afdf33909ee1412188bf89035950c2eb
# extract with Python's stdlib (no dependency on the image having `unzip`)
mkdir -p ConvFinQA_unzipped && python3 -c "import zipfile; zipfile.ZipFile('ConvFinQA/data.zip').extractall('ConvFinQA_unzipped')"
echo "[download] done"
