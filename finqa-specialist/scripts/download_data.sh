#!/usr/bin/env bash
# Clone raw datasets at the commits recorded in results/data_manifest.json.
set -euo pipefail
cd "$(dirname "$0")/.."; mkdir -p data/raw; cd data/raw
clone() { [ -d "$1" ] || git clone --filter=blob:none "$2" "$1"; git -C "$1" checkout -q "$3"; }
clone FinQA https://github.com/czyssrs/FinQA 0f16e2867befa6840783e58be38c9efb9229d742
clone TAT-QA https://github.com/NExTplusplus/TAT-QA 870accc41953dcde885aabeb963d94aabdc0fbc3
clone ConvFinQA https://github.com/czyssrs/ConvFinQA cf3eed2d5984960bf06bb8145bcea5e80b0222a6
clone financebench https://github.com/patronus-ai/financebench cc39aeb4afdf33909ee1412188bf89035950c2eb
mkdir -p ConvFinQA_unzipped && unzip -o -q ConvFinQA/data.zip -d ConvFinQA_unzipped
