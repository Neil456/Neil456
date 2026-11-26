#!/usr/bin/env bash
# Launch a RunPod GPU pod for the 9B QLoRA SFT. DRY RUN by default: prints the command and cost estimate.
# Requires: runpodctl configured with YOUR API key (not present in this environment), a network volume.
# Verify flag names with `runpodctl create pod --help` (CLI flags change between versions).
set -euo pipefail
GPU=${GPU:-"NVIDIA H100 80GB HBM3"}; PRICE_PER_HR=${PRICE_PER_HR:-2.99}; MAX_HOURS=${MAX_HOURS:-6}
MAX_USD=${MAX_USD:?set MAX_USD (hard budget cap in USD)}
IMAGE=${IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}
EST=$(python3 -c "print(round($PRICE_PER_HR*$MAX_HOURS,2))")
echo "Estimated worst-case cost: \$$EST (= $PRICE_PER_HR/hr x $MAX_HOURS h cap); budget cap \$$MAX_USD"
python3 -c "import sys; sys.exit(0 if $PRICE_PER_HR*$MAX_HOURS <= $MAX_USD else 1)" || { echo "ABORT: worst case exceeds MAX_USD"; exit 1; }
CMD=(runpodctl create pod --name finqa-sft --gpuType "$GPU" --gpuCount 1 --imageName "$IMAGE"
     --containerDiskSize 60 --volumeSize 150 --volumePath /workspace
     --env "STAGE=${STAGE:-all}" --env "MAX_HOURS=$MAX_HOURS"
     --args "bash -c 'bash /workspace/finqa-specialist/scripts/cloud/remote_job.sh'")
if [[ "${CONFIRM:-0}" != "1" ]]; then echo "DRY RUN (set CONFIRM=1 to launch):"; printf '%q ' "${CMD[@]}"; echo; exit 0; fi
"${CMD[@]}"
echo "Teardown: runpodctl stop pod <id> && runpodctl remove pod <id>   (network volume persists; delete separately)"
