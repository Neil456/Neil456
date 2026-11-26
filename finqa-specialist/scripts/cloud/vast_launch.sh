#!/usr/bin/env bash
# Vast.ai launch. DRY RUN by default. Requires `pip install vastai` and `vastai set api-key` (not configured here).
set -euo pipefail
MAX_USD=${MAX_USD:?set MAX_USD}; MAX_HOURS=${MAX_HOURS:-6}; MAX_DPH=$(python3 -c "print($MAX_USD/$MAX_HOURS)")
QUERY="gpu_name=H100_SXM num_gpus=1 reliability>0.98 inet_down>500 disk_space>=150 dph<=$MAX_DPH"
echo "Search: vastai search offers '$QUERY' -o 'dph'"
[[ "${CONFIRM:-0}" == "1" ]] || { echo "DRY RUN. Then: vastai create instance <OFFER_ID> --image pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel --disk 150 --env '-e STAGE=all -e MAX_HOURS=$MAX_HOURS' --onstart-cmd 'bash /workspace/finqa-specialist/scripts/cloud/remote_job.sh'"; echo "Teardown: vastai destroy instance <INSTANCE_ID>"; exit 0; }
vastai search offers "$QUERY" -o 'dph' | head -5
