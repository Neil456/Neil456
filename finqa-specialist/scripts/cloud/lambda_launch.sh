#!/usr/bin/env bash
# Lambda Cloud API launch. DRY RUN by default. Requires LAMBDA_API_KEY (not configured here) and an SSH key name.
set -euo pipefail
MAX_USD=${MAX_USD:?set MAX_USD}; PRICE_PER_HR=${PRICE_PER_HR:-4.29}; MAX_HOURS=${MAX_HOURS:-6}
python3 -c "import sys; sys.exit(0 if $PRICE_PER_HR*$MAX_HOURS <= $MAX_USD else 1)" || { echo "ABORT: worst case exceeds MAX_USD"; exit 1; }
BODY='{"region_name":"us-east-1","instance_type_name":"gpu_1x_h100_sxm5","ssh_key_names":["'"${SSH_KEY_NAME:-mykey}"'"],"file_system_names":["finqa-fs"],"quantity":1}'
echo "POST https://cloud.lambdalabs.com/api/v1/instance-operations/launch  $BODY"
[[ "${CONFIRM:-0}" == "1" ]] || { echo "DRY RUN. Teardown: POST /api/v1/instance-operations/terminate {\"instance_ids\":[\"<id>\"]}"; exit 0; }
curl -sS -u "$LAMBDA_API_KEY:" -H 'Content-Type: application/json' -d "$BODY" https://cloud.lambdalabs.com/api/v1/instance-operations/launch
