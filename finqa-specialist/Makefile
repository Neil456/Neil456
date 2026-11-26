# One reproducible command per stage. CPU-only by default; GPU stages are prepared (see NEXT_STEPS.md).
PY := .venv/bin/python
export HF_HOME ?= /home/user/hf_cache
export TOKENIZERS_PARALLELISM := false
Q08 := Qwen/Qwen3.5-0.8B
Q08_REV := 2fc06364715b967f1860aea9cf38778875588b17

.PHONY: setup download data synth sec test smoke eval-base eval-val100 sft-smoke sft-cpu eval-sft grpo-smoke report-tables cloud-dryrun

setup:            ## create venv and install pinned CPU deps
	uv venv --python 3.13 .venv && uv pip install --python .venv -r requirements-cpu.lock.txt && uv pip install --python .venv -e .

download:         ## clone raw datasets at pinned commits
	bash scripts/download_data.sh

data:             ## build canonical JSONL + manifest (hashes, dedup, leakage stats)
	$(PY) -m finqa.data.build --raw data/raw --out data/processed

synth:            ## verified synthetic pilot (1600 items) + audit report
	$(PY) -m finqa.synth.generator --n-docs 400 --out data/synth/pilot.jsonl --audit docs/SYNTH_AUDIT.md

sec:              ## fresh templated eval from recent 10-K XBRL facts (data.sec.gov)
	$(PY) -m finqa.sec.xbrl_eval --out data/processed/sec_fresh_eval.jsonl

test:
	$(PY) -m pytest -q

smoke:            ## 20-example smoke eval of base 0.8B on FinQA dev
	scripts/eval_local.sh q35-0.8b_base_direct_smoke20 $(Q08) $(Q08_REV) results/subsets/finqa_dev_smoke20.txt

eval-val100:      ## 100-example FinQA dev eval of base 0.8B
	scripts/eval_local.sh q35-0.8b_base_direct_val100 $(Q08) $(Q08_REV) results/subsets/finqa_dev_val100.txt

sft-smoke:        ## tiny overfit test (loss, grads, checkpoint save, reload + generate)
	$(PY) -m finqa.train.sft --config configs/sft_overfit_smoke.yaml && $(PY) scripts/check_overfit.py

sft-cpu:          ## CPU LoRA SFT proxy run (resumable)
	$(PY) -m finqa.train.sft --config configs/sft_cpu_q35_0.8b.yaml

eval-sft:         ## eval the selected CPU SFT checkpoint on the same val100 ids (paired with base)
	scripts/eval_local.sh q35-0.8b_sft_direct_val100 $(Q08) $(Q08_REV) results/subsets/finqa_dev_val100.txt --lora checkpoints/sft_cpu_q35_0.8b/best

grpo-smoke:
	$(PY) -m finqa.train.grpo --config configs/grpo_cpu_smoke.yaml

report-tables:    ## summary tables + paired bootstrap comparisons -> results/summary.md
	$(PY) scripts/make_tables.py

cloud-dryrun:     ## print GPU launch command + worst-case cost (never launches without CONFIRM=1)
	MAX_USD=20 bash scripts/cloud/runpod_launch.sh
