# Phase 0 decisions and provenance (recorded 2026-11-26)

## Environment actually available
* Cloud container: 4 vCPU (Intel Xeon w/ AVX-512 + AMX-BF16), 15 GiB RAM, ~60 GB free disk, **no GPU**.
* No Kaggle / Colab / RunPod / Vast / Lambda credentials; no OpenAI / Anthropic / Gemini API keys configured.
  (Generic AWS credential variables exist in the container environment; they were not inspected or used.)
* Network: originally HF / SEC / GPU providers blocked. With the user's approval a personal environment
  `neil456-finqa` was created allowing Hugging Face hosts + `www.sec.gov`, `data.sec.gov`.
  `www.sec.gov` still returns 403 (requires contact UA); `data.sec.gov` XBRL API works with a generic UA.
* **Spend: $0.** No paid provisioning or paid API calls were made (none authorized).

## Model selection (Hugging Face API, 2026-11-26)
| role | repo | revision | license | params | notes |
|---|---|---|---|---|---|
| primary fine-tune target (GPU, prepared) | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Apache-2.0 | 9.65B (8.95B text) | newest dense Qwen in 7–14B range (Feb 2026); hybrid Gated-DeltaNet + gated attention, 262k ctx, thinking on by default (`enable_thinking=False` for direct mode) |
| CPU proxy (run here) | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | Apache-2.0 | 0.87B (0.75B text) | same architecture family; non-thinking by default |
| CPU secondary baseline | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | Apache-2.0 | 2.27B | non-thinking by default |
| strong open baseline (GPU, prepared) | `Qwen/Qwen3.8-27B` | `37561816e66b5b9153b1b64c10968c18c1e6f4a0` | Apache-2.0 | 27.8B | Aug 2026, same `qwen3_5` architecture class; also `Qwen/Qwen3.6-27B` |

Not assumed newest: Qwen org listing (sorted by creation) shows Qwen3.6 (Apr 2026) and Qwen3.8 (Aug 2026)
only at 27B / MoE sizes; no newer Qwen dense model between 4B and 27B than Qwen3.5-9B.
Sources: `https://huggingface.co/api/models?author=Qwen&sort=createdAt&direction=-1`,
`https://huggingface.co/api/models/Qwen/Qwen3.5-9B`, model cards and `config.json` of each repo.

Software: `transformers==5.24.1` (qwen3_5 support starts at v5.2.0; 4.57 lacks it), `trl==1.19.0`,
`peft==0.22.0`, `torch==2.9.1`. Exact CPU environment: `requirements-cpu.lock.txt`.
On CPU, Qwen3.5 linear-attention layers use the reference PyTorch kernels (no `flash-linear-attention` /
`causal-conv1d`); the GPU container installs those kernels (`requirements-gpu.txt`).

## Data provenance and licensing
| dataset | commit | license | use here |
|---|---|---|---|
| FinQA (czyssrs/FinQA) | `0f16e28` | MIT (repo LICENSE) | train / dev / test (official splits) |
| TAT-QA (NExTplusplus/TAT-QA) | `870accc` | CC BY 4.0 (README) | secondary; dev eval prepared, numeric subset ingested |
| ConvFinQA (czyssrs/ConvFinQA) | `cf3eed2` | MIT | ingested for leakage analysis only (documents derive from FinQA) |
| FinanceBench open-source (patronus-ai/financebench) | `cc39aeb` | **CC BY-NC 4.0** (HF dataset card) | external evaluation only; non-commercial license -> never trained on |
| SEC EDGAR XBRL company facts | live API 2026-11-26 | US government public data | fresh templated evaluation set |
File SHA-256 hashes of every raw and processed file: `results/data_manifest.json`.

## Teacher outputs
No frontier-model outputs were generated or used for training. Provider terms (OpenAI Terms of Use /
Business Terms; Anthropic Commercial Terms) prohibit using outputs to develop competing models (OpenAI
carves out non-distributed classifiers; Anthropic permits non-competing models such as classifiers). Google
Gemini API terms were not verified in this session. Conclusion: teacher distillation from these APIs for a
commercially intended FinQA model is **not** assumed permitted; synthetic data here is generated
programmatically (no LLM).

## GPU options and price assumptions (third-party snapshots, Nov 2026; verify live before launch)
| provider | GPU | $/hr (range seen) |
|---|---|---|
| RunPod | H100 80GB (SXM/PCIe) | 1.99–4.39 (community vs secure; snapshots disagree) |
| RunPod | A100 80GB | 1.39–2.19 |
| RunPod | L40S 48GB | 0.79–1.09 |
| RunPod | RTX 4090 24GB | 0.34–0.69 |
| Vast.ai (marketplace) | H100 SXM | median ~1.93 (min ~1.73), Nov 16 2026 snapshot |
| Vast.ai | A100 SXM4 80GB | median ~0.84 |
| Vast.ai | RTX 4090 | median ~0.50 |
| Lambda | H100 SXM | 3.99–4.29 on-demand |
| Lambda | A100 40GB / 80GB | 1.99 / ~2.79 |
Free tiers (Kaggle, Colab) require accounts not available here and do not survive unattended sessions.

## Key methodological decisions
* Gold = FinQA `exe_ans` (official). ~7.7% of dev items have answer strings materially inconsistent with
  `exe_ans` (label noise; reported).
* Primary metric: final-answer accuracy, 1% relative tolerance, percent-aware, sign-sensitive; also lenient,
  exact-5 (official style) and program-executed (tool-assisted) scores, reported separately.
* Checkpoint selection on `finqa_dev_select60` (dev ids disjoint from reporting subsets) by validation loss.
* Test split refused by the runner unless `--final-test`; **not used** in this session.
