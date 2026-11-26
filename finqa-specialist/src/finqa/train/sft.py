"""LoRA / QLoRA supervised fine-tuning with TRL SFTTrainer on canonical FinQA examples.

* Prompt = the exact evaluation prompt (finqa.prompts.build_messages, variant "direct") rendered
  with the model's native chat template (non-thinking), target = verified gold program + executed
  answer. Tokenization is done here so the training prefix is byte-identical to the inference
  prompt; prompt tokens are masked from the loss via ``completion_mask``.
* Examples longer than ``max_length`` are DROPPED (never truncated, so answers are never cut)
  and counted in the run manifest.
* Resumable: re-running with the same output_dir resumes from the last checkpoint.

Usage:  python -m finqa.train.sft --config configs/sft_cpu_q35_0.8b.yaml [--override key=value ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import yaml


def load_cfg(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(open(path))
    for o in overrides or []:
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def build_examples(cfg: dict, tok) -> tuple[list[dict], dict]:
    from finqa.eval.metrics import load_jsonl
    from finqa.prompts import build_messages, build_target

    rows = []
    for f in cfg["train_files"]:
        rows += load_jsonl(f)
    st = {"loaded": len(rows)}
    rows = [r for r in rows if r["split"] == "train" and r.get("program") and r["meta"].get("program_verified", True)
            and not r["meta"].get("leak_doc") and not r["meta"].get("leak_table")]
    st["usable_verified_no_leak"] = len(rows)
    rows.sort(key=lambda r: r["id"])
    random.Random(cfg.get("seed", 0)).shuffle(rows)
    if cfg.get("max_train_examples"):
        rows = rows[: cfg["max_train_examples"]]
    st["selected"] = len(rows)
    out, dropped = [], 0
    eos = cfg.get("eos_text", "<|im_end|>\n")
    for r in rows:
        msgs = build_messages(r, "direct")
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        target = build_target(r) + eos
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(target, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(c_ids) > cfg["max_length"]:
            dropped += 1
            continue
        out.append({"input_ids": p_ids + c_ids, "completion_mask": [0] * len(p_ids) + [1] * len(c_ids), "id": r["id"]})
    st["dropped_too_long"] = dropped
    st["kept"] = len(out)
    st["tokens_total"] = sum(len(x["input_ids"]) for x in out)
    st["completion_tokens_total"] = sum(sum(x["completion_mask"]) for x in out)
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.override)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.get("seed", 0))
    if cfg.get("threads"):
        torch.set_num_threads(cfg["threads"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train, st = build_examples(cfg, tok)
    print("[sft] data:", json.dumps(st), flush=True)
    ds = Dataset.from_list([{k: v for k, v in x.items() if k != "id"} for x in train])

    eval_ds = None
    if cfg.get("eval_ids_file"):
        from finqa.eval.metrics import load_jsonl
        from finqa.prompts import build_messages, build_target
        want = set(l.strip() for l in open(cfg["eval_ids_file"]) if l.strip())
        ev = []
        for r in load_jsonl(cfg["eval_file"]):
            if r["id"] in want and r.get("program"):
                prompt = tok.apply_chat_template(build_messages(r, "direct"), tokenize=False, add_generation_prompt=True, enable_thinking=False)
                p = tok(prompt, add_special_tokens=False)["input_ids"]
                c = tok(build_target(r) + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
                if len(p) + len(c) <= cfg["max_length"]:
                    ev.append({"input_ids": p + c, "completion_mask": [0] * len(p) + [1] * len(c)})
        eval_ds = Dataset.from_list(ev)
        print(f"[sft] eval examples: {len(ev)}", flush=True)

    quant = None
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    if cfg.get("load_in_4bit") and use_cuda:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg.get("revision"), dtype=dtype,
                                                 quantization_config=quant,
                                                 attn_implementation=cfg.get("attn_implementation", "sdpa"))
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg.get("lora_dropout", 0.0),
                      target_modules=cfg.get("target_modules", "all-linear"), task_type="CAUSAL_LM")

    sargs = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg.get("epochs", 1),
        max_steps=cfg.get("max_steps", -1),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        weight_decay=cfg.get("weight_decay", 0.0),
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 5),
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", cfg.get("save_steps", 50)),
        per_device_eval_batch_size=1,
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        max_length=cfg["max_length"],
        completion_only_loss=True,
        packing=False,
        seed=cfg.get("seed", 0),
        data_seed=cfg.get("seed", 0),
        report_to=[],
        dataloader_num_workers=0,
        use_cpu=not use_cuda,
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        optim=cfg.get("optim", "adamw_torch"),
        save_only_model=False,
    )

    log_path = out_dir / "train_log.jsonl"

    class JsonlLogger(TrainerCallback):
        def on_log(self, args_, state, control, logs=None, **kw):
            if logs:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")

    trainer = SFTTrainer(model=model, args=sargs, train_dataset=ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=lcfg, callbacks=[JsonlLogger()])
    trainer.model.print_trainable_parameters()

    # sanity: label masking — only completion tokens contribute
    batch = trainer.data_collator([ds[0]])
    n_lab = int((batch["labels"] != -100).sum())
    print(f"[sft] masking check: {n_lab} supervised tokens of {batch['input_ids'].shape[1]}", flush=True)
    assert 0 < n_lab < batch["input_ids"].shape[1]

    manifest = {"config": cfg, "data_stats": st, "torch": torch.__version__, "cuda": use_cuda,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train_ids_sha256": hashlib.sha256("\n".join(x["id"] for x in train).encode()).hexdigest()}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "train_ids.txt").write_text("\n".join(x["id"] for x in train) + "\n")

    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    resume = str(ckpts[-1]) if ckpts and cfg.get("resume", True) else None
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    wall = time.time() - t0
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    manifest.update({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "train_wall_s": wall,
                     "global_step": trainer.state.global_step,
                     "log_history_tail": trainer.state.log_history[-5:]})
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[sft] done in {wall/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
