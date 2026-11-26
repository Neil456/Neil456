"""GRPO / RLVR with TRL GRPOTrainer and the verifiable FinQA reward (finqa.train.rewards).

Starts from a base model or an SFT LoRA adapter (merged before RL). Prompts are the identical
evaluation prompts (variant "direct"), rendered by TRL with chat_template_kwargs
{"enable_thinking": false}. Reward = verified correctness (+ small grounding bonus, only when
correct); formatting alone earns nothing (see tests/test_reward.py).

Usage:
  python -m finqa.train.grpo --config configs/grpo_cpu_smoke.yaml      # tiny CPU smoke run
  python -m finqa.train.grpo --config configs/grpo_gpu_q35_9b.yaml     # prepared GPU run (vLLM)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for o in args.override:
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    from finqa.eval.metrics import load_jsonl
    from finqa.prompts import build_messages
    from finqa.train.rewards import make_trl_reward

    set_seed(cfg.get("seed", 0))
    if cfg.get("threads"):
        torch.set_num_threads(cfg["threads"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in cfg["train_files"]:
        rows += [r for r in load_jsonl(f) if r["split"] == "train" and not r["meta"].get("leak_doc")]
    rows.sort(key=lambda r: r["id"])
    random.Random(cfg.get("seed", 0)).shuffle(rows)
    rows = rows[cfg.get("skip_first", 0):]  # e.g. skip the examples the SFT init adapter was trained on
    if cfg.get("max_train_examples"):
        rows = rows[: cfg["max_train_examples"]]
    by_id = {r["id"]: r for r in rows}
    ds = Dataset.from_list([{"prompt": build_messages(r, "direct"), "example_id": r["id"]} for r in rows])

    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg.get("revision"), dtype=dtype)
    if cfg.get("init_adapter"):
        model = PeftModel.from_pretrained(model, cfg["init_adapter"]).merge_and_unload()

    gargs = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=cfg.get("learning_rate", 1e-6),
        per_device_train_batch_size=cfg.get("per_device_batch_size", 2),
        gradient_accumulation_steps=cfg.get("grad_accum", 1),
        num_generations=cfg.get("num_generations", 4),
        max_completion_length=cfg.get("max_completion_length", 96),
        temperature=cfg.get("temperature", 1.0),
        beta=cfg.get("beta", 0.0),
        loss_type=cfg.get("loss_type", "dapo"),
        scale_rewards=cfg.get("scale_rewards", "group"),
        mask_truncated_completions=cfg.get("mask_truncated_completions", True),
        max_steps=cfg.get("max_steps", -1),
        num_train_epochs=cfg.get("epochs", 1),
        logging_steps=1,
        save_steps=cfg.get("save_steps", 50),
        save_total_limit=cfg.get("save_total_limit", 3),
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        use_vllm=cfg.get("use_vllm", False),
        vllm_mode=cfg.get("vllm_mode", "colocate"),
        chat_template_kwargs={"enable_thinking": False},
        use_cpu=not torch.cuda.is_available(),
        report_to=[],
        seed=cfg.get("seed", 0),
        log_completions=cfg.get("log_completions", False),
    )
    lcfg = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32), lora_dropout=0.0,
                      target_modules=cfg.get("target_modules", "all-linear"), task_type="CAUSAL_LM")
    log_path = out_dir / "train_log.jsonl"

    class JsonlLogger(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if logs:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")

    trainer = GRPOTrainer(model=model, reward_funcs=[make_trl_reward(by_id)], args=gargs, train_dataset=ds,
                          processing_class=tok, peft_config=lcfg, callbacks=[JsonlLogger()])
    t0 = time.time()
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
    trainer.save_model(str(out_dir / "final"))
    wall = time.time() - t0
    n_completions = trainer.state.global_step * gargs.per_device_train_batch_size * gargs.gradient_accumulation_steps
    mem = {}
    if torch.cuda.is_available():
        mem = {"max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
               "gpu": torch.cuda.get_device_name(0)}
    (out_dir / "run_manifest.json").write_text(json.dumps({"config": cfg, "wall_s": wall,
                                                           "steps": trainer.state.global_step,
                                                           "completions_total": n_completions,
                                                           "completions_per_s": n_completions / max(wall, 1e-9),
                                                           "s_per_step": wall / max(trainer.state.global_step, 1),
                                                           "memory": mem, "train_ids_first": rows[0]["id"] if rows else None,
                                                           "log_tail": trainer.state.log_history[-5:]}, indent=2, default=str))
    print(f"[grpo] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
