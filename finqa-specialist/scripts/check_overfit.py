"""Verify the overfit smoke test: loss fell, grad norms finite/nonzero, checkpoint reloads, and the
reloaded adapter reproduces the memorised training targets with greedy decoding."""
import json
import math
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finqa.eval.metrics import load_jsonl  # noqa: E402
from finqa.prompts import build_messages, build_target  # noqa: E402
from finqa.eval.scoring import score_output  # noqa: E402

out = Path("checkpoints/overfit_smoke")
logs = [json.loads(l) for l in open(out / "train_log.jsonl")]
losses = [l["loss"] for l in logs if "loss" in l]
gns = [l["grad_norm"] for l in logs if "grad_norm" in l]
res = {"first_loss": losses[0], "last_loss": losses[-1], "min_loss": min(losses), "n_logged": len(losses),
       "grad_norm_first": gns[0], "grad_norm_last": gns[-1], "grad_norm_all_finite": all(math.isfinite(g) for g in gns),
       "grad_norm_nonzero": all(g > 0 for g in gns)}
cfg = json.loads((out / "run_manifest.json").read_text())["config"]
ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
res["checkpoints"] = [p.name for p in ckpts]
ids = [l.strip() for l in open(out / "train_ids.txt") if l.strip()]
rows = {r["id"]: r for r in load_jsonl("data/processed/finqa_train.jsonl")}
tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
base = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, str(out / "final")).merge_and_unload().eval()
exact, correct = 0, 0
for i in ids:
    r = rows[i]
    p = tok.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(p, return_tensors="pt")
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=80, do_sample=False)
    text = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    exact += int(text == build_target(r))
    correct += int(score_output(r, text)["prog_correct"])
    print(repr(text[:120]), "| target:", repr(build_target(r)[:120]))
res.update({"reload_generate_exact_match": f"{exact}/{len(ids)}", "reload_prog_correct": f"{correct}/{len(ids)}"})
res["pass"] = bool(res["last_loss"] < 0.25 * res["first_loss"] and res["grad_norm_all_finite"] and correct >= len(ids) - 1)
print(json.dumps(res, indent=2))
Path("results/overfit_smoke.json").write_text(json.dumps(res, indent=2))
sys.exit(0 if res["pass"] else 1)
