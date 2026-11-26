"""Pick the checkpoint with the lowest validation loss (select60 subset, disjoint from reporting sets)
and symlink it as <run>/best. Usage: python scripts/select_checkpoint.py checkpoints/<run>"""
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
evals = {h["step"]: h["eval_loss"] for h in state["log_history"] if "eval_loss" in h}
avail = {int(p.name.split("-")[-1]): p for p in ckpts}
cands = {s: l for s, l in evals.items() if s in avail}
best = min(cands, key=cands.get)
link = run / "best"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(avail[best].name)
out = {"eval_loss_by_step": evals, "selected_step": best, "selected_eval_loss": cands[best],
       "criterion": "min eval_loss on finqa_dev_select60 (disjoint from val100/val300)"}
(run / "selection.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
