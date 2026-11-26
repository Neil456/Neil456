"""Rescore GPU-run outputs (results/cloud/payload/*.jsonl) locally with the same evaluator.

The payload carries raw model outputs plus timing/token counts; scoring is recomputed here against the local
canonical data so every number in REPORT.md comes from one evaluator implementation. Writes full records to
results/preds/<name>.jsonl (+ .summary.json) and reports ids missing from the local data.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finqa.eval.metrics import load_jsonl, summarize  # noqa: E402
from finqa.eval.scoring import score_output  # noqa: E402

DATA = {
    "finqa_dev": "data/processed/finqa_dev.jsonl",
    "val100": "data/processed/finqa_dev.jsonl",
    "sec_fresh": "data/processed/sec_fresh_eval.jsonl",
    "financebench_numeric": "data/processed/financebench_eval.jsonl",
}


def main():
    pay = ROOT / "results/cloud/payload"
    out_dir = ROOT / "results/preds"
    report = {}
    for f in sorted(pay.glob("q35-9b_*.jsonl")):
        key = next((k for k in DATA if f.stem.endswith(k)), None)
        if key is None:
            continue
        ex = {r["id"]: r for r in load_jsonl(ROOT / DATA[key])}
        recs, missing = [], 0
        for r in load_jsonl(f):
            e = ex.get(r["id"])
            if e is None:
                missing += 1
                continue
            rec = {k: r.get(k) for k in ("id", "config_hash", "model", "revision", "lora", "variant", "output",
                                         "input_tokens", "output_tokens", "latency_s", "finish_reason", "est_cost_usd")}
            rec.update({"source": e["source"], "split": e["split"], "group_id": e.get("group_id"), "tag": "gpu-vast-a100"})
            rec.update(score_output(e, r.get("output") or ""))
            if r.get("correct") is not None and rec.get("correct") != r.get("correct"):
                rec["remote_score_mismatch"] = True
            recs.append(rec)
        name = f.stem
        with open(out_dir / f"{name}.jsonl", "w") as fo:
            for rec in recs:
                fo.write(json.dumps(rec) + "\n")
        s = summarize(recs)
        (out_dir / f"{name}.jsonl.summary.json").write_text(json.dumps(s, indent=2))
        mism = sum(1 for r in recs if r.get("remote_score_mismatch"))
        report[name] = {"n": len(recs), "missing_ids": missing, "remote_local_score_mismatches": mism,
                        "raw": s.get("correct", {}).get("mean"), "prog": s.get("prog_correct", {}).get("mean")}
    (ROOT / "results/cloud/rescore_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
