"""Aggregate metrics, bootstrap confidence intervals and paired comparisons."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

METRICS = ["correct", "correct_lenient", "correct_exact5", "prog_correct", "prog_correct_exact5", "assisted_correct",
           "answer_parsed", "program_ok"]


def load_jsonl(path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def bootstrap_ci(xs: list[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += xs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def summarize(records: list[dict], n_boot: int = 2000) -> dict:
    recs = [r for r in records if r.get("scorable", True)]
    n = len(recs)
    out = {"n": n, "n_total_records": len(records)}
    for m in METRICS:
        xs = [1.0 if r.get(m) else 0.0 for r in recs]
        if not xs:
            continue
        lo, hi = bootstrap_ci(xs, n_boot)
        out[m] = {"mean": sum(xs) / n, "ci95": [lo, hi]}
    lat = sorted(r.get("latency_s", 0.0) for r in recs)
    if lat:
        out["latency_s"] = {"mean": sum(lat) / n, "p50": lat[n // 2], "p90": lat[min(n - 1, int(0.9 * n))]}
    ti = sum(r.get("input_tokens", 0) for r in recs)
    to = sum(r.get("output_tokens", 0) for r in recs)
    out["tokens"] = {"input_total": ti, "output_total": to, "input_mean": ti / max(n, 1), "output_mean": to / max(n, 1)}
    costs = [r.get("cost_usd") for r in recs if r.get("cost_usd") is not None]
    out["cost_usd_total"] = sum(costs) if costs else None
    out["truncated"] = sum(1 for r in recs if r.get("finish_reason") == "length")
    return out


def paired_bootstrap(a: list[dict], b: list[dict], metric: str = "correct", n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired comparison B - A on the intersection of example ids."""
    da = {r["id"]: r for r in a if r.get("scorable", True)}
    db = {r["id"]: r for r in b if r.get("scorable", True)}
    ids = sorted(set(da) & set(db))
    if not ids:
        return {"n": 0}
    diffs = [(1.0 if db[i].get(metric) else 0.0) - (1.0 if da[i].get(metric) else 0.0) for i in ids]
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    p_le0 = sum(1 for x in boots if x <= 0) / n_boot
    b_only = sum(1 for d in diffs if d > 0)
    a_only = sum(1 for d in diffs if d < 0)
    # exact McNemar (two-sided binomial) on discordant pairs
    k, m = min(a_only, b_only), a_only + b_only
    p_mcnemar = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m)
    return {"n": n, "metric": metric, "acc_a": sum(1 for i in ids if da[i].get(metric)) / n,
            "acc_b": sum(1 for i in ids if db[i].get(metric)) / n, "diff_b_minus_a": mean,
            "ci95": [lo, hi], "bootstrap_p_diff_le_0": p_le0, "b_only_correct": b_only, "a_only_correct": a_only,
            "mcnemar_exact_p": p_mcnemar}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Summarize or compare prediction JSONL files")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--paired", action="store_true", help="paired bootstrap of file2 vs file1")
    ap.add_argument("--metric", default="correct")
    args = ap.parse_args()
    if args.paired:
        a, b = load_jsonl(args.files[0]), load_jsonl(args.files[1])
        print(json.dumps(paired_bootstrap(a, b, args.metric), indent=2))
    else:
        for f in args.files:
            print(f)
            print(json.dumps(summarize(load_jsonl(f)), indent=2))


if __name__ == "__main__":
    main()
