"""Evaluation runner CLI.

Examples
  # 20-example smoke test of a local model on FinQA dev
  python -m finqa.eval.runner --adapter hf --model Qwen/Qwen3.5-0.8B --revision <sha> \
      --data data/processed/finqa_dev.jsonl --subset smoke20 --out results/preds/q35_08b_base_smoke20.jsonl

  # API model (refuses to run without --allow-paid and a --max-usd hard cap)
  python -m finqa.eval.runner --adapter anthropic --model <exact-model-id> --price-in 3 --price-out 15 \
      --data data/processed/finqa_dev.jsonl --subset val100 --allow-paid --max-usd 5 --out ...

Features: deterministic greedy decoding by default; fixed seeded subsets; sqlite cache keyed by
(model, revision, lora, adapter, prompt messages, generation config); resumable JSONL output
(skips ids already present with the same config hash); per-example correctness, latency,
tokens, actual (API) or estimated (self-hosted) cost; run manifest + summary JSON.
The FinQA/TAT-QA *test* splits are refused unless --final-test is passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from finqa.eval.adapters import GenConfig, GenResult, make_adapter
from finqa.eval.metrics import load_jsonl, summarize
from finqa.eval.scoring import score_output
from finqa.prompts import build_messages

SUBSET_SIZES = {"smoke20": 20, "val100": 100, "val200": 200, "val300": 300}


def file_sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def select(rows: list[dict], subset: str | None, limit: int | None, seed: int = 0, ids_file: str | None = None) -> list[dict]:
    if ids_file:
        want = [l.strip() for l in open(ids_file) if l.strip()]
        by = {r["id"]: r for r in rows}
        return [by[i] for i in want if i in by]
    if subset in SUBSET_SIZES or limit:
        order = sorted(rows, key=lambda r: r["id"])
        random.Random(seed).shuffle(order)
        n = SUBSET_SIZES.get(subset, None) or limit
        return order[:n]
    return rows


class Cache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS gen (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key):
        row = self.db.execute("SELECT value FROM gen WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, val):
        self.db.execute("INSERT OR REPLACE INTO gen VALUES (?, ?)", (key, json.dumps(val)))
        self.db.commit()


def git_rev() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, choices=["hf", "openai_compat", "anthropic", "gemini"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--lora")
    ap.add_argument("--base-url")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--data", required=True)
    ap.add_argument("--subset", choices=list(SUBSET_SIZES) + ["full"], default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids-file")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--variant", default="direct", choices=["direct", "cot"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--reasoning-effort")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/gen_cache.sqlite")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--price-in", type=float, help="USD per 1M input tokens (API list price)")
    ap.add_argument("--price-out", type=float, help="USD per 1M output tokens (API list price)")
    ap.add_argument("--hourly-usd", type=float, help="hardware $/hour for self-hosted cost estimate")
    ap.add_argument("--allow-paid", action="store_true")
    ap.add_argument("--max-usd", type=float)
    ap.add_argument("--final-test", action="store_true", help="permit evaluation on held-out test splits")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    rows = load_jsonl(args.data)
    if any(r["split"] == "test" for r in rows) and not args.final_test:
        sys.exit("Refusing to evaluate on a test split without --final-test (freeze the configuration first).")
    sel = select(rows, args.subset, args.limit, args.subset_seed, args.ids_file)

    paid = args.adapter in ("anthropic", "gemini") or (args.adapter == "openai_compat" and args.base_url and
                                                        "localhost" not in args.base_url and "127.0.0.1" not in args.base_url)
    if paid:
        if not args.allow_paid or args.max_usd is None or args.price_in is None or args.price_out is None:
            sys.exit("Paid API run requires --allow-paid, --max-usd and --price-in/--price-out (no budget was authorized).")

    cfg = GenConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed,
                    enable_thinking=args.enable_thinking, reasoning_effort=args.reasoning_effort,
                    batch_size=args.batch_size)
    spec = {"adapter": args.adapter, "model": args.model, "revision": args.revision, "lora": args.lora,
            "base_url": args.base_url, "dtype": args.dtype, "threads": args.threads, "load_in_4bit": args.load_in_4bit,
            "price_in": args.price_in, "price_out": args.price_out}
    cfg_d = asdict(cfg)
    cfg_d.pop("batch_size")
    config_hash = hashlib.sha256(json.dumps({"spec": {k: spec[k] for k in ("adapter", "model", "revision", "lora", "load_in_4bit")},
                                             "cfg": cfg_d, "variant": args.variant}, sort_keys=True).encode()).hexdigest()[:16]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for r in load_jsonl(out):
            if r.get("config_hash") == config_hash:
                done.add(r["id"])
    todo = [r for r in sel if r["id"] not in done]
    print(f"[runner] {len(sel)} selected, {len(done)} already done, {len(todo)} to run; config {config_hash}", flush=True)

    cache = None if args.no_cache else Cache(args.cache)
    adapter = None
    spent = 0.0
    if out.exists() and paid:
        spent = sum((r.get("cost_usd") or 0) for r in load_jsonl(out) if r.get("config_hash") == config_hash)

    manifest = {"config_hash": config_hash, "spec": spec, "gen_config": asdict(cfg), "variant": args.variant,
                "data": args.data, "data_sha256": file_sha256(args.data), "subset": args.subset, "limit": args.limit,
                "subset_seed": args.subset_seed, "n_selected": len(sel), "code_git_rev": git_rev(),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.platform(),
                "cpu_count": os.cpu_count(), "tag": args.tag}

    # longest-first improves batching efficiency and surfaces OOM early
    def plen(r):
        return len(build_messages(r, args.variant)[1]["content"])
    todo.sort(key=plen, reverse=True)

    with open(out, "a") as fo:
        i = 0
        while i < len(todo):
            batch = todo[i:i + args.batch_size]
            msgs = [build_messages(r, args.variant) for r in batch]
            keys = [hashlib.sha256(json.dumps({"h": config_hash, "m": m}, sort_keys=True).encode()).hexdigest() for m in msgs]
            results: list[GenResult | None] = [None] * len(batch)
            if cache:
                for j, k in enumerate(keys):
                    c = cache.get(k)
                    if c:
                        results[j] = GenResult(**c)
            need = [j for j, r in enumerate(results) if r is None]
            if need:
                if adapter is None:
                    adapter = make_adapter(spec)
                    manifest["adapter"] = adapter.describe()
                for attempt in range(4):
                    try:
                        gen = adapter.generate([msgs[j] for j in need], cfg)
                        break
                    except Exception as e:  # retry with backoff (network / rate limits)
                        if attempt == 3:
                            raise
                        wait = 5 * 2 ** attempt
                        print(f"[runner] error {type(e).__name__}: {str(e)[:200]} -- retry in {wait}s", flush=True)
                        time.sleep(wait)
                for j, g in zip(need, gen):
                    results[j] = g
                    if cache:
                        cache.put(keys[j], asdict(g))
            for ex, g in zip(batch, results):
                rec = {"id": ex["id"], "source": ex["source"], "split": ex["split"], "group_id": ex.get("group_id"),
                       "config_hash": config_hash, "model": args.model, "revision": args.revision, "lora": args.lora,
                       "variant": args.variant, "output": g.text, "input_tokens": g.input_tokens,
                       "output_tokens": g.output_tokens, "latency_s": g.latency_s, "finish_reason": g.finish_reason,
                       "cost_usd": g.cost_usd,
                       "est_cost_usd": (g.latency_s * args.hourly_usd / 3600.0) if args.hourly_usd else None,
                       "tag": args.tag}
                rec.update(score_output(ex, g.text))
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                if g.cost_usd:
                    spent += g.cost_usd
            i += len(batch)
            if paid and spent >= args.max_usd:
                print(f"[runner] budget cap reached (${spent:.4f} >= ${args.max_usd}); stopping.", flush=True)
                break
            if i % max(args.batch_size * 5, 20) < args.batch_size:
                recs = [r for r in load_jsonl(out) if r.get("config_hash") == config_hash]
                s = summarize(recs, n_boot=200)
                print(f"[runner] {len(recs)}/{len(sel)} acc={s.get('correct',{}).get('mean',0):.3f} "
                      f"prog={s.get('prog_correct',{}).get('mean',0):.3f} parsed={s.get('answer_parsed',{}).get('mean',0):.3f}",
                      flush=True)

    recs = [r for r in load_jsonl(out) if r.get("config_hash") == config_hash and r["id"] in {x["id"] for x in sel}]
    summary = summarize(recs)
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["spent_usd"] = spent if paid else 0.0
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k in ("n", "correct", "prog_correct", "assisted_correct", "answer_parsed", "latency_s", "tokens", "truncated")}, indent=1))


if __name__ == "__main__":
    main()
