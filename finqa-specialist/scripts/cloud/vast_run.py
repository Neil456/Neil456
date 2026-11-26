"""Vast.ai launcher for the 9B QLoRA SFT + evaluation job, with a hard dollar cap.

  python scripts/cloud/vast_run.py plan                    # search offers, print choice + worst-case cost (no spend)
  python scripts/cloud/vast_run.py launch --confirm        # create instance (refuses if worst case > --max-usd)
  python scripts/cloud/vast_run.py watch                   # poll status/logs, enforce cap, fetch payload, destroy
  python scripts/cloud/vast_run.py destroy                 # destroy the instance recorded in the state file

Key handling: read from ~/.config/vastai/vast_api_key (never printed, never sent to the instance, never
committed). Sent as the ``api_key`` query parameter because this environment's proxy strips Authorization
headers to console.vast.ai. The code is shipped to the instance as a compressed bundle in an environment
variable (no git push required); results return through the instance log (base64 payload) and are rescored
locally with the same evaluator.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
API = "https://console.vast.ai/api/v0"
STATE = ROOT / "results/cloud/vast_state.json"
IMAGE = "pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel"
BUNDLE_FILES = [
    "pyproject.toml", "requirements-gpu.txt", "Makefile",
    "src/finqa/__init__.py", "src/finqa/dsl.py", "src/finqa/answers.py", "src/finqa/prompts.py",
    "src/finqa/data/__init__.py", "src/finqa/data/build.py",
    "src/finqa/eval/__init__.py", "src/finqa/eval/adapters.py", "src/finqa/eval/runner.py",
    "src/finqa/eval/scoring.py", "src/finqa/eval/metrics.py",
    "src/finqa/train/__init__.py", "src/finqa/train/sft.py",
    "src/finqa/sec/__init__.py", "src/finqa/sec/xbrl_eval.py",
    "src/finqa/synth/__init__.py",
    "scripts/download_data.sh", "scripts/select_checkpoint.py", "scripts/cloud/vast_job.sh", "scripts/make_tables.py",
    "configs/sft_gpu_q35_9b.yaml",
    "results/subsets/finqa_dev_val100.txt", "results/subsets/finqa_dev_select60.txt",
    "results/subsets/financebench_numeric.txt",
]


def key() -> str:
    p = Path(os.path.expanduser("~/.config/vastai/vast_api_key"))
    return p.read_text().strip()


def api(method: str, path: str, **kw):
    params = kw.pop("params", {}) or {}
    params["api_key"] = key()
    for attempt in range(4):
        try:
            r = requests.request(method, f"{API}{path}", params=params, timeout=60, **kw)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            return r
        except requests.RequestException as e:
            if attempt == 3:
                raise
            time.sleep(5 * 2 ** attempt)


def search(min_ram_gb: int, max_dph: float, gpus: list[str], bad_machines=()):
    q = {"num_gpus": {"eq": 1}, "rentable": {"eq": True}, "verified": {"eq": True},
         "reliability2": {"gt": 0.98}, "disk_space": {"gt": 120}, "inet_down": {"gt": 300},
         "gpu_ram": {"gte": min_ram_gb * 1000}, "cuda_max_good": {"gte": 12.8},
         "dph_total": {"lte": max_dph}, "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 50}
    r = requests.get(f"{API}/bundles/", params={"q": json.dumps(q)}, timeout=60)
    r.raise_for_status()
    # exclude hosts where Hugging Face / GitHub access is commonly blocked or throttled
    blocked = ("China", ", CN", "Hong Kong")
    offers = [o for o in r.json().get("offers", []) if (not gpus or o.get("gpu_name") in gpus)
              and not any(b in (o.get("geolocation") or "") for b in blocked)
              and o.get("machine_id") not in set(bad_machines)]
    return sorted(offers, key=lambda o: o["dph_total"])


def build_bundle() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz", preset=9) as tf:
        for f in BUNDLE_FILES:
            tf.add(ROOT / f, arcname=f)
    return base64.b64encode(buf.getvalue()).decode()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def cmd_plan(a):
    offers = search(a.min_gpu_ram, a.max_dph, a.gpus, load_state().get("bad_machine_ids", []))
    if not offers:
        sys.exit("no matching offers")
    for o in offers[:8]:
        print(f"  offer {o['id']}: {o['gpu_name']} {o['gpu_ram']/1000:.0f}GB ${o['dph_total']:.3f}/h "
              f"rel={o.get('reliability2',0):.3f} cuda={o.get('cuda_max_good')} {o.get('geolocation','?')} "
              f"disk={o.get('disk_space',0):.0f}GB down={o.get('inet_down',0):.0f}Mbps")
    best = offers[0]
    hours = min(a.max_hours, (a.max_usd - a.reserve_usd) / best["dph_total"])
    print(f"choice: offer {best['id']} {best['gpu_name']} at ${best['dph_total']:.3f}/h; hard cap {hours:.1f} h "
          f"-> worst case ${best['dph_total']*hours:.2f} (+ storage reserve ${a.reserve_usd}) <= ${a.max_usd}")
    return best, hours


def cmd_launch(a):
    s = load_state()
    if s.get("instance_id") and not s.get("destroyed"):
        sys.exit(f"instance {s['instance_id']} already recorded; run watch/destroy first")
    best, hours = cmd_plan(a)
    if best["dph_total"] * hours + a.reserve_usd > a.max_usd:
        sys.exit("ABORT: worst case exceeds --max-usd")
    bundle = build_bundle()
    print(f"bundle: {len(bundle)} base64 chars")
    # Vast limits: total env <= 32KB, onstart <= 16KB -> split the bundle across both.
    head, rest = bundle[:30000], bundle[30000:]
    onstart = ("mkdir -p /workspace/repo/finqa-specialist && cd /workspace/repo/finqa-specialist && "
               f"printf '%s%s' \"$CODE_B64\" '{rest}' | base64 -d | tar xJ && "
               "setsid nohup bash -c 'bash scripts/cloud/vast_job.sh 2>&1 | tee -a /workspace/job.log | tee /proc/1/fd/1' >/dev/null 2>&1 &")
    if len(onstart) > 15800:
        sys.exit(f"ABORT: onstart too long ({len(onstart)} chars); trim the bundle")
    env = {"CODE_B64": head, "MAX_HOURS": str(int(hours)), "EPOCHS": str(a.epochs),
           "HOURLY_USD": f"{best['dph_total']:.4f}", "PYTHONUNBUFFERED": "1"}
    if not a.confirm:
        print("DRY RUN: add --confirm to create the instance.")
        return
    body = {"client_id": "me", "image": IMAGE, "env": env, "disk": a.disk, "label": "finqa-9b-sft",
            "onstart": onstart, "runtype": "ssh"}
    r = api("PUT", f"/asks/{best['id']}/", json=body)
    print("create:", r.status_code, r.text[:500])
    if not r.ok:
        sys.exit("create failed (no instance created)")
    iid = r.json().get("new_contract")
    prev = load_state()
    history = prev.get("history", []) + ([{k: prev.get(k) for k in ("instance_id", "gpu", "dph", "created", "destroyed_at", "destroy_reason")}] if prev.get("instance_id") else [])
    save_state({"instance_id": iid, "offer_id": best["id"], "machine_id": best.get("machine_id"), "gpu": best["gpu_name"],
                "dph": best["dph_total"], "created": time.time(), "max_usd": a.max_usd, "max_hours": hours, "destroyed": False,
                "bad_machine_ids": prev.get("bad_machine_ids", []), "history": history})
    print(f"instance {iid} created; run: python scripts/cloud/vast_run.py watch")


def show(iid):
    r = api("GET", f"/instances/{iid}/")
    if not r.ok:
        return {"error": r.status_code}
    d = r.json()
    return d.get("instances", d) if isinstance(d, dict) else d


def fetch_logs(iid, tail="200000") -> str:
    r = api("PUT", f"/instances/request_logs/{iid}/", json={"tail": tail})
    if not r.ok:
        return ""
    url = r.json().get("result_url")
    if not url:
        return ""
    for _ in range(40):
        time.sleep(1.5)
        g = requests.get(url, timeout=60)
        if g.status_code == 200:
            return g.text
    return ""


def destroy(iid, why):
    r = api("DELETE", f"/instances/{iid}/")
    s = load_state()
    s.update({"destroyed": r.ok, "destroy_reason": why, "destroyed_at": time.time()})
    save_state(s)
    print(f"destroy {iid}: {r.status_code} ({why})")


def extract_payload(text: str) -> bool:
    chunks = {}
    for l in text.splitlines():
        m = re.search(r"@P@ (\d{6}) ([A-Za-z0-9+/=]+)\s*$", l)
        if m:
            chunks[int(m.group(1))] = m.group(2)  # numbered lines: duplicates in the log are harmless
    if not chunks or "PAYLOAD_END" not in text:
        return False
    if sorted(chunks) != list(range(1, max(chunks) + 1)):
        print(f"payload incomplete: {len(chunks)} of {max(chunks)} lines")
        return False
    raw = base64.b64decode("".join(chunks[i] for i in sorted(chunks)))
    out = ROOT / "results/cloud/payload"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        for m in tf.getmembers():
            name = Path(m.name).name
            if not m.isfile() or not re.fullmatch(r"[\w.\-]+", name):
                continue  # only plain files with safe names (payload is untrusted remote data)
            (out / name).write_bytes(tf.extractfile(m).read())
    print(f"payload extracted to {out}")
    return True


def cmd_watch(a):
    s = load_state()
    iid = s["instance_id"]
    last_print = ""
    while True:
        inst = show(iid)
        status = (inst or {}).get("actual_status") or (inst or {}).get("cur_state")
        elapsed_h = (time.time() - s["created"]) / 3600
        spent_est = elapsed_h * s["dph"]
        logs = fetch_logs(iid) if status == "running" else ""
        prog = [l for l in logs.splitlines() if l.startswith("[finqa-job") or "'loss'" in l or "eval_loss" in l]
        tail = "\n".join(prog[-6:])
        new = tail if tail != last_print else "(no new job lines)"
        print(f"--- {time.strftime('%H:%M:%S')} status={status} elapsed={elapsed_h:.2f}h est=${spent_est:.2f}\n{new}", flush=True)
        last_print = tail
        if logs:
            (ROOT / "results/cloud").mkdir(parents=True, exist_ok=True)
            (ROOT / "results/cloud/job_log_tail.txt").write_text("\n".join(l for l in logs.splitlines() if "@P@" not in l)[-200000:])
        if "FINQA_JOB_DONE" in logs and extract_payload(logs):
            destroy(iid, "job done, payload retrieved")
            return
        if spent_est + 0.3 >= s["max_usd"] or elapsed_h >= s["max_hours"] + 0.25:
            destroy(iid, "budget/time cap")
            return
        if status in ("exited", "offline", "unknown") and elapsed_h > 0.3:
            if "FINQA_JOB_DONE" not in logs:
                print("instance not running; fetching final logs before destroy")
            destroy(iid, f"status {status}")
            return
        if status in (None, "loading", "created") and elapsed_h > 0.5:
            destroy(iid, "stuck loading > 30 min")
            return
        time.sleep(a.poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "launch", "watch", "destroy", "status"])
    ap.add_argument("--max-usd", type=float, default=8.0)
    ap.add_argument("--reserve-usd", type=float, default=0.5)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--max-dph", type=float, default=1.0)
    ap.add_argument("--min-gpu-ram", type=int, default=40)
    ap.add_argument("--gpus", nargs="*", default=["L40S", "A100 SXM4", "A100 PCIE", "A100X", "RTX 6000Ada", "A6000", "L40"])
    ap.add_argument("--disk", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a)
    elif a.cmd == "launch":
        cmd_launch(a)
    elif a.cmd == "watch":
        cmd_watch(a)
    elif a.cmd == "status":
        s = load_state()
        print(json.dumps(s, indent=1))
        if s.get("instance_id"):
            i = show(s["instance_id"])
            print({k: (i or {}).get(k) for k in ("actual_status", "cur_state", "gpu_name", "dph_total", "status_msg")})
    elif a.cmd == "destroy":
        destroy(load_state()["instance_id"], "manual")


if __name__ == "__main__":
    main()
