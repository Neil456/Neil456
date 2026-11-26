"""Fresh evaluation set from recent SEC 10-K filings (XBRL company facts, data.sec.gov).

For each company we take the MOST RECENT 10-K in the XBRL companyfacts feed and extract annual
values (current fiscal year and prior year) for a fixed set of us-gaap concepts.

Independent checks (an item is kept only if all pass):
  1. Cross-filing agreement: the prior-year value reported as a comparative in the latest 10-K must
     equal the value reported as the current year in the PREVIOUS 10-K (different accession).
     Restated / inconsistent values are dropped.
  2. Two computation paths: the answer computed by the restricted DSL interpreter from the rendered
     table (USD millions) must match, within 1e-6 relative, an independent float computation from
     the raw XBRL values in USD.
Questions are templated (growth, margins, change in margin, R&D intensity, free cash flow, leverage),
so this is a *templated* evaluation on real, recent numbers -- it tests numerical reasoning on
filings after most model training cutoffs, not open-ended document understanding. Split = "sec_fresh",
grouped by company (group_id = CIK) for grouped bootstrap.

SEC fair-access: <= 5 requests/second, descriptive User-Agent (generic; no personal contact data is sent).
Usage: python -m finqa.sec.xbrl_eval --out data/processed/sec_fresh_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from finqa import dsl

UA = "finqa-research-prototype/0.1 (non-commercial academic research)"
CIKS = {
    320193: "Apple", 789019: "Microsoft", 1652044: "Alphabet", 1018724: "Amazon", 1326801: "Meta Platforms",
    1045810: "NVIDIA", 104169: "Walmart", 200406: "Johnson & Johnson", 80424: "Procter & Gamble",
    21344: "Coca-Cola", 77476: "PepsiCo", 50863: "Intel", 858877: "Cisco", 1341439: "Oracle", 51143: "IBM",
    354950: "Home Depot", 909832: "Costco", 320187: "Nike", 63908: "McDonald's", 18230: "Caterpillar",
    66740: "3M", 773840: "Honeywell", 78003: "Pfizer", 310158: "Merck", 1551152: "AbbVie", 93410: "Chevron",
    34088: "Exxon Mobil", 732712: "Verizon", 1166691: "Comcast", 1744489: "Walt Disney", 1090727: "UPS",
    936468: "Lockheed Martin", 1065280: "Netflix", 796343: "Adobe", 1108524: "Salesforce", 804328: "Qualcomm",
    97476: "Texas Instruments", 2488: "Advanced Micro Devices", 1800: "Abbott Laboratories", 59478: "Eli Lilly",
}
CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "cost of revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross profit": ["GrossProfit"],
    "research and development": ["ResearchAndDevelopmentExpense"],
    "operating income": ["OperatingIncomeLoss"],
    "net income": ["NetIncomeLoss"],
    "operating cash flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "total assets": ["Assets"],
    "total liabilities": ["Liabilities"],
}
INSTANT = {"total assets", "total liabilities"}


def fetch(client: httpx.Client, cik: int) -> dict:
    r = client.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json")
    r.raise_for_status()
    time.sleep(0.25)
    return r.json()


def _days(a: str, b: str) -> int:
    from datetime import date
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def annual_facts(facts: dict, names: list[str], instant: bool):
    """Return list of (concept, fact) for 10-K annual facts in USD across candidate concept names."""
    out = []
    g = facts.get("facts", {}).get("us-gaap", {})
    for n in names:
        for x in g.get(n, {}).get("units", {}).get("USD", []):
            if x.get("form") != "10-K":
                continue
            if instant:
                if "start" in x:
                    continue
            else:
                if "start" not in x or not (350 <= _days(x["start"], x["end"]) <= 380):
                    continue
            out.append((n, x))
    return out


def extract_company(facts: dict):
    """Pick latest and previous 10-K accessions; return {item: (cur, prev)} passing cross-filing check."""
    rev = annual_facts(facts, CONCEPTS["revenue"], False) or annual_facts(facts, CONCEPTS["net income"], False)
    accns = sorted({(x["filed"], x["accn"]) for _, x in rev})
    if len(accns) < 2:
        return None, {"reason": "fewer than two 10-K filings"}
    latest_accn, prev_accn = accns[-1][1], accns[-2][1]
    latest_filed = accns[-1][0]
    items, info = {}, {"latest_accn": latest_accn, "prev_accn": prev_accn, "filed": latest_filed, "dropped": {}}
    cur_end = None
    for item, names in CONCEPTS.items():
        inst = item in INSTANT
        best = None
        for n in names:  # concept priority order
            fs = [x for c, x in annual_facts(facts, [n], inst)]
            in_latest = [x for x in fs if x["accn"] == latest_accn]
            if not in_latest:
                continue
            ends = sorted({x["end"] for x in in_latest})
            if len(ends) < 2:
                continue
            cur = [x for x in in_latest if x["end"] == ends[-1]][0]
            prior = [x for x in in_latest if x["end"] == ends[-2]][0]
            prev_filing = [x for x in fs if x["accn"] == prev_accn and x["end"] == prior["end"]]
            if not prev_filing:
                info["dropped"][item] = "prior year not in previous filing"
                continue
            if prev_filing[0]["val"] != prior["val"]:
                info["dropped"][item] = "restated (cross-filing mismatch)"
                break
            best = (n, cur, prior)
            break
        if best:
            n, cur, prior = best
            if cur_end is None and not inst:
                cur_end = cur["end"]
            items[item] = {"concept": n, "cur": cur["val"], "prior": prior["val"], "cur_end": cur["end"],
                           "prior_end": prior["end"]}
    # keep items sharing the same fiscal period end as revenue/net income
    ref = items.get("revenue") or items.get("net income")
    if not ref:
        return None, {"reason": "no revenue/net income"}
    items = {k: v for k, v in items.items() if v["cur_end"][:4] == ref["cur_end"][:4]}
    info["fiscal_year_end"] = ref["cur_end"]
    info["prior_year_end"] = ref["prior_end"]
    return items, info


def mm(v: float) -> str:
    x = v / 1e6
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.2f}".rstrip("0").rstrip(".")


def build_items(name: str, cik: int, items: dict, info: dict) -> list[dict]:
    y1, y0 = info["fiscal_year_end"][:4], info["prior_year_end"][:4]
    header = ["( in millions of usd )", f"fiscal {y1}", f"fiscal {y0}"]
    table = [header]
    for k, v in items.items():
        table.append([k, mm(v["cur"]), mm(v["prior"])])
    ctx = {"pre_text": [f"{name.lower()} reported the following amounts in its annual report on form 10-k for the fiscal year ended {info['fiscal_year_end']} ( comparative prior-year amounts as reported in the same filing ) ."],
           "table": table,
           "post_text": ["capital expenditures represent payments to acquire property , plant and equipment ."]}
    cell = {k: (mm(v["cur"]), mm(v["prior"])) for k, v in items.items()}
    raw = {k: (float(v["cur"]), float(v["prior"])) for k, v in items.items()}
    qs = []

    def add(q, prog, ref_val):
        r = dsl.execute(prog, table, round_digits=None)
        if not r.ok:
            return
        if isinstance(ref_val, str):
            if r.value != ref_val:
                return
        elif abs(r.value - ref_val) > 1e-6 * max(1.0, abs(ref_val)):
            return  # two computation paths disagree -> drop
        qs.append((q, prog, round(r.value, 5) if not isinstance(r.value, str) else r.value))

    if "revenue" in cell:
        a, b = cell["revenue"]
        ra, rb = raw["revenue"]
        add(f"what was the percentage change in revenue from fiscal {y0} to fiscal {y1}?",
            f"subtract({a}, {b}), divide(#0, {b})", (ra - rb) / rb)
    if "net income" in cell:
        a, b = cell["net income"]
        ra, rb = raw["net income"]
        if rb != 0:
            add(f"what was the percentage change in net income from fiscal {y0} to fiscal {y1}?",
                f"subtract({a}, {b}), divide(#0, {b})", (ra - rb) / rb)
    if "operating income" in cell and "revenue" in cell:
        add(f"what was the operating margin in fiscal {y1}?",
            f"divide({cell['operating income'][0]}, {cell['revenue'][0]})", raw["operating income"][0] / raw["revenue"][0])
        add(f"by how much did the operating margin change from fiscal {y0} to fiscal {y1}?",
            f"divide({cell['operating income'][0]}, {cell['revenue'][0]}), divide({cell['operating income'][1]}, {cell['revenue'][1]}), subtract(#0, #1)",
            raw["operating income"][0] / raw["revenue"][0] - raw["operating income"][1] / raw["revenue"][1])
    if "gross profit" in cell and "revenue" in cell:
        add(f"what was the gross margin in fiscal {y0}?",
            f"divide({cell['gross profit'][1]}, {cell['revenue'][1]})", raw["gross profit"][1] / raw["revenue"][1])
    elif "cost of revenue" in cell and "revenue" in cell:
        add(f"what was the gross margin in fiscal {y1}, computed as revenue minus cost of revenue divided by revenue?",
            f"subtract({cell['revenue'][0]}, {cell['cost of revenue'][0]}), divide(#0, {cell['revenue'][0]})",
            (raw["revenue"][0] - raw["cost of revenue"][0]) / raw["revenue"][0])
    if "research and development" in cell and "revenue" in cell:
        add(f"what percentage of revenue was spent on research and development in fiscal {y1}?",
            f"divide({cell['research and development'][0]}, {cell['revenue'][0]})",
            raw["research and development"][0] / raw["revenue"][0])
    if "operating cash flow" in cell and "capital expenditures" in cell:
        add(f"what was free cash flow ( operating cash flow minus capital expenditures ) in fiscal {y1}, in millions?",
            f"subtract({cell['operating cash flow'][0]}, {cell['capital expenditures'][0]})",
            (raw["operating cash flow"][0] - raw["capital expenditures"][0]) / 1e6)
        if "net income" in cell and raw["net income"][0] > 0:
            add(f"what was the ratio of operating cash flow to net income in fiscal {y1}?",
                f"divide({cell['operating cash flow'][0]}, {cell['net income'][0]})",
                raw["operating cash flow"][0] / raw["net income"][0])
    if "total liabilities" in cell and "total assets" in cell:
        add(f"what proportion of total assets was funded by total liabilities at the end of fiscal {y1}?",
            f"divide({cell['total liabilities'][0]}, {cell['total assets'][0]})",
            raw["total liabilities"][0] / raw["total assets"][0])
    if "revenue" in cell and "net income" in cell and raw["revenue"][1] > 0 and raw["net income"][1] != 0:
        ga = (raw["revenue"][0] - raw["revenue"][1]) / raw["revenue"][1]
        gb = (raw["net income"][0] - raw["net income"][1]) / abs(raw["net income"][1])
        if raw["net income"][1] > 0:
            add(f"did net income grow faster than revenue from fiscal {y0} to fiscal {y1}?",
                f"subtract({cell['net income'][0]}, {cell['net income'][1]}), divide(#0, {cell['net income'][1]}), "
                f"subtract({cell['revenue'][0]}, {cell['revenue'][1]}), divide(#2, {cell['revenue'][1]}), greater(#1, #3)",
                "yes" if gb > ga else "no")
    out = []
    for i, (q, prog, val) in enumerate(qs):
        out.append({
            "id": f"sec:{cik}:{info['latest_accn']}:{i}", "source": "sec_fresh", "document_id": info["latest_accn"],
            "group_id": str(cik), "split": "sec_fresh", "context": ctx, "question": q, "answer": val,
            "answer_text": str(val), "unit": "yes_no" if isinstance(val, str) else "ratio_or_number", "scale": "millions",
            "program": prog,
            "provenance": {"dataset": "SEC EDGAR XBRL companyfacts", "url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                           "accession": info["latest_accn"], "previous_accession": info["prev_accn"], "filed": info["filed"],
                           "concepts": {k: v["concept"] for k, v in items.items()}},
            "meta": {"company": name, "fiscal_year_end": info["fiscal_year_end"], "checks": ["cross_filing", "two_path"],
                     "dropped_items": info["dropped"]},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/sec_fresh_eval.jsonl")
    ap.add_argument("--min-filed", default="2025-06-01", help="only use 10-Ks filed on/after this date")
    ap.add_argument("--report", default="results/sec_fresh_build.json")
    args = ap.parse_args()
    client = httpx.Client(headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=60)
    rows, report = [], {}
    for cik, name in CIKS.items():
        try:
            facts = fetch(client, cik)
        except Exception as e:
            report[name] = {"error": str(e)[:200]}
            continue
        items, info = extract_company(facts)
        if not items:
            report[name] = info
            continue
        if info["filed"] < args.min_filed:
            report[name] = {"skipped": f"latest 10-K filed {info['filed']} < {args.min_filed}"}
            continue
        its = build_items(name, cik, items, info)
        report[name] = {"filed": info["filed"], "fy_end": info["fiscal_year_end"], "n_items": len(its),
                        "dropped": info["dropped"]}
        rows += its
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path(args.report).write_text(json.dumps({"n_items": len(rows), "n_companies": len({r['group_id'] for r in rows}),
                                             "companies": report}, indent=2))
    print(len(rows), "items from", len({r["group_id"] for r in rows}), "companies")


if __name__ == "__main__":
    main()
