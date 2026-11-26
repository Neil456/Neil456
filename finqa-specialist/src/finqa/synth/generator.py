"""Verified synthetic FinQA-style data from programmatically constructed financial statements.

Pipeline (programs first, language second):
 1. Build a realistic multi-year statement for a fictional company: revenue path with growth,
    cost structure, operating income, interest, taxes, net income (losses allowed), cash flow items
    (capex outflows negative), segments, share counts. Accounting identities hold by construction.
 2. Render it FinQA-style: pre_text sentences (some figures only in text, possibly in a different
    scale), a table (FinQA cell conventions such as "$ 1234" and "-56 ( 56 )"), post_text.
 3. Sample a program template (ratio, growth, margin, change in margin, average, weighted average,
    CAGR, share of total, unit conversion, multi-step comparison, table aggregates), bind it to
    concrete cells, execute it with the restricted interpreter -> gold answer.
 4. Only then realise a natural-language question from one of several phrasings of that template.
 5. Automatic checks: program executes; every literal appears in the rendered document;
    referenced row names unique; years unambiguous; no zero/near-zero denominators; answer not a
    verbatim table cell for multi-step templates; |answer| bounded; no duplicate questions per doc.
    Rejected items are counted by reason.
 6. Document-level split (train / synth_dev) by hash of the document id; difficulty bucket by steps.

Automated checks do NOT guarantee semantic correctness; see docs/SYNTH_AUDIT.md (human review).

Usage: python -m finqa.synth.generator --n-docs 400 --out data/synth/pilot.jsonl --audit docs/SYNTH_AUDIT.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from finqa import dsl
from finqa.prompts import render_document

SECTORS = {
    "software": dict(gm=(0.65, 0.85), opex=(0.35, 0.6), growth=(-0.05, 0.35), capex=(0.02, 0.06)),
    "retail": dict(gm=(0.22, 0.42), opex=(0.15, 0.3), growth=(-0.08, 0.12), capex=(0.02, 0.05)),
    "industrial": dict(gm=(0.2, 0.38), opex=(0.08, 0.2), growth=(-0.1, 0.12), capex=(0.04, 0.09)),
    "utilities": dict(gm=(0.3, 0.5), opex=(0.1, 0.2), growth=(-0.03, 0.06), capex=(0.15, 0.3)),
    "biotech": dict(gm=(0.5, 0.9), opex=(0.5, 1.4), growth=(-0.2, 0.6), capex=(0.03, 0.08)),
}
NAME_A = ["Aster", "Boreal", "Cobalt", "Delmar", "Everline", "Fairmont", "Granite", "Halcyon", "Ironwood", "Juniper",
          "Kestrel", "Larkspur", "Meridian", "Northgate", "Oakridge", "Pinecrest", "Quarry", "Redfern", "Sablewood",
          "Tidewater", "Umber", "Vantage", "Westmark", "Yarrow", "Zephyr"]
NAME_B = ["Holdings", "Systems", "Industries", "Group", "Corp", "Technologies", "Partners", "Brands", "Energy", "Labs"]
SEGMENTS = [["north america", "europe", "asia pacific"], ["products", "services", "licensing"],
            ["consumer", "commercial", "government"]]


def fmt_cell(v: float, dollar: bool = False, decimals: int = 0) -> str:
    """FinQA-style table cell: '$ 1234', '-56 ( 56 )'."""
    s = f"{abs(v):.{decimals}f}"
    if v < 0:
        body = f"-{s} ( {s} )"
    else:
        body = s
    return f"$ {body}" if dollar else body


def fmt_text_num(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}" if decimals else f"{v:.0f}"


class Doc:
    def __init__(self, rng: random.Random, idx: int):
        self.rng = rng
        self.sector = rng.choice(list(SECTORS))
        p = SECTORS[self.sector]
        self.company = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)}"
        self.doc_id = f"synth/{self.sector}/{idx:05d}"
        end_year = rng.randint(2016, 2024)
        n_years = rng.choice([2, 3, 3])
        self.years = [end_year - i for i in range(n_years)]  # most recent first (FinQA style)
        self.scale = rng.choice(["millions", "thousands"])
        base_rev = rng.uniform(80, 9000) if self.scale == "millions" else rng.uniform(20000, 900000)
        revs = []
        r = base_rev
        for _ in self.years[::-1]:
            revs.append(r)
            r = r * (1 + rng.uniform(*p["growth"]))
        revs = revs[::-1]  # aligned with self.years (recent first)
        # round to integers like real filings
        rows = {}
        rows["total revenues"] = [round(x) for x in revs]
        gm = [rng.uniform(*p["gm"]) for _ in self.years]
        rows["cost of revenues"] = [round(rv * (1 - g)) for rv, g in zip(rows["total revenues"], gm)]
        rows["gross profit"] = [a - b for a, b in zip(rows["total revenues"], rows["cost of revenues"])]
        opx = [rng.uniform(*p["opex"]) for _ in self.years]
        rows["operating expenses"] = [round(rv * o) for rv, o in zip(rows["total revenues"], opx)]
        rows["operating income"] = [a - b for a, b in zip(rows["gross profit"], rows["operating expenses"])]
        rows["interest expense"] = [round(abs(rv) * rng.uniform(0.002, 0.03)) for rv in rows["total revenues"]]
        pretax = [a - b for a, b in zip(rows["operating income"], rows["interest expense"])]
        tax_rate = rng.uniform(0.15, 0.3)
        rows["income tax expense"] = [round(max(0, x) * tax_rate) for x in pretax]
        rows["net income"] = [a - b for a, b in zip(pretax, rows["income tax expense"])]
        rows["capital expenditures"] = [-round(rv * rng.uniform(*p["capex"])) for rv in rows["total revenues"]]
        cfo = [round(ni + rv * rng.uniform(0.02, 0.12)) for ni, rv in zip(rows["net income"], rows["total revenues"])]
        rows["net cash provided by operating activities"] = cfo
        rows["free cash flow"] = [a + b for a, b in zip(cfo, rows["capital expenditures"])]
        self.rows = rows
        self.segments = rng.choice(SEGMENTS)
        w = [rng.uniform(0.15, 0.6) for _ in self.segments]
        tot = sum(w)
        seg_rev = [round(rows["total revenues"][0] * x / tot) for x in w[:-1]]
        seg_rev.append(rows["total revenues"][0] - sum(seg_rev))
        self.seg_rev = seg_rev
        self.seg_margin = [round(rng.uniform(0.05, 0.45) * 100, 1) for _ in self.segments]
        self.shares = round(rng.uniform(40, 900), 1)  # millions of shares, in text
        self.employees = rng.randint(800, 90000)
        self.table_rows = [k for k in ["total revenues", "cost of revenues", "gross profit", "operating expenses",
                                       "operating income", "interest expense", "income tax expense", "net income",
                                       "capital expenditures", "net cash provided by operating activities"]
                           if rng.random() > 0.12 or k in ("total revenues", "net income", "operating income")]

    def context(self) -> dict:
        unit = f"( in {self.scale} )"
        header = [unit] + [f"{y}" for y in self.years]
        table = [header]
        for k in self.table_rows:
            table.append([k] + [fmt_cell(v, dollar=(k in ("total revenues", "net income"))) for v in self.rows[k]])
        y0 = self.years[0]
        seg_txt = ", ".join(f"{s} contributed $ {v} {self.scale} with a segment margin of {m}% ( {m} % )"
                            for s, v, m in zip(self.segments, self.seg_rev, self.seg_margin))
        pre = [
            f"{self.company.lower()} is a {self.sector} company .",
            f"the following table summarizes our consolidated results of operations for the years ended december 31 , {', '.join(map(str, self.years))} .",
            f"in {y0} , {seg_txt} .",
        ]
        fcf = self.rows["free cash flow"][0]
        post = [
            f"as of december 31 , {y0} , we had approximately {self.employees} employees and {self.shares} million weighted average diluted shares outstanding .",
            f"free cash flow , defined as net cash provided by operating activities less capital expenditures , was $ {fmt_text_num(fcf, 0)} {self.scale} in {y0} .",
        ]
        return {"pre_text": pre, "table": table, "post_text": post}


# --------------------------------------------------------------------------- templates
# Each template returns (program, question_variants, needs) or None if not applicable.
def _pick_row(doc, rng, exclude=()):
    cands = [k for k in doc.table_rows if k not in exclude]
    return rng.choice(cands) if cands else None


def t_growth(doc, rng):
    if len(doc.years) < 2:
        return None
    k = _pick_row(doc, rng)
    i = rng.randrange(len(doc.years) - 1)
    a, b = doc.rows[k][i], doc.rows[k][i + 1]
    y1, y0 = doc.years[i], doc.years[i + 1]
    prog = f"subtract({a}, {b}), divide(#0, {b})"
    qs = [f"what was the percentage change in {k} from {y0} to {y1}?",
          f"what is the growth rate of {k} between {y0} and {y1}?",
          f"by what percent did {k} change from {y0} to {y1}?"]
    return prog, qs, "growth"


def t_margin(doc, rng):
    k = rng.choice([x for x in ("gross profit", "operating income", "net income") if x in doc.table_rows] or [None])
    if not k or "total revenues" not in doc.table_rows:
        return None
    i = rng.randrange(len(doc.years))
    prog = f"divide({doc.rows[k][i]}, {doc.rows['total revenues'][i]})"
    name = {"gross profit": "gross margin", "operating income": "operating margin", "net income": "net profit margin"}[k]
    qs = [f"what was the {name} in {doc.years[i]}?", f"what percentage of total revenues was {k} in {doc.years[i]}?"]
    return prog, qs, "margin"


def t_margin_change(doc, rng):
    if len(doc.years) < 2 or "total revenues" not in doc.table_rows:
        return None
    k = rng.choice([x for x in ("gross profit", "operating income") if x in doc.table_rows] or [None])
    if not k:
        return None
    R = doc.rows
    prog = (f"divide({R[k][0]}, {R['total revenues'][0]}), divide({R[k][1]}, {R['total revenues'][1]}), "
            f"subtract(#0, #1)")
    qs = [f"what was the change in {k} as a percentage of total revenues from {doc.years[1]} to {doc.years[0]}?",
          f"by how many percentage points did the ratio of {k} to total revenues change between {doc.years[1]} and {doc.years[0]}?"]
    return prog, qs, "margin_change"


def t_average(doc, rng):
    if len(doc.years) < 2:
        return None
    k = _pick_row(doc, rng)
    prog = f"add({doc.rows[k][0]}, {doc.rows[k][1]}), divide(#0, const_2)"
    qs = [f"what was the average {k} for {doc.years[1]} and {doc.years[0]}?",
          f"what is the mean value of {k} over {doc.years[1]} and {doc.years[0]}?"]
    return prog, qs, "average"


def t_table_agg(doc, rng):
    k = _pick_row(doc, rng)
    op, phr = rng.choice([("table_sum", "total"), ("table_average", "average"), ("table_max", "highest"), ("table_min", "lowest")])
    prog = f"{op}({k}, none)"
    span = f"{doc.years[-1]} through {doc.years[0]}"
    qs = [f"what was the {phr} {k} from {span}?", f"across {span} , what is the {phr} value of {k}?"]
    return prog, qs, f"table_{phr}"


def t_ratio(doc, rng):
    pairs = [("net income", "operating income"), ("income tax expense", "operating income"),
             ("operating expenses", "gross profit"), ("interest expense", "operating income"),
             ("cost of revenues", "total revenues")]
    pairs = [p for p in pairs if p[0] in doc.table_rows and p[1] in doc.table_rows]
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    i = rng.randrange(len(doc.years))
    prog = f"divide({doc.rows[a][i]}, {doc.rows[b][i]})"
    qs = [f"what was the ratio of {a} to {b} in {doc.years[i]}?", f"in {doc.years[i]} , {a} was what portion of {b}?"]
    return prog, qs, "ratio"


def t_share_of_total(doc, rng):
    j = rng.randrange(len(doc.segments))
    total = doc.rows["total revenues"][0]
    if "total revenues" not in doc.table_rows:
        return None
    prog = f"divide({doc.seg_rev[j]}, {total})"
    qs = [f"what percentage of total revenues in {doc.years[0]} came from the {doc.segments[j]} segment?",
          f"what portion of {doc.years[0]} total revenues did {doc.segments[j]} represent?"]
    return prog, qs, "share_of_total"


def t_weighted_margin(doc, rng):
    s, m = doc.seg_rev, doc.seg_margin
    prog = (f"multiply({s[0]}, {m[0]}%), multiply({s[1]}, {m[1]}%), multiply({s[2]}, {m[2]}%), "
            f"add(#0, #1), add(#3, #2), add({s[0]}, {s[1]}), add(#5, {s[2]}), divide(#4, #6)")
    qs = [f"what was the revenue-weighted average segment margin across all segments in {doc.years[0]}?",
          f"weighting each segment by its {doc.years[0]} revenue , what is the average segment margin?"]
    return prog, qs, "weighted_average"


def t_cagr(doc, rng):
    if len(doc.years) < 3:
        return None
    k = rng.choice([x for x in ("total revenues", "gross profit") if x in doc.table_rows] or [None])
    if not k:
        return None
    a, b = doc.rows[k][0], doc.rows[k][2]
    if a <= 0 or b <= 0:
        return None
    prog = f"divide({a}, {b}), exp(#0, 0.5), subtract(#1, const_1)"
    qs = [f"what was the compound annual growth rate of {k} from {doc.years[2]} to {doc.years[0]}?",
          f"what is the two-year cagr of {k} ending in {doc.years[0]}?"]
    return prog, qs, "cagr"


def t_per_share(doc, rng):
    if "net income" not in doc.table_rows:
        return None
    ni = doc.rows["net income"][0]
    if doc.scale == "millions":
        prog = f"divide({ni}, {doc.shares})"
        unit_note = ""
    else:  # thousands of dollars / millions of shares -> convert
        prog = f"divide({ni}, const_1000), divide(#0, {doc.shares})"
        unit_note = " ( net income is reported in thousands )"
    qs = [f"what was net income per diluted share in {doc.years[0]}{unit_note}?",
          f"based on weighted average diluted shares , what were {doc.years[0]} earnings per share?"]
    return prog, qs, "unit_conversion" if doc.scale == "thousands" else "per_share"


def t_compare_growth(doc, rng):
    if len(doc.years) < 2:
        return None
    ks = [k for k in doc.table_rows if all(v > 0 for v in doc.rows[k][:2])]
    if len(ks) < 2:
        return None
    a, b = rng.sample(ks, 2)
    R = doc.rows
    prog = (f"subtract({R[a][0]}, {R[a][1]}), divide(#0, {R[a][1]}), subtract({R[b][0]}, {R[b][1]}), "
            f"divide(#2, {R[b][1]}), greater(#1, #3)")
    qs = [f"did {a} grow faster than {b} from {doc.years[1]} to {doc.years[0]}?",
          f"was the growth rate of {a} higher than that of {b} between {doc.years[1]} and {doc.years[0]}?"]
    return prog, qs, "comparison"


def t_fcf_check(doc, rng):
    if "net cash provided by operating activities" not in doc.table_rows or "capital expenditures" not in doc.table_rows:
        return None
    R = doc.rows
    prog = f"add({R['net cash provided by operating activities'][0]}, {R['capital expenditures'][0]}), divide(#0, {R['total revenues'][0]})"
    if "total revenues" not in doc.table_rows:
        return None
    qs = [f"what was free cash flow as a percentage of total revenues in {doc.years[0]}?",
          f"free cash flow represented what share of {doc.years[0]} total revenues?"]
    return prog, qs, "multi_step_negative"


TEMPLATES = [t_growth, t_margin, t_margin_change, t_average, t_table_agg, t_ratio, t_share_of_total,
             t_weighted_margin, t_cagr, t_per_share, t_compare_growth, t_fcf_check]


def literals(prog: str) -> list[str]:
    out = []
    for st in dsl.parse_program(prog):
        if st.op in dsl.TABLE_OPS:
            continue
        for a in st.args:
            if not a.startswith("#") and not a.startswith("const_"):
                out.append(a)
    return out


def check(doc: Doc, ctx: dict, prog: str, value, kind: str) -> str | None:
    """Return a rejection reason or None if the item passes automatic checks."""
    text = render_document(ctx)
    text_nums = set()
    for t in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            text_nums.add(float(t.replace(",", "")))
        except ValueError:
            pass
    for a in literals(prog):
        v = dsl.str_to_num(a)
        if a.endswith("%"):
            v = float(a[:-1])
        if v is None:
            return "bad_literal"
        if v not in text_nums and -v not in text_nums and a not in ("0.5",):
            return "literal_not_in_document"
    rows = [r[0] for r in ctx["table"]]
    if len(rows) != len(set(rows)):
        return "ambiguous_row_names"
    for st in dsl.parse_program(prog):
        if st.op == "divide":
            d = st.args[1]
            dv = dsl.str_to_num(d) if not d.startswith("#") else None
            if dv is not None and abs(dv) < 1:
                return "near_zero_denominator"
    if isinstance(value, str):
        return None
    if abs(value) > 1e9:
        return "answer_too_large"
    steps = len(dsl.parse_program(prog))
    cells = {dsl.str_to_num(c.replace("$", "").split("(")[0].strip()) for r in ctx["table"][1:] for c in r[1:]}
    if steps >= 2 and round(value, 5) in {round(c, 5) for c in cells if c is not None}:
        return "trivial_answer_is_cell"
    if kind == "growth" and abs(value) > 5:
        return "implausible_growth"
    return None


def generate(n_docs: int, seed: int = 0, max_q_per_doc: int = 4):
    rng = random.Random(seed)
    items, rejects = [], Counter()
    for d in range(n_docs):
        doc = Doc(rng, d)
        ctx = doc.context()
        split = "train" if int(hashlib.md5(doc.doc_id.encode()).hexdigest(), 16) % 10 else "synth_dev"
        seen_q = set()
        tmpls = rng.sample(TEMPLATES, k=min(len(TEMPLATES), max_q_per_doc + 3))
        made = 0
        for t in tmpls:
            if made >= max_q_per_doc:
                break
            res = t(doc, rng)
            if res is None:
                rejects["template_not_applicable"] += 1
                continue
            prog, qs, kind = res
            r = dsl.execute(prog, ctx["table"])
            if not r.ok:
                rejects["exec_failed:" + (r.error or "")[:30]] += 1
                continue
            why = check(doc, ctx, prog, r.value, kind)
            if why:
                rejects[why] += 1
                continue
            q = rng.choice(qs)
            if q in seen_q:
                rejects["duplicate_question"] += 1
                continue
            seen_q.add(q)
            steps = len(r.steps)
            items.append({
                "id": f"synth:{doc.doc_id}:{made}",
                "source": "synth",
                "document_id": doc.doc_id,
                "group_id": doc.doc_id,
                "split": split,
                "context": ctx,
                "question": q,
                "answer": r.value,
                "answer_text": str(r.value),
                "unit": "yes_no" if isinstance(r.value, str) else "ratio_or_number",
                "scale": doc.scale,
                "program": prog,
                "provenance": {"dataset": "synthetic", "generator": "finqa.synth.generator", "seed": seed,
                               "template": t.__name__},
                "meta": {"program_verified": True, "kind": kind, "steps": steps,
                         "difficulty": "easy" if steps == 1 else ("medium" if steps <= 3 else "hard"),
                         "sector": doc.sector},
            })
            made += 1
    return items, rejects


def write_audit(items: list[dict], rejects: Counter, path: Path, n: int = 25, seed: int = 0):
    rng = random.Random(seed)
    sample = rng.sample(items, min(n, len(items)))
    kinds = Counter(i["meta"]["kind"] for i in items)
    diff = Counter(i["meta"]["difficulty"] for i in items)
    splits = Counter(i["split"] for i in items)
    lines = ["# Synthetic data audit (pilot)", "",
             "Automatically generated; automatic checks passed for every item below. Automatic checks do **not** "
             "guarantee semantic correctness. Each sample needs human review: tick the box if the question is "
             "unambiguous and the program answers it.", "",
             f"* items: {len(items)}; splits: {dict(splits)}", f"* kinds: {dict(kinds)}", f"* difficulty: {dict(diff)}",
             f"* rejections: {dict(rejects)}", "", "## Samples for human review", ""]
    for i, it in enumerate(sample):
        t = it["context"]["table"]
        lines += [f"### {i+1}. `{it['id']}` ({it['meta']['kind']}, {it['meta']['difficulty']})", "",
                  "Table:", "", "| " + " | ".join(t[0]) + " |", "|" + "---|" * len(t[0])]
        lines += ["| " + " | ".join(r) + " |" for r in t[1:]]
        lines += ["", "Text: " + " ".join(it["context"]["pre_text"][2:3] + it["context"]["post_text"]), "",
                  f"**Q:** {it['question']}", f"**Program:** `{it['program']}`", f"**Answer:** {it['answer']}", "",
                  "- [ ] reviewed: unambiguous and correct", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/synth/pilot.jsonl")
    ap.add_argument("--audit", default="docs/SYNTH_AUDIT.md")
    args = ap.parse_args()
    items, rejects = generate(args.n_docs, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    write_audit(items, rejects, Path(args.audit))
    stats = {"items": len(items), "rejects": dict(rejects), "kinds": dict(Counter(i["meta"]["kind"] for i in items)),
             "splits": dict(Counter(i["split"] for i in items))}
    Path(args.out + ".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
