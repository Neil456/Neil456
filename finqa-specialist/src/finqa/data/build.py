"""Build canonical JSONL datasets from raw FinQA / TAT-QA / ConvFinQA / FinanceBench.

Canonical example schema (one JSON object per line):
  id            str   globally unique ("finqa:<orig id>")
  source        str   finqa | tatqa | convfinqa | financebench
  document_id   str   source document (FinQA filename, TAT-QA table uid, FB doc_name)
  group_id      str   coarser grouping (company/year) for grouped splits/bootstraps
  split         str   official split name (train/dev/test/...)
  context       dict  {"pre_text": [str], "table": [[str]], "post_text": [str]}
  question      str
  answer        float | "yes" | "no" | str   (gold value; FinQA uses exe_ans)
  answer_text   str   original answer string
  unit          str   percent | currency | ratio_or_number | yes_no | count | text
  scale         str   '' | thousand | million | billion | percent (TAT-QA)
  program       str | None  FinQA DSL program (verified executable) if available
  provenance    dict  dataset, repo, commit, file, file_sha256, original_id
  meta          dict  source-specific extras (gold_inds, answer_type, ...)

Usage:  python -m finqa.data.build --raw data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from finqa import dsl
from finqa.answers import parse_value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.$ ]", " ", q.lower())).strip()


def infer_unit(answer_text: str, value) -> str:
    if isinstance(value, str) and value in ("yes", "no"):
        return "yes_no"
    t = (answer_text or "").strip()
    if "%" in t:
        return "percent"
    if "$" in t:
        return "currency"
    return "ratio_or_number"


# ----------------------------------------------------------------- FinQA
def load_finqa(raw: Path, stats: dict) -> list[dict]:
    repo = raw / "FinQA"
    commit = git_commit(repo)
    out = []
    for split in ["train", "dev", "test"]:
        f = repo / "dataset" / f"{split}.json"
        fsha = sha256(f)
        data = json.loads(f.read_text())
        st = Counter()
        for e in data:
            qa = e["qa"]
            gold = qa["exe_ans"]
            if isinstance(gold, str):
                gold = gold.strip().lower()
            prog = qa.get("program")
            r = dsl.execute(prog, e["table"])
            prog_ok = r.ok and (
                (isinstance(gold, str) and r.value == gold)
                or (not isinstance(gold, str) and not isinstance(r.value, str) and round(float(r.value), 5) == round(float(gold), 5))
            )
            st["n"] += 1
            st["program_exec_ok"] += int(r.ok)
            st["program_matches_exe_ans"] += int(prog_ok)
            pv = parse_value(qa.get("answer", ""))
            from finqa.answers import match as amatch
            ans_consistent = pv.value is not None and amatch(pv, gold, "lenient")
            st["answer_text_consistent_with_exe_ans"] += int(bool(ans_consistent))
            st["answer_text_empty"] += int(not str(qa.get("answer", "")).strip())
            doc = e["filename"]
            company, year = (doc.split("/") + ["", ""])[:2]
            out.append({
                "id": f"finqa:{e['id']}",
                "source": "finqa",
                "document_id": doc,
                "group_id": f"{company}/{year}",
                "split": split,
                "context": {"pre_text": e["pre_text"], "table": e["table"], "post_text": e["post_text"]},
                "question": qa["question"],
                "answer": gold,
                "answer_text": str(qa.get("answer", "")),
                "unit": infer_unit(str(qa.get("answer", "")), gold),
                "scale": "",
                "program": prog if prog_ok else None,
                "provenance": {"dataset": "FinQA", "repo": "https://github.com/czyssrs/FinQA", "commit": commit,
                               "file": f"dataset/{split}.json", "file_sha256": fsha, "original_id": e["id"]},
                "meta": {"gold_inds": qa.get("gold_inds"), "program_verified": prog_ok,
                         "answer_text_consistent": bool(ans_consistent), "steps": len(r.steps) if r.ok else None},
            })
        stats[f"finqa/{split}"] = dict(st)
    return out


# ----------------------------------------------------------------- TAT-QA
def _tatqa_answer(q: dict):
    a = q["answer"]
    if isinstance(a, list):
        if len(a) == 1:
            a = a[0]
        else:
            return None, json.dumps(a)
    if isinstance(a, (int, float)):
        return float(a), str(a)
    pv = parse_value(str(a))
    if pv.value is not None and not isinstance(pv.value, str) and re.fullmatch(r"\s*[-$(]*[\d,.]+\)?\s*%?\s*", str(a)):
        return float(pv.value), str(a)
    return None, str(a)


def load_tatqa(raw: Path, stats: dict) -> list[dict]:
    repo = raw / "TAT-QA"
    commit = git_commit(repo)
    out = []
    for split, fname in [("train", "tatqa_dataset_train.json"), ("dev", "tatqa_dataset_dev.json"), ("test", "tatqa_dataset_test_gold.json")]:
        f = repo / "dataset_raw" / fname
        fsha = sha256(f)
        data = json.loads(f.read_text())
        st = Counter()
        for doc in data:
            table = doc["table"]["table"]
            paras = [p["text"] for p in sorted(doc["paragraphs"], key=lambda p: p["order"])]
            for q in doc["questions"]:
                st["n"] += 1
                val, atext = _tatqa_answer(q)
                numeric = q["answer_type"] in ("arithmetic", "count") or val is not None
                if val is None or not numeric:
                    st["skipped_non_numeric"] += 1
                    continue
                st["kept"] += 1
                scale = q.get("scale", "")
                unit = "percent" if scale == "percent" else ("count" if q["answer_type"] == "count" else "ratio_or_number")
                out.append({
                    "id": f"tatqa:{q['uid']}",
                    "source": "tatqa",
                    "document_id": doc["table"]["uid"],
                    "group_id": doc["table"]["uid"],
                    "split": split,
                    "context": {"pre_text": paras, "table": table, "post_text": []},
                    "question": q["question"],
                    "answer": val,
                    "answer_text": atext,
                    "unit": unit,
                    "scale": scale,
                    "program": None,
                    "provenance": {"dataset": "TAT-QA", "repo": "https://github.com/NExTplusplus/TAT-QA", "commit": commit,
                                   "file": f"dataset_raw/{fname}", "file_sha256": fsha, "original_id": q["uid"]},
                    "meta": {"answer_type": q["answer_type"], "answer_from": q.get("answer_from"), "derivation": q.get("derivation")},
                })
        stats[f"tatqa/{split}"] = dict(st)
    return out


# ----------------------------------------------------------------- ConvFinQA (leakage analysis + optional secondary)
def load_convfinqa(raw: Path, stats: dict) -> list[dict]:
    repo = raw / "ConvFinQA"
    commit = git_commit(repo)
    zsha = sha256(repo / "data.zip")
    base = raw / "ConvFinQA_unzipped" / "data"
    out = []
    for split, fname in [("train", "train_turn.json"), ("dev", "dev_turn.json")]:
        data = json.loads((base / fname).read_text())
        st = Counter()
        for e in data:
            ann = e["annotation"]
            st["n"] += 1
            gold = ann.get("exe_ans")
            if gold is None:
                continue
            if isinstance(gold, str):
                gold = gold.strip().lower()
            hist = ann.get("cur_dial", [])
            q = hist[-1] if hist else ann.get("cur_question", "")
            doc = e["filename"]
            company, year = (doc.split("/") + ["", ""])[:2]
            out.append({
                "id": f"convfinqa:{e['id']}",
                "source": "convfinqa",
                "document_id": doc,
                "group_id": f"{company}/{year}",
                "split": split,
                "context": {"pre_text": e["pre_text"], "table": e["table"], "post_text": e["post_text"]},
                "question": q,
                "answer": gold,
                "answer_text": str(gold),
                "unit": "ratio_or_number" if not isinstance(gold, str) else "yes_no",
                "scale": "",
                "program": ann.get("cur_program"),
                "provenance": {"dataset": "ConvFinQA", "repo": "https://github.com/czyssrs/ConvFinQA", "commit": commit,
                               "file": f"data.zip:data/{fname}", "file_sha256": zsha, "original_id": e["id"]},
                "meta": {"dialogue_history": hist[:-1], "turn_ind": ann.get("turn_ind")},
            })
        stats[f"convfinqa/{split}"] = dict(st)
    return out


# ----------------------------------------------------------------- FinanceBench (external eval only)
def load_financebench(raw: Path, stats: dict) -> list[dict]:
    repo = raw / "financebench"
    commit = git_commit(repo)
    f = repo / "data" / "financebench_open_source.jsonl"
    fsha = sha256(f)
    out = []
    st = Counter()
    for line in f.read_text().splitlines():
        r = json.loads(line)
        st["n"] += 1
        atext = r["answer"].strip()
        # numeric only if the whole answer is a single number (optionally $ / % / units word)
        m = re.fullmatch(r"\s*(?:approximately\s*)?\(?-?\$?\s*-?[\d,]*\.?\d+\)?\s*(%|percent|million|billion|thousand)?\s*\.?\s*", atext, flags=re.I)
        pv = parse_value(atext)
        numeric = bool(m) and pv.value is not None and not isinstance(pv.value, str)
        st["numeric"] += int(numeric)
        ev = "\n\n".join(x.get("evidence_text", "") for x in r.get("evidence", []))
        out.append({
            "id": f"financebench:{r['financebench_id']}",
            "source": "financebench",
            "document_id": r["doc_name"],
            "group_id": r["company"],
            "split": "eval",
            "context": {"pre_text": [ev], "table": [], "post_text": []},
            "question": r["question"],
            "answer": (float(pv.value) if numeric else atext),
            "answer_text": atext,
            "unit": ("percent" if numeric and pv.has_percent else ("currency" if numeric and "$" in atext else ("ratio_or_number" if numeric else "text"))),
            "scale": "",
            "program": None,
            "provenance": {"dataset": "FinanceBench (open source subset)", "repo": "https://github.com/patronus-ai/financebench",
                           "commit": commit, "file": "data/financebench_open_source.jsonl", "file_sha256": fsha,
                           "original_id": r["financebench_id"]},
            "meta": {"question_type": r.get("question_type"), "question_reasoning": r.get("question_reasoning"),
                     "numeric_scorable": numeric, "company": r["company"]},
        })
    stats["financebench/eval"] = dict(st)
    return out


# ----------------------------------------------------------------- dedup / leakage
def dedup_and_leakage(rows: list[dict], stats: dict) -> list[dict]:
    """Drop exact & near duplicates inside train; drop train items that overlap eval splits.

    Official dev/test splits are never modified. Keys:
      exact: (normalized question, document_id)
      near : same document_id and SequenceMatcher ratio >= 0.9 on normalized question
    Cross-source: ConvFinQA / TAT-QA train items whose document appears in FinQA dev/test are
    flagged ``leak_doc`` and excluded from any training mix.
    """
    eval_keys = set()
    eval_q_by_doc = defaultdict(list)
    eval_docs = set()
    eval_tables = set()
    for r in rows:
        if r["split"] != "train":
            k = (norm_q(r["question"]), r["document_id"])
            eval_keys.add(k)
            eval_q_by_doc[r["document_id"]].append(norm_q(r["question"]))
            if r["source"] in ("finqa", "convfinqa"):
                eval_docs.add(r["document_id"])
            eval_tables.add(hashlib.md5(json.dumps(r["context"]["table"]).encode()).hexdigest())
    seen = {}
    kept = []
    st = Counter()
    for r in rows:
        if r["split"] != "train":
            kept.append(r)
            continue
        nq = norm_q(r["question"])
        k = (nq, r["document_id"])
        r["meta"]["leak_doc"] = r["document_id"] in eval_docs
        r["meta"]["leak_table"] = hashlib.md5(json.dumps(r["context"]["table"]).encode()).hexdigest() in eval_tables and bool(r["context"]["table"])
        if k in eval_keys:
            st[f"{r['source']}:train_exact_overlap_with_eval_dropped"] += 1
            continue
        if any(SequenceMatcher(None, nq, q2).ratio() >= 0.9 for q2 in eval_q_by_doc.get(r["document_id"], [])):
            st[f"{r['source']}:train_near_overlap_with_eval_dropped"] += 1
            continue
        sk = (r["source"],) + k
        if sk in seen:
            st[f"{r['source']}:train_exact_duplicate_dropped"] += 1
            continue
        seen[sk] = r["id"]
        st[f"{r['source']}:train_doc_shared_with_eval_flagged"] += int(r["meta"]["leak_doc"])
        st[f"{r['source']}:train_table_shared_with_eval_flagged"] += int(r["meta"]["leak_table"])
        kept.append(r)
    stats["dedup"] = dict(st)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--manifest", default="results/data_manifest.json")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    rows = load_finqa(raw, stats) + load_tatqa(raw, stats) + load_convfinqa(raw, stats)
    rows = dedup_and_leakage(rows, stats)
    rows += load_financebench(raw, stats)
    ids = Counter(r["id"] for r in rows)
    dup_ids = [i for i, c in ids.items() if c > 1]
    assert not dup_ids, f"duplicate ids: {dup_ids[:5]}"
    files = {}
    by = defaultdict(list)
    for r in rows:
        by[(r["source"], r["split"])].append(r)
    for (src, split), rs in sorted(by.items()):
        p = out / f"{src}_{split}.jsonl"
        with open(p, "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[p.name] = {"n": len(rs), "sha256": sha256(p)}
    raw_files = {}
    for rel in ["FinQA/dataset/train.json", "FinQA/dataset/dev.json", "FinQA/dataset/test.json",
                "TAT-QA/dataset_raw/tatqa_dataset_train.json", "TAT-QA/dataset_raw/tatqa_dataset_dev.json",
                "TAT-QA/dataset_raw/tatqa_dataset_test_gold.json", "ConvFinQA/data.zip",
                "financebench/data/financebench_open_source.jsonl"]:
        raw_files[rel] = sha256(raw / rel)
    commits = {d: git_commit(raw / d) for d in ["FinQA", "TAT-QA", "ConvFinQA", "financebench"]}
    manifest = {"raw_commits": commits, "raw_sha256": raw_files, "processed": files, "stats": stats}
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["stats"], indent=1))
    print(json.dumps(files, indent=1))


if __name__ == "__main__":
    main()
