import json
from collections import defaultdict
from pathlib import Path

import pytest

from finqa import dsl
from finqa.synth.generator import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synth():
    items, rejects = generate(60, seed=123)
    return items


def test_synth_programs_execute_to_gold(synth):
    assert len(synth) > 150
    for it in synth:
        r = dsl.execute(it["program"], it["context"]["table"])
        assert r.ok and r.value == it["answer"], it["id"]


def test_synth_document_level_split(synth):
    splits = defaultdict(set)
    for it in synth:
        splits[it["document_id"]].add(it["split"])
    assert all(len(s) == 1 for s in splits.values())


def test_synth_unique_questions_per_doc(synth):
    seen = set()
    for it in synth:
        k = (it["document_id"], it["question"])
        assert k not in seen
        seen.add(k)


def test_synth_deterministic():
    a, _ = generate(5, seed=7)
    b, _ = generate(5, seed=7)
    assert [x["question"] for x in a] == [x["question"] for x in b]


SEC = ROOT / "data/processed/sec_fresh_eval.jsonl"


@pytest.mark.skipif(not SEC.exists(), reason="run `make sec` first")
def test_sec_items_execute_and_are_fresh():
    rows = [json.loads(l) for l in open(SEC)]
    assert rows
    for r in rows:
        v = dsl.execute(r["program"], r["context"]["table"]).value
        assert v == r["answer"] if isinstance(v, str) else round(v, 5) == r["answer"]
        assert r["provenance"]["filed"] >= "2025-06-01"
