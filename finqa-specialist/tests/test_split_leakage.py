"""Split-leakage guards on the processed canonical data (skipped if data not built)."""
import json
from pathlib import Path

import pytest

from finqa.data.build import norm_q

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
pytestmark = pytest.mark.skipif(not (P / "finqa_train.jsonl").exists(), reason="run `make data` first")


def load(name):
    return [json.loads(l) for l in open(P / f"{name}.jsonl")]


@pytest.fixture(scope="module")
def finqa():
    return {s: load(f"finqa_{s}") for s in ["train", "dev", "test"]}


def test_unique_ids(finqa):
    ids = [r["id"] for s in finqa.values() for r in s]
    assert len(ids) == len(set(ids))


def test_split_labels(finqa):
    for s, rows in finqa.items():
        assert all(r["split"] == s for r in rows)


def test_document_level_disjoint(finqa):
    tr = {r["document_id"] for r in finqa["train"]}
    for s in ["dev", "test"]:
        assert not tr & {r["document_id"] for r in finqa[s]}


def test_no_question_doc_overlap(finqa):
    tr = {(norm_q(r["question"]), r["document_id"]) for r in finqa["train"]}
    for s in ["dev", "test"]:
        assert not tr & {(norm_q(r["question"]), r["document_id"]) for r in finqa[s]}


def test_train_has_no_exact_duplicates(finqa):
    keys = [(norm_q(r["question"]), r["document_id"]) for r in finqa["train"]]
    assert len(keys) == len(set(keys))


def test_convfinqa_train_not_in_finqa_eval(finqa):
    conv = load("convfinqa_train")
    ev = {r["document_id"] for s in ["dev", "test"] for r in finqa[s]}
    assert not {r["document_id"] for r in conv} & ev


def test_eval_subsets_are_dev_only(finqa):
    dev = {r["id"] for r in finqa["dev"]}
    test = {r["id"] for r in finqa["test"]}
    for f in (ROOT / "results/subsets").glob("finqa_dev_*.txt"):
        ids = {l.strip() for l in open(f) if l.strip()}
        assert ids <= dev and not ids & test, f


def test_sft_train_ids_disjoint_from_eval():
    for f in (ROOT / "checkpoints").glob("*/train_ids.txt"):
        ids = {l.strip() for l in open(f) if l.strip()}
        assert all(i.startswith("finqa:") or i.startswith("synth:") for i in ids)
        dev = {r["id"] for r in load("finqa_dev")} | {r["id"] for r in load("finqa_test")}
        assert not ids & dev, f


def test_programs_verified(finqa):
    from finqa import dsl
    for r in finqa["dev"][:200]:
        if r["program"]:
            v = dsl.execute(r["program"], r["context"]["table"]).value
            g = r["answer"]
            assert (v == g) if isinstance(g, str) else round(v, 5) == round(g, 5)
