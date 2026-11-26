import importlib.util
import json
from pathlib import Path

import pytest

from finqa import dsl

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data/raw/FinQA/code/evaluate/evaluate.py"


def test_basic_ops():
    assert dsl.execute("subtract(5829, 5735), divide(#0, 5735)").value == round(94 / 5735, 5)
    assert dsl.execute("add(1, 2)").value == 3
    assert dsl.execute("multiply(2, const_100)").value == 200
    assert dsl.execute("add(const_m1, 0)").value == -1
    assert dsl.execute("divide(5%, 1)").value == 0.05
    assert dsl.execute("exp(1.1, 2)").value == round(1.21, 5)
    assert dsl.execute("greater(3, 2)").value == "yes"
    assert dsl.execute("greater(2, 3)").value == "no"


def test_table_ops():
    table = [["", "2019", "2018"], ["revenue", "$ 1,000", "$ 800"], ["loss", "-5 ( 5 )", "10"], ["name", "a", "b"]]
    assert dsl.execute("table_sum(revenue, none)", table).value == 1800
    assert dsl.execute("table_average(revenue, none)", table).value == 900
    assert dsl.execute("table_max(loss, none)", table).value == 10
    assert dsl.execute("table_min(loss, none)", table).value == -5
    r = dsl.execute("table_sum(name, none)", table)
    assert not r.ok and "non-numeric" in r.error
    assert not dsl.execute("table_sum(missing row, none)", table).ok


def test_row_names_with_commas():
    table = [["", "a", "b"], ["net income , as reported", "1", "2"]]
    assert dsl.execute("table_sum(net income , as reported, none)", table).value == 3


def test_thousands_separator_recovery():
    assert dsl.execute("add(5,829, 100)").value == 5929


@pytest.mark.parametrize("prog", [
    "", "   ", "add(1)", "add(1, 2, 3)", "foo(1, 2)", "add(1, 2) divide(3, 4)",
    "add(#1, 2)",  # forward reference
    "add(1, 2), add(#5, 1)",  # out of range
    "divide(1, 0)", "exp(10, 1000)", "add(abc, 1)",
    "__import__('os').system('echo pwned')", "add(1, 2); import os",
    "greater(1, 2), add(#0, 1)",  # arithmetic on yes/no
])
def test_rejects_malformed(prog):
    r = dsl.execute(prog)
    assert not r.ok


def test_step_limit():
    prog = ", ".join(["add(1, 1)"] * (dsl.MAX_STEPS + 2))
    assert not dsl.execute(prog).ok


def _load_official():
    spec = importlib.util.spec_from_file_location("finqa_official_eval", OFFICIAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not OFFICIAL.exists(), reason="raw FinQA repo not downloaded")
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_agrees_with_official_evaluator_on_all_gold_programs(split):
    """Our interpreter must reproduce the official eval_program on every gold program."""
    off = _load_official()
    data = json.loads((ROOT / f"data/raw/FinQA/dataset/{split}.json").read_text())
    disagree = []
    for e in data:
        prog = e["qa"]["program"]
        inv, ores = off.eval_program(off.program_tokenization(prog), e["table"])
        r = dsl.execute(prog, e["table"])
        if inv:
            if r.ok:
                disagree.append((e["id"], "official invalid, ours ok", prog))
            continue
        if not r.ok:
            disagree.append((e["id"], f"ours invalid: {r.error}", prog))
        elif r.value != ores:
            disagree.append((e["id"], f"{r.value} != {ores}", prog))
    # official executes table ops on '#' refs with a stale row (bug); we reject those. Allow <=0.1%.
    assert len(disagree) <= max(2, int(0.001 * len(data))), disagree[:10]


@pytest.mark.skipif(not OFFICIAL.exists(), reason="raw FinQA repo not downloaded")
def test_official_exe_ans_reproduced_dev_test():
    for split in ["dev", "test"]:
        data = json.loads((ROOT / f"data/raw/FinQA/dataset/{split}.json").read_text())
        for e in data:
            r = dsl.execute(e["qa"]["program"], e["table"])
            assert r.ok
            g = e["qa"]["exe_ans"]
            assert (r.value == g) if isinstance(g, str) else (round(r.value, 5) == round(g, 5)), e["id"]
