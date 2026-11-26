"""Runner guards, caching and resume using a fake adapter (no network, no model download)."""
import json
from pathlib import Path

import pytest

from finqa.eval import runner
from finqa.eval.adapters import Adapter, GenResult


def _write_data(tmp_path, split="dev", n=5):
    rows = []
    for i in range(n):
        rows.append({"id": f"t:{i}", "source": "finqa", "split": split, "group_id": "g", "document_id": "d",
                     "context": {"pre_text": ["revenue was 10 and 12"], "table": [["", "a"], ["rev", "10"]], "post_text": []},
                     "question": f"q{i}", "answer": 2.0, "answer_text": "2", "unit": "ratio_or_number", "scale": "",
                     "program": "subtract(12, 10)", "provenance": {}, "meta": {}})
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


class Fake(Adapter):
    calls = 0

    def describe(self):
        return {"adapter": "fake"}

    def generate(self, batch, cfg):
        Fake.calls += len(batch)
        return [GenResult("Program: subtract(12, 10)\nAnswer: 2", 100, 10, 0.01, "stop") for _ in batch]


@pytest.fixture
def fake(monkeypatch):
    Fake.calls = 0
    monkeypatch.setattr(runner, "make_adapter", lambda spec: Fake())
    return Fake


def test_refuses_test_split(tmp_path, fake):
    d = _write_data(tmp_path, "test")
    with pytest.raises(SystemExit):
        runner.main(["--adapter", "hf", "--model", "m", "--data", str(d), "--out", str(tmp_path / "o.jsonl")])


@pytest.mark.parametrize("adapter", ["anthropic", "gemini"])
def test_paid_requires_budget(tmp_path, fake, adapter):
    d = _write_data(tmp_path)
    with pytest.raises(SystemExit):
        runner.main(["--adapter", adapter, "--model", "m", "--data", str(d), "--out", str(tmp_path / "o.jsonl")])
    with pytest.raises(SystemExit):
        runner.main(["--adapter", adapter, "--model", "m", "--data", str(d), "--out", str(tmp_path / "o.jsonl"),
                     "--allow-paid", "--price-in", "1", "--price-out", "1"])  # no --max-usd


def test_openai_remote_requires_budget(tmp_path, fake):
    d = _write_data(tmp_path)
    with pytest.raises(SystemExit):
        runner.main(["--adapter", "openai_compat", "--model", "m", "--base-url", "https://api.openai.com/v1",
                     "--data", str(d), "--out", str(tmp_path / "o.jsonl")])


def test_local_openai_compat_allowed_without_budget(tmp_path, fake):
    d = _write_data(tmp_path)
    runner.main(["--adapter", "openai_compat", "--model", "m", "--base-url", "http://127.0.0.1:8000/v1",
                 "--data", str(d), "--out", str(tmp_path / "o.jsonl"), "--no-cache"])
    assert fake.calls == 5


def test_scoring_resume_and_cache(tmp_path, fake):
    d = _write_data(tmp_path)
    out = tmp_path / "o.jsonl"
    cache = tmp_path / "c.sqlite"
    args = ["--adapter", "hf", "--model", "m", "--data", str(d), "--out", str(out), "--cache", str(cache)]
    runner.main(args)
    assert fake.calls == 5
    recs = [json.loads(l) for l in open(out)]
    assert all(r["correct"] and r["prog_correct"] for r in recs)
    runner.main(args)  # resume: nothing to do
    assert fake.calls == 5
    assert len(open(out).readlines()) == 5
    runner.main(args[:-4] + ["--out", str(tmp_path / "o2.jsonl"), "--cache", str(cache)])  # cache hit
    assert fake.calls == 5
    summ = json.loads(Path(str(out) + ".summary.json").read_text())
    assert summ["correct"]["mean"] == 1.0


def test_anthropic_adapter_ignores_generic_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://evil.example")
    from finqa.eval.adapters import AnthropicAdapter
    a = AnthropicAdapter("claude-x")
    assert a.base_url == "https://api.anthropic.com"


def test_stacked_lora_paths_are_split(monkeypatch):
    """--lora a,b must merge both adapters in order (SFT then GRPO)."""
    import types
    calls = []

    class FakePeft:
        @staticmethod
        def from_pretrained(model, path):
            calls.append(path)
            return types.SimpleNamespace(merge_and_unload=lambda: model)

    import sys
    fake_peft = types.ModuleType("peft"); fake_peft.PeftModel = FakePeft
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    from finqa.eval import adapters
    src = open(adapters.__file__).read()
    assert 'adapter_path.split(",")' in src
    model = object()
    for path in "ckpt/sft, ckpt/grpo".split(","):
        model = FakePeft.from_pretrained(model, path.strip()).merge_and_unload()
    assert calls == ["ckpt/sft", "ckpt/grpo"]
