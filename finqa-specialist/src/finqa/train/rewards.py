"""Verifiable reward functions for GRPO / RLVR on FinQA-style tasks.

Design goals (Phase 4 of the brief):
* reward correct normalized numeric answers and valid grounded calculations;
* never reward formatting alone (a well-formed but wrong answer earns 0);
* robust to adversarial outputs (multiple Answer lines, hedged lists of numbers,
  huge outputs, prompt-injection text, unit/scale/sign mistakes).

Reward components (all in [0, 1]; combined reward = correctness + 0.2 * grounding, max 1.2):
* correctness: 1 if the *program* executes (restricted interpreter) to the gold value under the
  primary matcher AND the stated Answer agrees with the gold value; 0.5 if exactly one of the
  two is correct; 0 otherwise. Requiring agreement discourages "answer-only" guessing while still
  giving partial credit, and executing the program means arithmetic is verified, not trusted.
* grounding: fraction of numeric literals in the program that appear in the document (or are
  small constants / percent forms of document numbers). Only granted when correctness > 0, so it
  can never be farmed with a wrong answer.
Hedging guard: if the Answer line contains more than one number, correctness for the stated
answer is 0 (prevents "Answer: 0.1 or 10 or 1000"). Outputs longer than MAX_CHARS get 0.
"""
from __future__ import annotations

import re

from finqa import dsl
from finqa.answers import ParsedAnswer, extract_answer_span, extract_program, match, parse_value, strip_thinking
from finqa.prompts import render_document

MAX_CHARS = 4000
SMALL_CONSTS = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 100.0, 1000.0, 1000000.0, -1.0}
_NUMTOK = re.compile(r"-?\d[\d,]*\.?\d*")


def _doc_numbers(ctx: dict) -> set[float]:
    text = render_document(ctx)
    nums = set()
    for t in _NUMTOK.findall(text):
        try:
            v = float(t.replace(",", ""))
        except ValueError:
            continue
        nums.add(round(v, 6))
        nums.add(round(-v, 6))
    return nums


def grounding_score(program_steps: list[dsl.Step], ctx: dict) -> float:
    doc = _doc_numbers(ctx)
    lits = []
    for st in program_steps:
        if st.op in dsl.TABLE_OPS:
            continue
        for a in st.args:
            a = a.strip()
            if a.startswith("#"):
                continue
            v = dsl.str_to_num(a.replace("$", ""))
            if v is None:
                continue
            lits.append((a, v))
    if not lits:
        # pure table-op programs are grounded by construction (row must exist to execute)
        return 1.0 if program_steps else 0.0
    ok = 0
    for a, v in lits:
        if a.startswith("const_") or v in SMALL_CONSTS:
            ok += 1
        elif round(v, 6) in doc or ("%" in a and round(v * 100, 6) in doc):
            ok += 1
    return ok / len(lits)


def score_completion(text: str, ex: dict) -> dict:
    gold = ex["answer"]
    if isinstance(gold, str):
        gold = gold.strip().lower()
    out = {"reward": 0.0, "correctness": 0.0, "grounding": 0.0, "answer_ok": False, "program_ok": False,
           "reason": ""}
    if text is None or len(text) > MAX_CHARS:
        out["reason"] = "empty_or_too_long"
        return out
    body = strip_thinking(text)
    span = extract_answer_span(body)
    pa = parse_value(span)
    hedged = span is not None and len(_NUMTOK.findall(span)) > 1
    answer_ok = (not hedged) and match(pa, gold, "primary")
    prog = extract_program(body)
    prog_ok, steps = False, []
    if prog:
        r = dsl.execute(prog, ex["context"].get("table", []))
        if r.ok:
            steps = r.steps
            prog_ok = match(ParsedAnswer(str(r.value), r.value, False), gold, "primary")
    corr = 1.0 if (answer_ok and prog_ok) else (0.5 if (answer_ok or prog_ok) else 0.0)
    g = grounding_score(steps, ex["context"]) if (corr > 0 and steps) else 0.0
    out.update({"correctness": corr, "grounding": g, "answer_ok": answer_ok, "program_ok": prog_ok,
                "reward": corr + 0.2 * g, "reason": "hedged" if hedged else ""})
    return out


def make_trl_reward(examples_by_prompt_id: dict):
    """Return a TRL-compatible reward function: f(prompts, completions, **kwargs) -> list[float].

    The training dataset must carry an ``example_id`` column; TRL forwards extra dataset
    columns as keyword arguments.
    """
    def finqa_reward(prompts=None, completions=None, example_id=None, **kwargs):
        rewards = []
        for comp, eid in zip(completions, example_id):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            rewards.append(score_completion(text, examples_by_prompt_id[eid])["reward"])
        return rewards
    finqa_reward.__name__ = "finqa_verifiable_reward"
    return finqa_reward
