"""Answer extraction from model outputs and numeric answer matching.

Matching rules (documented in REPORT.md):

* ``primary`` -- percent-aware, sign-sensitive, relative tolerance 1%
  (absolute floor 1e-4 when the gold value is 0). If the prediction carries a
  ``%`` sign it is accepted when either p/100 or p matches the gold value
  (FinQA golds are inconsistently expressed as fractions or as percent
  numbers; "14.1%" denotes the same quantity in both conventions). A bare
  number p is compared to the gold value directly.
* ``lenient`` -- additionally accepts a bare number off by exactly x100 or
  /100 (percent/fraction confusion without a % sign). Reported separately.
* ``exact5`` -- official FinQA execution-accuracy style: round(p, 5) == gold.

Malformed or missing answers are failures (never silently dropped).
Scale words (thousand/million/billion) are ignored: answers are expected in the
document's units, as instructed in the prompt.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

REL_TOL = 0.01
ABS_TOL_ZERO = 1e-4

_ANSWER_LINE = re.compile(r"(?im)^\s*(?:\*\*)?\s*(?:final\s+)?answer\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$")
_PROGRAM_LINE = re.compile(r"(?im)^\s*(?:\*\*)?\s*(?:program|calculation)\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?\s*$")
_NUM = re.compile(r"(?<![\w.])[-−–]?\(?\$?\s*[-−–]?\d[\d,]*(?:\.\d+)?|(?<![\w.])[-−–]?\(?\$?\s*[-−–]?\.\d+")
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


@dataclass
class ParsedAnswer:
    raw: str | None
    value: float | str | None  # float, "yes"/"no", or None if unparseable
    has_percent: bool = False


def strip_thinking(text: str) -> str:
    if text is None:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text


def extract_program(text: str) -> str | None:
    text = strip_thinking(text)
    ms = _PROGRAM_LINE.findall(text)
    if not ms:
        return None
    prog = ms[-1].strip().strip("`").strip()
    return prog or None


def extract_answer_span(text: str) -> str | None:
    text = strip_thinking(text)
    ms = _ANSWER_LINE.findall(text)
    if ms:
        return ms[-1].strip().strip("`").strip()
    b = _BOXED.findall(text)
    if b:
        return b[-1].strip()
    return None


def parse_value(span: str | None) -> ParsedAnswer:
    if span is None:
        return ParsedAnswer(None, None)
    s = span.strip()
    low = s.lower().strip(" .*`")
    if re.match(r"^(yes|no)\b", low):
        return ParsedAnswer(span, low[:2] if low.startswith("no") else "yes")
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    m = _NUM.search(s)
    if not m:
        return ParsedAnswer(span, None)
    tok = m.group(0)
    neg = False
    t = tok.replace("$", "").replace(" ", "")
    if t.startswith("(") :
        # accounting negative: (123) -> -123 only if a closing paren follows
        rest = s[m.end():m.end() + 2]
        if ")" in rest:
            neg = True
        t = t.lstrip("(")
    if t.startswith("-"):
        neg = not neg if t.count("-") == 1 else neg
        t = t.lstrip("-")
    t = t.replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return ParsedAnswer(span, None)
    if neg:
        v = -v
    after = s[m.end():m.end() + 3]
    has_pct = "%" in after or "percent" in s[m.end():m.end() + 12].lower()
    return ParsedAnswer(span, v, has_pct)


def parse_answer(text: str) -> ParsedAnswer:
    return parse_value(extract_answer_span(text))


def _close(p: float, g: float, rel: float = REL_TOL) -> bool:
    if math.isnan(p) or math.isinf(p):
        return False
    if g == 0:
        return abs(p) <= ABS_TOL_ZERO
    return abs(p - g) <= rel * abs(g) + 1e-12


def match(pred: ParsedAnswer, gold: float | str, mode: str = "primary") -> bool:
    if pred is None or pred.value is None:
        return False
    if isinstance(gold, str):
        g = gold.strip().lower()
        return isinstance(pred.value, str) and pred.value == g
    if isinstance(pred.value, str):
        return False
    p = float(pred.value)
    g = float(gold)
    if mode == "exact5":
        cands = [p / 100.0, p] if pred.has_percent else [p]
        return any(round(c, 5) == round(g, 5) for c in cands)
    cands = [p / 100.0, p] if pred.has_percent else [p]
    if mode == "lenient":
        cands += [p * 100.0, p / 100.0]
    return any(_close(c, g) for c in cands)


def gold_value(ex: dict) -> float | str:
    """Gold numeric value for a canonical example."""
    a = ex["answer"]
    if isinstance(a, str):
        return a.strip().lower()
    return float(a)
