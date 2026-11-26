"""Restricted interpreter for the FinQA program DSL.

Semantics mirror the official evaluator (czyssrs/FinQA code/evaluate/evaluate.py,
commit 0f16e28): ops add/subtract/multiply/divide/exp/greater and
table_max/min/sum/average; numeric literals parsed like ``str_to_num``
(commas removed, ``x%`` -> x/100, ``const_k`` -> k, ``const_m1`` -> -1);
``#k`` references the result of step k; the final numeric result is rounded
to 5 decimals; ``greater`` yields "yes"/"no".

No ``eval``/``exec`` is used. Programs are parsed with a small regex grammar
and executed with explicit guards (step limit, magnitude limit, zero division,
exponent bounds). Document text is treated as untrusted data: table row names
are only used as dictionary keys.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Union

ARITH_OPS = ("add", "subtract", "multiply", "divide", "exp", "greater")
TABLE_OPS = ("table_max", "table_min", "table_sum", "table_average")
ALL_OPS = ARITH_OPS + TABLE_OPS

MAX_STEPS = 12
MAX_ABS = 1e15
MAX_EXP = 64

Value = Union[float, str]


class DSLError(ValueError):
    pass


@dataclass
class Step:
    op: str
    args: list[str]


@dataclass
class ExecResult:
    ok: bool
    value: Value | None = None
    error: str | None = None
    steps: list[Step] = field(default_factory=list)
    trace: list[Value] = field(default_factory=list)


def str_to_num(text: str) -> float | None:
    """Port of the official ``str_to_num`` (returns None instead of "n/a")."""
    text = text.replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        pass
    if "%" in text:
        try:
            return float(text.replace("%", "")) / 100.0
        except ValueError:
            return None
    if text.startswith("const_"):
        t = text[len("const_"):]
        if t == "m1":
            t = "-1"
        try:
            return float(t)
        except ValueError:
            return None
    return None


def process_row(row: list[str]) -> list[float] | None:
    """Port of the official ``process_row``: '$' stripped, '(...)' suffix dropped."""
    out = []
    for cell in row:
        c = cell.replace("$", "").strip()
        c = c.split("(")[0].strip()
        v = str_to_num(c)
        if v is None:
            return None
        out.append(v)
    return out


_STEP_RE = re.compile(r"\s*([a-z_]+)\s*\(([^()]*)\)\s*")


def _split_args(argstr: str, op: str) -> list[str]:
    if op in TABLE_OPS:
        # Row names may themselves contain commas; split at the LAST comma (2nd arg is 'none').
        head, sep, tail = argstr.rpartition(",")
        if sep and tail.strip().lower() == "none":
            return [head.strip(), "none"]
        return [argstr.strip(), "none"]
    parts = [a.strip() for a in argstr.split(",")]
    if len(parts) == 2:
        return parts
    # Model outputs sometimes use thousands separators: add(5,829, 100).
    # Merge "<1-3 digits>" + "<3 digits>" pairs (left to right) only while more than 2 args remain.
    merged = list(parts)
    while len(merged) > 2:
        for i in range(len(merged) - 1):
            if re.fullmatch(r"-?\d{1,3}(,\d{3})*", merged[i]) and re.fullmatch(r"\d{3}(\.\d+)?%?", merged[i + 1]):
                merged[i:i + 2] = [merged[i] + "," + merged[i + 1]]
                break
        else:
            break
    if len(merged) != 2:
        raise DSLError(f"{op} expects 2 args, got {len(merged)}: {argstr!r}")
    return merged


def parse_program(program: str) -> list[Step]:
    """Parse 'op(a, b), op(#0, c)' into steps. Raises DSLError on malformed input."""
    if program is None:
        raise DSLError("empty program")
    s = program.strip().rstrip(".").strip()
    if s.endswith("EOF"):
        s = s[:-3].rstrip(", ")
    if not s:
        raise DSLError("empty program")
    steps: list[Step] = []
    pos = 0
    while pos < len(s):
        m = _STEP_RE.match(s, pos)
        if not m:
            raise DSLError(f"cannot parse near: {s[pos:pos+40]!r}")
        op = m.group(1)
        if op not in ALL_OPS:
            raise DSLError(f"unknown op {op!r}")
        steps.append(Step(op, _split_args(m.group(2), op)))
        pos = m.end()
        if pos < len(s):
            if s[pos] != ",":
                raise DSLError(f"expected ',' between steps near {s[pos:pos+20]!r}")
            pos += 1
        if len(steps) > MAX_STEPS:
            raise DSLError("too many steps")
    return steps


def _norm_row(name: str) -> str:
    """Whitespace/case-insensitive row key (lenient fallback, used only if exact match fails)."""
    n = re.sub(r"\s+", " ", name.strip().lower())
    return re.sub(r"\s+([,.;:)%])", r"\1", n).replace("( ", "(")


def _guard(x: float) -> float:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x) or abs(x) > MAX_ABS):
        raise DSLError("numeric overflow / nan")
    return x


def execute(program: str | list[Step], table: list[list[str]] | None = None, round_digits: int | None = 5) -> ExecResult:
    """Execute a DSL program against an optional table. Never raises."""
    try:
        steps = parse_program(program) if isinstance(program, str) else program
    except DSLError as e:
        return ExecResult(False, error=f"parse: {e}")
    table = table or []
    table_dict = {}
    norm_dict = {}
    for row in table:
        if row:
            table_dict.setdefault(row[0], row[1:])
            norm_dict.setdefault(_norm_row(row[0]), row[1:])
    res: list[Value] = []
    try:
        for i, st in enumerate(steps):
            a1, a2 = st.args

            def val(a: str) -> Value:
                a = a.strip()
                if a.startswith("#"):
                    try:
                        k = int(a[1:])
                    except ValueError:
                        raise DSLError(f"bad ref {a!r}")
                    if k < 0 or k >= i:
                        raise DSLError(f"forward/invalid ref {a!r} at step {i}")
                    return res[k]
                v = str_to_num(a.replace("$", ""))
                if v is None:
                    raise DSLError(f"non-numeric arg {a!r}")
                return v

            if st.op in ARITH_OPS:
                x, y = val(a1), val(a2)
                if isinstance(x, str) or isinstance(y, str):
                    raise DSLError("arithmetic on yes/no")
                if st.op == "add":
                    r: Value = x + y
                elif st.op == "subtract":
                    r = x - y
                elif st.op == "multiply":
                    r = x * y
                elif st.op == "divide":
                    if y == 0:
                        raise DSLError("division by zero")
                    r = x / y
                elif st.op == "exp":
                    if abs(y) > MAX_EXP:
                        raise DSLError("exponent too large")
                    try:
                        r = float(x) ** float(y)
                    except (OverflowError, ZeroDivisionError):
                        raise DSLError("exp overflow")
                    if isinstance(r, complex):
                        raise DSLError("complex result")
                else:  # greater
                    r = "yes" if x > y else "no"
                if not isinstance(r, str):
                    r = _guard(float(r))
            else:
                if a1.startswith("#"):
                    raise DSLError("table op on reference not supported")
                if a1 in table_dict:  # official semantics: exact row-name match
                    cells = table_dict[a1]
                elif _norm_row(a1) in norm_dict:  # lenient fallback for model-written programs
                    cells = norm_dict[_norm_row(a1)]
                else:
                    raise DSLError(f"row not found: {a1[:60]!r}")
                row = process_row(cells)
                if not row:
                    raise DSLError("non-numeric or empty table row")
                if st.op == "table_max":
                    r = max(row)
                elif st.op == "table_min":
                    r = min(row)
                elif st.op == "table_sum":
                    r = sum(row)
                else:
                    r = sum(row) / len(row)
                r = _guard(float(r))
            res.append(r)
    except DSLError as e:
        return ExecResult(False, error=str(e), steps=steps, trace=res)
    if not res:
        return ExecResult(False, error="no steps", steps=steps)
    final = res[-1]
    if not isinstance(final, str) and round_digits is not None:
        final = round(final, round_digits)
    return ExecResult(True, value=final, steps=steps, trace=res)


def format_program(steps: list[Step]) -> str:
    return ", ".join(f"{s.op}({s.args[0]}, {s.args[1]})" for s in steps)
