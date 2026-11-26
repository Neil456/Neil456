"""Per-example scoring of a model output against a canonical example."""
from __future__ import annotations

from finqa import dsl
from finqa.answers import ParsedAnswer, extract_program, gold_value, match, parse_answer


def score_output(ex: dict, text: str) -> dict:
    gold = gold_value(ex)
    pa = parse_answer(text)
    prog = extract_program(text)
    pe = dsl.execute(prog, ex["context"].get("table", [])) if prog else None
    pe_ans = None
    if pe is not None and pe.ok:
        v = pe.value
        pe_ans = ParsedAnswer(str(v), v, False)
    numeric_gold = not isinstance(gold, str) or gold in ("yes", "no")
    rec = {
        "gold": gold,
        "pred_answer_raw": pa.raw,
        "pred_value": pa.value,
        "pred_has_percent": pa.has_percent,
        "answer_parsed": pa.value is not None,
        "program": prog,
        "program_ok": bool(pe and pe.ok),
        "program_error": (pe.error if pe is not None and not pe.ok else (None if pe else "no program")),
        "program_value": (pe.value if pe is not None and pe.ok else None),
        "scorable": numeric_gold,
    }
    if numeric_gold:
        rec.update({
            # raw model answer (model did its own arithmetic)
            "correct": match(pa, gold, "primary"),
            "correct_lenient": match(pa, gold, "lenient"),
            "correct_exact5": match(pa, gold, "exact5"),
            # program executed by restricted interpreter (tool-assisted / harness score)
            "prog_correct": bool(pe_ans and match(pe_ans, gold, "primary")),
            "prog_correct_exact5": bool(pe_ans and match(pe_ans, gold, "exact5")),
        })
        # assisted fallback: program result when it parses, else raw answer
        rec["assisted_correct"] = rec["prog_correct"] if rec["program_ok"] else rec["correct"]
    return rec
