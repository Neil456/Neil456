import pytest

from finqa.answers import ParsedAnswer, extract_answer_span, extract_program, match, parse_answer, parse_value


@pytest.mark.parametrize("text,value,pct", [
    ("Program: x\nAnswer: 1.64%", 1.64, True),
    ("Answer: 0.0164", 0.0164, False),
    ("Answer: $1,234.5 million", 1234.5, False),
    ("Answer: -5.2%", -5.2, True),
    ("Answer: (123)", -123.0, False),
    ("**Answer:** 3", 3.0, False),
    ("**Answer: 3.5**", 3.5, False),
    ("The result is \\boxed{42}", 42.0, False),
    ("Final answer: 12 percent", 12.0, True),
    ("Answer: −7.5", -7.5, False),  # unicode minus
    ("Answer: .5", 0.5, False),
    ("<think>Answer: 99</think>\nAnswer: 7", 7.0, False),
    ("Answer: 1\nAnswer: 2", 2.0, False),  # last answer wins
])
def test_parse(text, value, pct):
    p = parse_answer(text)
    assert p.value == pytest.approx(value)
    assert p.has_percent == pct


@pytest.mark.parametrize("text", ["", "I don't know", "Answer:", "Answer: n/a", "Program: add(1, 2)", None])
def test_unparseable_is_failure(text):
    p = parse_answer(text)
    assert p.value is None
    assert not match(p, 3.0)


def test_yes_no():
    assert parse_answer("Answer: Yes.").value == "yes"
    assert parse_answer("Answer: no").value == "no"
    assert match(parse_answer("Answer: yes"), "yes")
    assert not match(parse_answer("Answer: yes"), "no")
    assert not match(parse_answer("Answer: 1"), "yes")


def P(v, pct=False):
    return ParsedAnswer(str(v), v, pct)


def test_match_percent_conventions():
    # gold as fraction
    assert match(P(1.64, True), 0.01639)
    assert match(P(0.0164), 0.01639)
    assert not match(P(1.64), 0.01639)          # bare percent number -> primary fails
    assert match(P(1.64), 0.01639, "lenient")   # ... lenient accepts x100 confusion
    # gold as percent number
    assert match(P(14.1, True), 14.1)
    assert match(P(14.1), 14.1)


def test_match_tolerance_and_sign():
    assert match(P(100.9), 100.0)
    assert not match(P(101.5), 100.0)
    assert not match(P(5.0, True), -0.05)       # sign errors are wrong
    assert match(P(-5.0, True), -0.05)
    assert match(P(0.0), 0.0)
    assert not match(P(0.01), 0.0)


def test_exact5():
    assert match(P(0.01639), 0.01639, "exact5")
    assert not match(P(0.0164), 0.01639, "exact5")
    assert match(P(1.639, True), 0.01639, "exact5")


def test_extract_program():
    assert extract_program("Program: add(1, 2)\nAnswer: 3") == "add(1, 2)"
    assert extract_program("Calculation: `divide(1, 2)`\nAnswer: 0.5") == "divide(1, 2)"
    assert extract_program("Answer: 3") is None
    assert extract_answer_span("blah\nAnswer: 3\n") == "3"
