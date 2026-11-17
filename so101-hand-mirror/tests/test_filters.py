import numpy as np
import pytest

from hand_mirror.filters import OneEuroFilter, RateLimiter


def test_one_euro_reduces_jitter():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    raw = 1.0 + rng.normal(0, 0.05, 300)
    out = np.array([f(np.array([x]), i / 30)[0] for i, x in enumerate(raw)])
    assert out[50:].std() < raw[50:].std() / 2


def test_rate_limiter():
    r = RateLimiter({"a": 100.0})
    r.reset({"a": 0.0}, 0.0)
    assert r({"a": 90.0}, 0.05)["a"] == pytest.approx(5.0)
