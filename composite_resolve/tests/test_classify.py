# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T7: Singularity classification."""
import math

import pytest

from composite_resolve import classify, Regular, Removable, Pole, Essential
from composite_resolve.math import sin, cos, exp, tan


@pytest.mark.parametrize("name,f,at,expected_type,expected_attrs", [
    ("T7.01 sin(x)/x at 0",
     lambda x: sin(x)/x,         0, Removable, {"value": 1.0}),

    ("T7.02 1/x at 0",
     lambda x: 1/x,              0, Pole,      {"order": 1}),

    ("T7.03 1/x² at 0",
     lambda x: 1/x**2,           0, Pole,      {"order": 2}),

    ("T7.04 exp(x) at 0",
     lambda x: exp(x),           0, Regular,   {"value": 1.0}),

    ("T7.05 (x²-4)/(x-2) at 2",
     lambda x: (x**2-4)/(x-2),   2, Removable, {"value": 4.0}),

    ("T7.06 1/(x²+1) at 0",
     lambda x: 1/(x**2+1),       0, Regular,   {"value": 1.0}),

    pytest.param(
        "T7.07 tan(x) at pi/2",
        lambda x: tan(x), math.pi/2, Pole, {"order": 1},
        marks=pytest.mark.xfail(
            reason="math.pi/2 is not exactly pi/2 — float precision limit")),
])
def test_classify(name, f, at, expected_type, expected_attrs):
    info = classify(f, at=at)
    assert isinstance(info, expected_type), (
        f"{name}: expected {expected_type.__name__}, got {type(info).__name__}"
    )
    for attr, val in expected_attrs.items():
        actual = getattr(info, attr, None)
        assert actual is not None, f"{name}: missing attribute '{attr}'"
        assert abs(actual - val) < 1e-4, (
            f"{name}: {attr}={actual}, expected {val}"
        )
