# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T6: Taylor coefficient extraction."""
import math

import pytest

from composite_resolve import taylor
from composite_resolve.math import sin, cos, exp, log


@pytest.mark.parametrize("name,f,at,order,expected", [
    ("T6.01 exp(x) at 0",    lambda x: exp(x),    0, 5,
     [1, 1, 0.5, 1/6, 1/24, 1/120]),

    ("T6.02 sin(x) at 0",    lambda x: sin(x),    0, 5,
     [0, 1, 0, -1/6, 0, 1/120]),

    ("T6.03 cos(x) at 0",    lambda x: cos(x),    0, 5,
     [1, 0, -0.5, 0, 1/24, 0]),

    ("T6.04 1/(1-x) at 0",   lambda x: 1/(1-x),   0, 5,
     [1, 1, 1, 1, 1, 1]),

    ("T6.05 log(1+x) at 0",  lambda x: log(1+x),  0, 5,
     [0, 1, -0.5, 1/3, -0.25, 0.2]),

    ("T6.06 exp(x) at 1",    lambda x: exp(x),    1, 3,
     [math.e, math.e, math.e/2, math.e/6]),
])
def test_taylor(name, f, at, order, expected):
    c = taylor(f, at=at, order=order)
    for i in range(len(expected)):
        assert abs(c[i] - expected[i]) < 1e-8, (
            f"{name}: c[{i}]={c[i]:.10f}, expected {expected[i]:.10f}"
        )
