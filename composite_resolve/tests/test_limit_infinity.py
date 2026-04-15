# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T4: Limits at infinity."""
import math

import pytest

from composite_resolve import limit
from composite_resolve.math import sin, cos, exp, log, sqrt, atan


@pytest.mark.parametrize("name,f,at,expected", [
    ("T4.01 (1+1/x)^x → ∞",
     lambda x: (1+1/x)**x,              math.inf, math.e),

    ("T4.02 x·sin(1/x) → ∞",
     lambda x: x*sin(1/x),              math.inf, 1.0),

    ("T4.03 (x²+1)/(x²-1) → ∞",
     lambda x: (x**2+1)/(x**2-1),       math.inf, 1.0),

    ("T4.04 sqrt(x²+x) - x → ∞",
     lambda x: sqrt(x**2+x) - x,        math.inf, 0.5),

    ("T4.05 x·(sqrt(x²+1)-x) → ∞",
     lambda x: x*(sqrt(x**2+1)-x),      math.inf, 0.5),

    ("T4.06 (2x+3)/(5x-1) → ∞",
     lambda x: (2*x+3)/(5*x-1),         math.inf, 0.4),

    ("T4.07 atan(x) → ∞",
     lambda x: atan(x),                 math.inf, math.pi/2),

    ("T4.08 x^(-1/log(x)) → ∞",
     lambda x: x**(-1/log(x)),          math.inf, 1/math.e),
])
def test_limit_infinity(name, f, at, expected):
    result = limit(f, to=at)
    assert abs(result - expected) < 1e-4, (
        f"{name}: got {result}, expected {expected}, err={abs(result - expected):.2e}"
    )
