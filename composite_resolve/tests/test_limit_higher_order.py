# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T3: Deep cancellation (higher-order Taylor matching)."""
import pytest
from composite_resolve import limit
from composite_resolve.math import sin, cos, exp, tan


@pytest.mark.parametrize("name,order,f,at,expected", [
    ("(exp(x)-1-x)/x^2", 2,
     lambda x: (exp(x)-1-x)/x**2, 0, 0.5),
    ("(exp(x)-1-x-x^2/2)/x^3", 3,
     lambda x: (exp(x)-1-x-x**2/2)/x**3, 0, 1.0/6),
    ("(exp(x)-1-x-x^2/2-x^3/6)/x^4", 4,
     lambda x: (exp(x)-1-x-x**2/2-x**3/6)/x**4, 0, 1.0/24),
    ("(sin(x)-x+x^3/6)/x^5", 5,
     lambda x: (sin(x)-x+x**3/6)/x**5, 0, 1.0/120),
    ("(sin(tan(x))-tan(sin(x)))/x^7", 7,
     lambda x: (sin(tan(x))-tan(sin(x)))/x**7, 0, -1.0/30),
    ("(cos(x)-1+x^2/2-x^4/24)/x^6", 6,
     lambda x: (cos(x)-1+x**2/2-x**4/24)/x**6, 0, -1.0/720),
])
def test_higher_order(name, order, f, at, expected):
    trunc = max(20, order + 5)
    result = limit(f, to=at, truncation=trunc)
    tol = 1e-6 if order <= 4 else 1e-4
    err = abs(result - expected)
    rel = err / (abs(expected) + 1e-100)
    assert err < tol or rel < tol, f"{name}: got {result}, expected {expected}, err={err:.2e}"
