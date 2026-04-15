# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T1: Basic removable singularities."""
import pytest
from composite_resolve import limit
from composite_resolve.math import sin, cos, exp, log, sqrt, tan


@pytest.mark.parametrize("name,f,at,expected", [
    ("sin(x)/x",              lambda x: sin(x)/x,                    0,  1.0),
    ("(exp(x)-1)/x",          lambda x: (exp(x)-1)/x,               0,  1.0),
    ("(1-cos(x))/x^2",        lambda x: (1-cos(x))/x**2,            0,  0.5),
    ("tan(x)/x",              lambda x: tan(x)/x,                   0,  1.0),
    ("(x^2-1)/(x-1)",         lambda x: (x**2-1)/(x-1),             1,  2.0),
    ("(x^3-8)/(x-2)",         lambda x: (x**3-8)/(x-2),             2,  12.0),
    ("log(1+x)/x",            lambda x: log(1+x)/x,                 0,  1.0),
    ("(exp(x)-exp(-x))/(2x)", lambda x: (exp(x)-exp(-x))/(2*x),     0,  1.0),
    ("(sqrt(1+x)-1)/x",       lambda x: (sqrt(1+x)-1)/x,            0,  0.5),
    ("x/sin(x)",              lambda x: x/sin(x),                   0,  1.0),
])
def test_basic_limit(name, f, at, expected):
    result = limit(f, to=at)
    assert abs(result - expected) < 1e-10, f"{name}: got {result}, expected {expected}"
