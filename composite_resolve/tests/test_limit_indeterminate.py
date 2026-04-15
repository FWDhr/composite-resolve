# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T2: All indeterminate forms (0/0, 0*inf, 0^0, 1^inf, inf-inf)."""
import math
import pytest
from composite_resolve import limit, LimitDivergesError
from composite_resolve.math import sin, cos, exp, log, sqrt, tan


@pytest.mark.parametrize("name,f,at,dir,expected", [
    ("x*log(x) [0*inf]",         lambda x: x*log(x),          0, "+", 0.0),
    ("x^x [0^0]",                lambda x: x**x,              0, "+", 1.0),
    ("(1+x)^(1/x) [1^inf]",     lambda x: (1+x)**(1/x),     0, None, math.e),
    ("(1+1/x)^x [1^inf at inf]", lambda x: (1+1/x)**x,  math.inf, None, math.e),
    ("x^(1/log(x)) [0^0]",      lambda x: x**(1/log(x)),     0, "+", math.e),
    ("(exp(x)+x)^(1/x) [1^inf]", lambda x: (exp(x)+x)**(1/x), 0, None, math.e**2),
    ("(tan(x))^x [0^0]",        lambda x: tan(x)**x,         0, "+", 1.0),
    ("1/x - 1/sin(x) [inf-inf]", lambda x: 1/x - 1/sin(x),  0, None, 0.0),
    ("1/x^2 - 1/sin^2(x)",      lambda x: 1/x**2 - 1/sin(x)**2, 0, None, -1.0/3),
])
def test_indeterminate(name, f, at, dir, expected):
    kw = {"dir": dir} if dir else {}
    result = limit(f, to=at, **kw)
    assert abs(result - expected) < 1e-4, f"{name}: got {result}, expected {expected}"


def test_divergent_0_times_inf():
    """x*exp(1/x) at 0+ diverges to +inf."""
    with pytest.raises(LimitDivergesError):
        limit(lambda x: x*exp(1/x), to=0, dir="+")
