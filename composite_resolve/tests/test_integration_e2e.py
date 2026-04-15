# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T9: Real-world end-to-end scenarios."""
import math

import pytest

from composite_resolve import limit, resolve
from composite_resolve.math import sin, cos, exp, log, sqrt


@pytest.mark.parametrize("name,f,at,kwargs,expected,tol", [
    ("T9.01 sinc²(0)",
     lambda x: (sin(x)/x)**2,              0, {},          1.0,            1e-6),

    ("T9.02 cross-entropy(p=0,y=0)",
     lambda p: -(0*log(p) + 1*log(1-p)),   0, {"dir": "+"}, 0.0,           1e-6),

    ("T9.03 (1-cos x)/x²",
     lambda x: (1-cos(x))/x**2,            0, {},          0.5,            1e-6),

    ("T9.04 transfer fn at s=1",
     lambda s: (s**2-1)/(s-1),              1, {},          2.0,            1e-6),

    ("T9.06 entropy(-p·log p)",
     lambda p: -p*log(p),                   0, {"dir": "+"}, 0.0,           1e-4),
])
def test_resolve_e2e(name, f, at, kwargs, expected, tol):
    result = resolve(f, at=at, **kwargs)
    assert abs(result - expected) < tol, (
        f"{name}: got {result}, expected {expected}, err={abs(result - expected):.2e}"
    )


@pytest.mark.parametrize("name,f,at,expected,tol", [
    ("T9.05 compound interest",
     lambda n: (1+0.05/n)**n,     math.inf, math.exp(0.05), 1e-4),

    ("T9.07 x²·exp(-x²) at ∞",
     lambda x: x**2 * exp(-x**2), math.inf, 0.0,            1e-4),
])
def test_limit_e2e(name, f, at, expected, tol):
    result = limit(f, to=at)
    assert abs(result - expected) < tol, (
        f"{name}: got {result}, expected {expected}, err={abs(result - expected):.2e}"
    )
