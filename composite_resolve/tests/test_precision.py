# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T10: Numerical precision requirements."""
import math

import pytest

from composite_resolve import limit, taylor
from composite_resolve.math import sin, cos, exp, log, sqrt, tan


@pytest.mark.parametrize("name,f,at,expected,min_digits", [
    ("sin(x)/x",        lambda x: sin(x)/x,           0, 1.0, 14),
    ("(exp(x)-1)/x",    lambda x: (exp(x)-1)/x,       0, 1.0, 14),
    ("(1-cos x)/x²",    lambda x: (1-cos(x))/x**2,    0, 0.5, 14),
    ("log(1+x)/x",      lambda x: log(1+x)/x,         0, 1.0, 14),
    ("(x²-1)/(x-1)",    lambda x: (x**2-1)/(x-1),     1, 2.0, 14),
])
def test_T10_01_basic_limits_14_digits(name, f, at, expected, min_digits):
    r = limit(f, to=at)
    err = abs(r - expected)
    sig_digits = -math.log10(err / (abs(expected) + 1e-100) + 1e-100)
    assert sig_digits >= min_digits or err < 1e-14, (
        f"{name}: err={err:.2e}, ~{sig_digits:.0f} digits (need {min_digits})"
    )


@pytest.mark.parametrize("name,f,at,expected,min_digits", [
    ("(exp(x)-1-x)/x²",
     lambda x: (exp(x)-1-x)/x**2,             0, 0.5, 10),

    ("(exp(x)-1-x-x²/2)/x³",
     lambda x: (exp(x)-1-x-x**2/2)/x**3,     0, 1/6, 10),

    ("(sin(x)-x)/x³",
     lambda x: (sin(x)-x)/x**3,               0, -1/6, 10),
])
def test_T10_02_higher_order_10_digits(name, f, at, expected, min_digits):
    r = limit(f, to=at)
    err = abs(r - expected)
    sig_digits = -math.log10(err / (abs(expected) + 1e-100) + 1e-100)
    assert sig_digits >= min_digits or err < 1e-10, (
        f"{name}: err={err:.2e}, ~{sig_digits:.0f} digits (need {min_digits})"
    )


@pytest.mark.parametrize("name,f,at,expected,min_digits", [
    ("(x²+1)/(x²-1) →∞",
     lambda x: (x**2+1)/(x**2-1), math.inf, 1.0, 12),

    ("(2x+3)/(5x-1) →∞",
     lambda x: (2*x+3)/(5*x-1),  math.inf, 0.4, 12),
])
def test_T10_04_infinity_limits_12_digits(name, f, at, expected, min_digits):
    r = limit(f, to=at)
    err = abs(r - expected)
    sig_digits = -math.log10(err / (abs(expected) + 1e-100) + 1e-100)
    assert sig_digits >= min_digits or err < 1e-12, (
        f"{name}: err={err:.2e}, ~{sig_digits:.0f} digits (need {min_digits})"
    )


def test_T10_05_taylor_coefficients_14_digits():
    c = taylor(lambda x: exp(x), at=0, order=5)
    expected = [1, 1, 0.5, 1/6, 1/24, 1/120]
    max_err = max(abs(c[i] - expected[i]) for i in range(6))
    assert max_err < 1e-14, (
        f"exp(x) Taylor coeffs: max_err={max_err:.2e} (need < 1e-14)"
    )
