# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Tests for LimitUndecidableError — cases where composite arithmetic
plus numerical extrapolation cannot determine the limit, but the
mathematical limit may still exist.

These are NOT bugs. They document the boundary of what composite-resolve
can handle — honest refusal rather than wrong answers. Users encountering
these should fall back to a symbolic engine (SymPy) or higher-precision
tool (mpmath).
"""
import math
import pytest
from composite_resolve import limit
from composite_resolve._errors import LimitUndecidableError
from composite_resolve.math import exp, log, factorial, sqrt


class TestOverflowBeyondDoublePrecision:
    """Limits that require probe values beyond double-precision range."""

    def test_stirling_ratio(self):
        """n / factorial(n)^(1/n) → e. Needs n >> 170 but factorial
        overflows double at n=171."""
        with pytest.raises(LimitUndecidableError):
            limit(lambda n: n / factorial(n)**(1/n), to=math.inf, dir="+")

    def test_stirling_approximation(self):
        """n! / ((n/e)^n · √(2πn)) → 1. Probes converge but not within
        tolerance due to factorial overflow at n=171. Returns a value ~1.0006
        which is close but not within 1e-4 — a precision limitation, not a
        crash. We accept either Undecidable or a near-1 value."""
        try:
            v = limit(lambda n: factorial(n) / ((n/math.e)**n * sqrt(2*math.pi*n)),
                      to=math.inf, dir="+")
            assert abs(v - 1.0) < 0.01, f"expected ~1, got {v}"
        except LimitUndecidableError:
            pass

    def test_exp_ratio_overflow(self):
        """(2·exp(3x) / (exp(2x)+1))^(1/x) → e. exp(3x) overflows at x≈230."""
        with pytest.raises(LimitUndecidableError):
            limit(lambda x: (2*math.exp(3*x) / (math.exp(2*x)+1))**(1/x),
                  to=math.inf, dir="+")

    def test_exponential_base_overflow(self):
        """3^x · 3^(-x-1) · (x+1)²/x² → 1/3. 3^x overflows at x≈631."""
        with pytest.raises(LimitUndecidableError):
            limit(lambda x: 3**x * 3**(-x-1) * (x+1)**2 / x**2,
                  to=math.inf, dir="+")


class TestSubPolynomialGrowthRates:
    """Limits involving log's sub-polynomial growth — integer dimensions
    can't represent the growth-rate difference between log(x) and x^ε."""

    def test_log_plus_inverse_hyperbolic_at_inf(self):
        """log(x) + asech(x) at ∞. asech(x) undefined for x > 1 in reals."""
        from composite_resolve.math import asech
        with pytest.raises(LimitUndecidableError):
            limit(lambda x: log(x) + asech(x), to=math.inf, dir="+")

    def test_complex_log_polynomial_ratio(self):
        """(-x³·log(x)³ + (x-1)·(x+1)²·log(x+1)³) / (x²·log(x)³) → 1.
        Involves log cancellations the integer-dim system can't track."""
        with pytest.raises(LimitUndecidableError):
            limit(lambda x: (-x**3 * log(x)**3 + (x-1)*(x+1)**2 * log(x+1)**3)
                  / (x**2 * log(x)**3),
                  to=math.inf, dir="+")


class TestHighOrderPolynomialOverflow:
    """Polynomials of degree so high that probes overflow before convergence."""

    def test_binomial_difference_2000(self):
        """(x^2000 - (x+1)^2000) / x^1999 → -2000. x^2000 overflows at x≈10."""
        with pytest.raises(LimitUndecidableError):
            limit(lambda x: (x**2000 - (x+1)**2000) / x**1999,
                  to=math.inf, dir="+")


class TestCompositeArithmeticLimitations:
    """Cases where composite arithmetic's integer-dimension model
    is insufficient but the limit exists."""

    def test_nested_1_inf_form(self):
        """((x+1)^(1/x) - E) / x → ~-1.359. The 1^∞ form resolves to e
        algebraically, but the NEXT order correction (the - E part) requires
        higher-order Taylor that composite arithmetic drops."""
        with pytest.raises(LimitUndecidableError):
            limit(lambda x: ((x+1)**(1/x) - math.e) / x, to=0.0, dir="+")

    def test_inverse_trig_at_boundary(self):
        """asec(sin(x)/x) → 0 as x→0. sin(x)/x → 1, asec(1) = 0. But
        asec is undefined for |arg| < 1, which makes all neighborhood probes
        fail even though the boundary value is well-defined."""
        from composite_resolve.math import asec
        with pytest.raises(LimitUndecidableError):
            limit(lambda x: asec(math.sin(x)/x), to=0.0)
