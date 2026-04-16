# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Showcase tests for limits that composite-resolve handles algebraically.

These are the library's strength — resolved via composite arithmetic's
dimensional structure, not numerical extrapolation. Every test here should
produce the EXACT answer (within float precision) because the composite
primitive directly resolves the singularity or indeterminate form.
"""
import math
import pytest
from composite_resolve import limit
from composite_resolve._errors import LimitDivergesError
from composite_resolve.math import (
    sin, cos, tan, exp, log, ln, sqrt,
    atan, asin, acos, sinh, cosh, tanh,
    cot, sec, csc, acot,
    asinh, acosh, atanh,
    expm1, log1p, cosm1,
    floor, ceiling, frac,
    erf, gamma, factorial, binomial,
)


# =========================================================================
# REMOVABLE SINGULARITIES (0/0 forms)
# =========================================================================

class TestRemovableSingularities:
    """Classic 0/0 forms resolved algebraically via composite."""

    @pytest.mark.parametrize("name,f,at,expected", [
        ("sin(x)/x",              lambda x: sin(x)/x,               0,   1.0),
        ("(exp(x)-1)/x",          lambda x: (exp(x)-1)/x,           0,   1.0),
        ("(1-cos(x))/x²",        lambda x: (1-cos(x))/x**2,        0,   0.5),
        ("tan(x)/x",              lambda x: tan(x)/x,               0,   1.0),
        ("(x²-1)/(x-1)",         lambda x: (x**2-1)/(x-1),         1,   2.0),
        ("(x³-8)/(x-2)",         lambda x: (x**3-8)/(x-2),         2,  12.0),
        ("log(1+x)/x",           lambda x: log(1+x)/x,             0,   1.0),
        ("(exp(x)-exp(-x))/(2x)", lambda x: (exp(x)-exp(-x))/(2*x), 0,   1.0),
        ("(sqrt(1+x)-1)/x",      lambda x: (sqrt(1+x)-1)/x,        0,   0.5),
        ("x/sin(x)",             lambda x: x/sin(x),               0,   1.0),
        ("sin(x)/sqrt(1-cos(x))+", lambda x: sin(x)/sqrt(1-cos(x)), 0,   math.sqrt(2)),  # dir=+ only; left gives -√2
    ])
    def test_removable(self, name, f, at, expected):
        # Use dir="+" for cases where left/right limits differ in sign
        result = limit(f, to=at, dir="+")
        assert abs(result - expected) < 1e-10, f"{name}: {result} ≠ {expected}"


# =========================================================================
# INDETERMINATE FORMS (0·∞, 0⁰, 1^∞, ∞−∞)
# =========================================================================

class TestIndeterminateForms:
    """All five indeterminate form types."""

    def test_zero_times_infinity(self):
        """x·log(x) → 0 as x→0⁺ (0·∞ form)."""
        assert abs(limit(lambda x: x*log(x), to=0, dir="+")) < 1e-6

    def test_zero_to_zero(self):
        """x^x → 1 as x→0⁺ (0⁰ form)."""
        assert abs(limit(lambda x: x**x, to=0, dir="+") - 1.0) < 1e-4

    def test_one_to_infinity(self):
        """(1+x)^(1/x) → e as x→0 (1^∞ form)."""
        assert abs(limit(lambda x: (1+x)**(1/x), to=0) - math.e) < 1e-4

    def test_one_to_infinity_at_inf(self):
        """(1+1/x)^x → e as x→∞ (1^∞ at infinity)."""
        assert abs(limit(lambda x: (1+1/x)**x, to=math.inf) - math.e) < 1e-4

    def test_inf_minus_inf(self):
        """1/x − 1/sin(x) → 0 as x→0 (∞−∞ form)."""
        assert abs(limit(lambda x: 1/x - 1/sin(x), to=0)) < 1e-4

    def test_entropy(self):
        """−p·log(p) → 0 as p→0⁺ (Shannon entropy boundary)."""
        assert abs(limit(lambda p: -p*log(p), to=0, dir="+")) < 1e-6


# =========================================================================
# LIMITS AT INFINITY
# =========================================================================

class TestLimitsAtInfinity:
    """Finite limits at ±∞."""

    @pytest.mark.parametrize("name,f,expected", [
        ("sin(x)/x → 0",        lambda x: sin(x)/x,            0.0),
        ("1/x → 0",             lambda x: 1/x,                 0.0),
        ("atan(x) → π/2",       lambda x: atan(x),             math.pi/2),
        ("(2x+1)/(3x+2) → 2/3", lambda x: (2*x+1)/(3*x+2),    2/3),
        ("x/(x+1) → 1",         lambda x: x/(x+1),             1.0),
        ("exp(-x) → 0",         lambda x: exp(-x),              0.0),
    ])
    def test_finite_limit_at_inf(self, name, f, expected):
        result = limit(f, to=math.inf, dir="+")
        assert abs(result - expected) < 1e-6, f"{name}: {result} ≠ {expected}"


class TestDivergentLimits:
    """Limits that diverge to ±∞."""

    @pytest.mark.parametrize("name,f,to,dir,sign", [
        ("exp(x) → +∞",       lambda x: exp(x),     math.inf, "+", +1),
        ("-exp(x) → -∞",      lambda x: -exp(x),    math.inf, "+", -1),
        ("1/x → +∞ from right", lambda x: 1/x,       0,        "+", +1),
        ("1/x → -∞ from left",  lambda x: 1/x,       0,        "-", -1),
        ("log(x) → -∞ at 0⁺",  lambda x: log(x),    0,        "+", -1),
        ("log(x) → +∞ at ∞",   lambda x: log(x),    math.inf, "+", +1),
        ("x² → +∞",           lambda x: x**2,       math.inf, "+", +1),
        ("x−x² → −∞",         lambda x: x - x**2,   math.inf, "+", -1),
    ])
    def test_divergent(self, name, f, to, dir, sign):
        with pytest.raises(LimitDivergesError) as exc:
            limit(f, to=to, dir=dir)
        assert (exc.value.value > 0) == (sign > 0), (
            f"{name}: expected {'+'if sign>0 else '-'}∞, got {exc.value.value}")


# =========================================================================
# DIRECTIONAL LIMITS (one-sided, discontinuities)
# =========================================================================

class TestDirectionalLimits:
    """Floor, ceiling, and other discontinuous functions."""

    @pytest.mark.parametrize("name,f,at,dir,expected", [
        ("floor(x) at 2+",       lambda x: floor(x),      2,  "+",  2),
        ("floor(x) at 2-",       lambda x: floor(x),      2,  "-",  1),
        ("ceiling(x) at 2+",     lambda x: ceiling(x),    2,  "+",  3),
        ("ceiling(x) at 2-",     lambda x: ceiling(x),    2,  "-",  2),
        ("floor(cos(x)) at 0+",  lambda x: floor(cos(x)), 0,  "+",  0),
        ("floor(cos(x)) at 0-",  lambda x: floor(cos(x)), 0,  "-",  0),
        ("acot(x) at 0+",       lambda x: acot(x),       0,  "+",  math.pi/2),
        ("acot(x) at 0-",       lambda x: acot(x),       0,  "-", -math.pi/2),
        ("frac(x) at 2+",       lambda x: frac(x),       2,  "+",  0),
        ("frac(x) at 2-",       lambda x: frac(x),       2,  "-",  1),
    ])
    def test_directional(self, name, f, at, dir, expected):
        result = limit(f, to=at, dir=dir)
        assert abs(result - expected) < 1e-8, f"{name}: {result} ≠ {expected}"


# =========================================================================
# SPECIAL FUNCTIONS
# =========================================================================

class TestSpecialFunctions:
    """Limits involving gamma, erf, and other special functions."""

    def test_gamma_pole_cancellation(self):
        """x·Γ(x) → 1 as x→0⁺ (simple pole residue)."""
        v = limit(lambda x: x * gamma(x), to=0, dir="+")
        assert abs(v - 1.0) < 1e-4

    def test_erf_normalized(self):
        """erf(x)/x → 2/√π as x→0."""
        v = limit(lambda x: erf(x)/x, to=0)
        assert abs(v - 2/math.sqrt(math.pi)) < 1e-6

    def test_gamma_at_regular_point(self):
        """Γ(1/x + 3) → 2 as x→∞ (since Γ(3) = 2! = 2)."""
        v = limit(lambda x: gamma(1/x + 3), to=math.inf)
        assert abs(v - 2.0) < 1e-6

    def test_erf_saturation(self):
        """erf(x) → 1 as x→∞."""
        v = limit(lambda x: erf(x), to=math.inf, dir="+")
        assert abs(v - 1.0) < 1e-6


# =========================================================================
# EARLY-CANCELLATION PRIMITIVES
# =========================================================================

class TestEarlyCancellation:
    """expm1, log1p, cosm1 — numerically stable near their cancellation."""

    def test_expm1_over_x(self):
        """expm1(x)/x → 1 as x→0."""
        assert abs(limit(lambda x: expm1(x)/x, to=0) - 1.0) < 1e-10

    def test_log1p_over_x(self):
        """log1p(x)/x → 1 as x→0."""
        assert abs(limit(lambda x: log1p(x)/x, to=0) - 1.0) < 1e-10

    def test_cosm1_over_x_squared(self):
        """cosm1(x)/x² → −1/2 as x→0."""
        assert abs(limit(lambda x: cosm1(x)/x**2, to=0) - (-0.5)) < 1e-10


# =========================================================================
# FAST-PATH (regular points — no composite arithmetic needed)
# =========================================================================

class TestFastPath:
    """Regular points that resolve via plain float evaluation."""

    def test_polynomial_at_regular_point(self):
        v = limit(lambda x: x**3 + 2*x + 1, to=3)
        assert v == 34.0

    def test_trig_at_regular_point(self):
        v = limit(lambda x: sin(x) + cos(x), to=math.pi/4)
        assert abs(v - math.sqrt(2)) < 1e-10

    def test_exp_at_regular_point(self):
        v = limit(lambda x: exp(x), to=1)
        assert abs(v - math.e) < 1e-10

    def test_complex_at_regular_point(self):
        """(5^(1/x) + 3^(1/x))^x at x=2 — regular, no singularity."""
        v = limit(lambda x: (5**(1/x) + 3**(1/x))**x, to=2)
        expected = (5**0.5 + 3**0.5)**2
        assert abs(v - expected) < 1e-10
