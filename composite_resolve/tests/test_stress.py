# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Stress tests and edge cases.

Deeply nested compositions, extreme values, barely-convergent limits,
and pathological functions that probe the boundaries of composite arithmetic.
"""
import math
import pytest
from composite_resolve import resolve, limit, safe, LimitDoesNotExistError, LimitDivergesError
from composite_resolve.math import sin, cos, exp, log, sqrt, tan, atan, tanh


# =============================================================================
# DEEPLY NESTED COMPOSITIONS
# =============================================================================

class TestNested:

    def test_sin_of_sin_over_x(self):
        """sin(sin(x)) / x at 0."""
        assert abs(resolve(lambda x: sin(sin(x))/x, at=0) - 1.0) < 1e-8

    def test_exp_of_sin_over_x(self):
        """(exp(sin(x)) - 1) / x at 0."""
        assert abs(resolve(lambda x: (exp(sin(x)) - 1)/x, at=0) - 1.0) < 1e-8

    def test_log_of_cos(self):
        """log(cos(x)) / x^2 at 0 → -1/2."""
        assert abs(resolve(lambda x: log(cos(x))/x**2, at=0) - (-0.5)) < 1e-8

    def test_triple_sin(self):
        """sin(sin(sin(x))) / x at 0 → 1."""
        assert abs(resolve(lambda x: sin(sin(sin(x)))/x, at=0) - 1.0) < 1e-6

    def test_exp_minus_1_nested(self):
        """(exp(exp(x)-1) - 1) / x at 0 → 1."""
        assert abs(resolve(lambda x: (exp(exp(x)-1) - 1)/x, at=0) - 1.0) < 1e-8

    def test_sin_over_tan(self):
        """sin(x) / tan(x) at 0 → 1."""
        assert abs(resolve(lambda x: sin(x)/tan(x), at=0) - 1.0) < 1e-8

    def test_atan_over_sin(self):
        """atan(x) / sin(x) at 0 → 1."""
        assert abs(resolve(lambda x: atan(x)/sin(x), at=0) - 1.0) < 1e-8

    def test_log_1_plus_sin(self):
        """log(1 + sin(x)) / x at 0 → 1."""
        assert abs(resolve(lambda x: log(1 + sin(x))/x, at=0) - 1.0) < 1e-8

    def test_sqrt_of_ratio(self):
        """sqrt((1-cos(x))/2) / sin(x/2) at 0+: sqrt is positive, sin(x/2) > 0 → 1."""
        # From right only — sqrt is always positive, sin(x/2) changes sign at 0
        assert abs(limit(lambda x: sqrt((1-cos(x))/2)/sin(x/2), to=0, dir="+") - 1.0) < 1e-6

    @pytest.mark.timeout(120)
    def test_chain_five_deep(self):
        """sin(atan(sin(atan(x)))) / x at 0 -> 1.
        Deeply nested trig chain causes O(n^2) dimension explosion
        through convolution. Slow by design at default truncation."""
        assert abs(resolve(
            lambda x: sin(atan(sin(atan(x))))/x, at=0) - 1.0) < 1e-6


# =============================================================================
# VERY LARGE / VERY SMALL NUMBERS
# =============================================================================

class TestExtremeValues:

    def test_at_large_point(self):
        """(x^2 - 1e12) / (x - 1e6) at x=1e6."""
        assert abs(resolve(lambda x: (x**2 - 1e12)/(x - 1e6), at=1e6) - 2e6) < 1

    def test_at_small_point(self):
        """(x^2 - 1e-12) / (x - 1e-6) at x=1e-6."""
        assert abs(resolve(lambda x: (x**2 - 1e-12)/(x - 1e-6), at=1e-6) - 2e-6) < 1e-12

    def test_large_coefficient(self):
        """1000 * sin(x) / x at 0 → 1000."""
        assert abs(resolve(lambda x: 1000*sin(x)/x, at=0) - 1000.0) < 1e-6

    def test_tiny_coefficient(self):
        """1e-10 * sin(x) / x at 0 → 1e-10."""
        assert abs(resolve(lambda x: 1e-10*sin(x)/x, at=0) - 1e-10) < 1e-20

    def test_ratio_of_large(self):
        """(1e6*x - 1e6) / (x - 1) at x=1 → 1e6."""
        assert abs(resolve(lambda x: (1e6*x - 1e6)/(x - 1), at=1) - 1e6) < 1

    def test_exp_large_negative(self):
        """exp(-100*x) * x at inf → 0."""
        assert abs(limit(lambda x: exp(-100*x) * x, to=math.inf) - 0.0) < 1e-4


# =============================================================================
# BARELY CONVERGENT / EDGE OF DETECTION
# =============================================================================

class TestBarely:

    def test_fourth_order_cancellation(self):
        """(cos(x) - 1 + x^2/2) / x^4 at 0 → 1/24."""
        assert abs(resolve(
            lambda x: (cos(x) - 1 + x**2/2)/x**4, at=0) - 1.0/24) < 1e-8

    def test_fifth_order(self):
        """(sin(x) - x + x^3/6) / x^5 at 0 → 1/120."""
        assert abs(resolve(
            lambda x: (sin(x) - x + x**3/6)/x**5, at=0) - 1.0/120) < 1e-8

    def test_sixth_order(self):
        """(cos(x) - 1 + x^2/2 - x^4/24) / x^6 at 0 → -1/720."""
        assert abs(resolve(
            lambda x: (cos(x) - 1 + x**2/2 - x**4/24)/x**6, at=0) - (-1.0/720)) < 1e-6

    def test_x_to_the_x_to_the_x(self):
        """x^(x^x) at 0+ → 0^1 = 0."""
        # x^x → 1, so x^(x^x) → x^1 = x → 0
        assert abs(limit(lambda x: x**(x**x), to=0, dir="+") - 0.0) < 1e-4

    def test_barely_removable(self):
        """(x^10 - 1) / (x - 1) at x=1 → 10."""
        assert abs(resolve(
            lambda x: (x**10 - 1)/(x - 1), at=1) - 10.0) < 1e-6

    def test_high_degree_polynomial_ratio(self):
        """(x^5 - 32) / (x - 2) at x=2 → 5*16 = 80."""
        assert abs(resolve(
            lambda x: (x**5 - 32)/(x - 2), at=2) - 80.0) < 1e-6


# =============================================================================
# PRODUCTS AND QUOTIENTS OF SINGULARITIES
# =============================================================================

class TestCombined:

    def test_product_of_two_singularities(self):
        """(sin(x)/x) * ((exp(x)-1)/x) at 0 → 1 * 1 = 1."""
        assert abs(resolve(
            lambda x: (sin(x)/x) * ((exp(x)-1)/x), at=0) - 1.0) < 1e-8

    def test_quotient_of_singularities(self):
        """(sin(x)/x) / ((exp(x)-1)/x) at 0 → 1/1 = 1."""
        assert abs(resolve(
            lambda x: (sin(x)/x) / ((exp(x)-1)/x), at=0) - 1.0) < 1e-8

    def test_sin_over_atan(self):
        """sin(x) / atan(x) at 0 → 1."""
        assert abs(resolve(lambda x: sin(x)/atan(x), at=0) - 1.0) < 1e-8

    def test_sum_of_singularities(self):
        """sin(x)/x + (exp(x)-1)/x at 0 → 2."""
        assert abs(resolve(
            lambda x: sin(x)/x + (exp(x)-1)/x, at=0) - 2.0) < 1e-8

    def test_difference_of_singularities(self):
        """sin(x)/x - (1-cos(x))/x^2 at 0 → 1 - 0.5 = 0.5."""
        assert abs(resolve(
            lambda x: sin(x)/x - (1-cos(x))/x**2, at=0) - 0.5) < 1e-8

    def test_triple_product(self):
        """(sin(x)/x) * ((exp(x)-1)/x) * (atan(x)/x) at 0 → 1."""
        assert abs(resolve(
            lambda x: (sin(x)/x) * ((exp(x)-1)/x) * (atan(x)/x), at=0) - 1.0) < 1e-6


# =============================================================================
# SAFE DECORATOR EDGE CASES
# =============================================================================

class TestSafeEdge:

    def test_safe_with_integer_input(self):
        @safe
        def f(x):
            return (x**2 - 4) / (x - 2)
        assert abs(f(2) - 4.0) < 1e-8

    def test_safe_no_singularity(self):
        """Function that never fails — @safe should be transparent."""
        @safe
        def f(x):
            return x**2 + 1
        for x in range(-10, 11):
            assert f(x) == x**2 + 1

    def test_safe_multiple_calls(self):
        """Repeated calls to same @safe function."""
        @safe
        def sinc(x):
            return math.sin(x) / x
        results = [sinc(0) for _ in range(100)]
        assert all(abs(r - 1.0) < 1e-10 for r in results)

    def test_safe_with_negative_singularity(self):
        @safe
        def f(x):
            return (x**2 - 9) / (x + 3)
        assert abs(f(-3) - (-6.0)) < 1e-8


# =============================================================================
# INFINITY EDGE CASES
# =============================================================================

class TestInfinityEdge:

    def test_polynomial_over_polynomial_same_degree(self):
        """(3x^3 + x) / (7x^3 - 2) at inf → 3/7."""
        assert abs(limit(
            lambda x: (3*x**3 + x)/(7*x**3 - 2), to=math.inf) - 3.0/7) < 1e-4

    def test_lower_over_higher_degree(self):
        """x^2 / x^3 at inf → 0."""
        assert abs(limit(lambda x: x**2/x**3, to=math.inf) - 0.0) < 1e-4

    def test_negative_infinity(self):
        """(x^2 + 1) / (x^2 - 1) at -inf → 1."""
        assert abs(limit(
            lambda x: (x**2 + 1)/(x**2 - 1), to=-math.inf) - 1.0) < 1e-4

    def test_exp_dominates_polynomial_at_inf(self):
        """x^5 * exp(-x) at inf → 0."""
        assert abs(limit(lambda x: x**5 * exp(-x), to=math.inf) - 0.0) < 1e-4

    def test_atan_at_negative_inf(self):
        """atan(x) at -inf → -pi/2."""
        assert abs(limit(lambda x: atan(x), to=-math.inf) - (-math.pi/2)) < 1e-4

    def test_tanh_at_inf(self):
        """tanh(x) at inf → 1."""
        assert abs(limit(lambda x: tanh(x), to=math.inf) - 1.0) < 1e-4


# =============================================================================
# PATHOLOGICAL FUNCTIONS
# =============================================================================

class TestPathological:

    def test_zero_over_zero_over_zero(self):
        """((x^2-1)/(x-1)) / ((x^3-1)/(x-1)) at x=1 → 2/3."""
        assert abs(resolve(
            lambda x: ((x**2-1)/(x-1)) / ((x**3-1)/(x-1)), at=1) - 2.0/3) < 1e-8

    def test_iterated_lhopital(self):
        """(exp(x) - 1 - x - x^2/2 - x^3/6) / x^4 at 0 → 1/24."""
        assert abs(resolve(
            lambda x: (exp(x) - 1 - x - x**2/2 - x**3/6)/x**4, at=0) - 1.0/24) < 1e-8

    def test_sin_squared_over_x_squared(self):
        """sin^2(x) / x^2 at 0 → 1."""
        assert abs(resolve(lambda x: sin(x)**2/x**2, at=0) - 1.0) < 1e-8

    def test_one_minus_cos_over_sin(self):
        """(1-cos(x))/sin(x) at 0 → 0."""
        assert abs(resolve(lambda x: (1-cos(x))/sin(x), at=0) - 0.0) < 1e-8

    def test_tan_minus_x_over_x_cubed(self):
        """(tan(x) - x) / x^3 at 0 → 1/3."""
        assert abs(resolve(
            lambda x: (tan(x) - x)/x**3, at=0) - 1.0/3) < 1e-6

    def test_exp_symmetry(self):
        """(exp(x) + exp(-x) - 2) / x^2 at 0 → 1."""
        assert abs(resolve(
            lambda x: (exp(x) + exp(-x) - 2)/x**2, at=0) - 1.0) < 1e-8

    def test_multiple_singularities_same_function(self):
        """(x^2-1)/(x-1) has singularity at x=1 but is fine at x=-1, 0, 2."""
        f = lambda x: (x**2 - 1)/(x - 1)
        assert abs(resolve(f, at=1) - 2.0) < 1e-8
        assert abs(resolve(f, at=0) - 1.0) < 1e-8
        assert abs(resolve(f, at=-1) - 0.0) < 1e-8
        assert abs(resolve(f, at=2) - 3.0) < 1e-8
