# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Tests for the @safe decorator."""
import math
import pytest
from composite_resolve import safe


class TestSafeNormalPath:
    """Normal inputs should run the original function with no change."""

    def test_float_passthrough(self):
        @safe
        def f(x):
            return x ** 2
        assert f(3.0) == 9.0

    def test_negative(self):
        @safe
        def f(x):
            return x ** 2
        assert f(-2.0) == 4.0

    def test_trig(self):
        @safe
        def f(x):
            return math.sin(x)
        assert abs(f(1.0) - math.sin(1.0)) < 1e-15

    def test_returns_int(self):
        @safe
        def f(x):
            return x + 1
        assert f(5) == 6


class TestSafeZeroDivision:
    """Functions that raise ZeroDivisionError at the singularity."""

    def test_sinc(self):
        @safe
        def sinc(x):
            return math.sin(x) / x
        assert abs(sinc(0) - 1.0) < 1e-10

    def test_polynomial_ratio(self):
        @safe
        def f(x):
            return (x ** 2 - 1) / (x - 1)
        assert abs(f(1) - 2.0) < 1e-10

    def test_cubic_ratio(self):
        @safe
        def f(x):
            return (x ** 3 - 8) / (x - 2)
        assert abs(f(2) - 12.0) < 1e-10

    def test_expm1_over_x(self):
        @safe
        def f(x):
            return (math.exp(x) - 1) / x
        assert abs(f(0) - 1.0) < 1e-10


class TestSafeValueError:
    """Functions that raise ValueError (e.g., log(0))."""

    def test_entropy(self):
        @safe
        def entropy(p):
            return -p * math.log(p)
        assert abs(entropy(0) - 0.0) < 1e-6

    def test_x_ln_x(self):
        @safe
        def f(x):
            return x * math.log(x)
        assert abs(f(0) - 0.0) < 1e-6


class TestSafeNaN:
    """Functions that return NaN instead of raising."""

    def test_inf_minus_inf(self):
        """Create a NaN from inf - inf."""
        @safe
        def f(x):
            if x == 0:
                return float('nan')
            return math.sin(x) / x
        assert abs(f(0) - 1.0) < 1e-10


class TestSafeMixedUsage:
    """Test @safe across a range including the singularity."""

    def test_sinc_range(self):
        @safe
        def sinc(x):
            return math.sin(x) / x

        for x in [-2, -1, -0.5, 0, 0.5, 1, 2]:
            y = sinc(x)
            if x == 0:
                assert abs(y - 1.0) < 1e-10
            else:
                assert abs(y - math.sin(x)/x) < 1e-10

    def test_ratio_range(self):
        @safe
        def f(x):
            return (x ** 2 - 4) / (x - 2)

        for x in range(-3, 6):
            y = f(x)
            expected = x + 2  # simplified form
            assert abs(y - expected) < 1e-8, f"f({x}) = {y}, expected {expected}"

    def test_preserves_name(self):
        @safe
        def my_function(x):
            return x
        assert my_function.__name__ == "my_function"
