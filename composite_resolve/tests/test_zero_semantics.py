# composite-resolve - Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later - see LICENSE
"""Tests for scalar zero handling.

In composite arithmetic, scalar 0 is NOT annihilation - it is ZERO (|1|_{-1}).
Multiplying by 0 shifts dimensions down by 1, preserving information.
This is what makes 0/0 resolvable.
"""
import math
import pytest
from composite_resolve._core import Composite, ZERO, INF, R, sin, cos


class TestScalarZeroMultiplication:
    """0 * Composite must convolve with ZERO, not annihilate."""

    def test_r5_times_zero(self):
        result = R(5) * 0
        assert result.coeff(-1) == 5.0, f"expected |5|_-1, got {result}"

    def test_zero_times_r5(self):
        result = 0 * R(5)
        assert result.coeff(-1) == 5.0

    def test_inf_times_zero(self):
        result = INF * 0
        assert result.st() == 1.0, f"INF * 0 should be |1|_0, got {result}"

    def test_chained_zero_inf(self):
        result = 5 * INF * 0 * INF
        assert result.coeff(1) == 5.0, f"expected |5|_1, got {result}"


class TestScalarZeroDivision:
    """Composite / 0 must uplift to Composite / ZERO."""

    def test_r5_div_zero(self):
        result = R(5) / 0
        assert result.coeff(1) == 5.0, f"expected |5|_1, got {result}"

    def test_zero_div_zero(self):
        result = ZERO / 0
        assert result.st() == 1.0, f"ZERO / 0 = ZERO / ZERO = |1|_0"


class TestScalarZeroAddition:
    """Adding 0: no uplift when dim-0 exists, uplift when only non-dim-0."""

    def test_r5_plus_zero_no_uplift(self):
        result = R(5) + 0
        assert result.st() == 5.0

    def test_inf_plus_zero_uplift(self):
        result = INF + 0
        assert result.coeff(-1) == 1.0, f"expected ZERO added, got {result}"


class TestScalarZeroSubtraction:
    """Subtracting 0: same uplift logic as addition."""

    def test_r5_minus_zero_no_uplift(self):
        result = R(5) - 0
        assert result.st() == 5.0

    def test_inf_minus_zero_uplift(self):
        result = INF - 0
        # INF - ZERO = |1|_1 + |-1|_{-1} (two terms, different dims)
        assert result.coeff(1) == 1.0


class TestExpressedZero:
    """When subtraction cancels a real value, tag as expressed zero."""

    def test_r5_minus_5_tagged(self):
        result = R(5) - 5
        assert result._expressed_zero is True

    def test_expressed_zero_times_inf(self):
        """(R(5) - 5) * INF: expressed zero uplifts to ZERO, ZERO * INF = 1."""
        ez = R(5) - 5
        result = ez * INF
        assert result.st() == 1.0, f"expected |1|_0, got {result}"

    def test_zero_minus_zero_shift(self):
        """ZERO - ZERO should shift to higher zero power |1|_{-2}."""
        result = ZERO - ZERO
        assert result.coeff(-2) == 1.0, f"expected |1|_-2, got {result}"


class TestSinCosPreservation:
    """sin/cos must still work correctly after the zero uplift changes."""

    def test_sin_zero_derivative(self):
        assert sin(ZERO).d(1) == 1.0

    def test_sin_zero_over_zero(self):
        assert abs((sin(ZERO) / ZERO).st() - 1.0) < 1e-10

    def test_cos_zero_value(self):
        assert abs(cos(ZERO).st() - 1.0) < 1e-10

    def test_sin_at_pi(self):
        """sin(pi + h): sin(pi) ~ 1e-16 (not exactly 0 in float).
        The skip should handle this by checking != 0."""
        x = Composite({0: math.pi, -1: 1.0})
        result = sin(x)
        assert abs(result.st()) < 1e-10, f"sin(pi) should be ~0, got {result.st()}"

    def test_cos_at_pi(self):
        """cos(pi + h) = -1 + h²/2 - ..."""
        x = Composite({0: math.pi, -1: 1.0})
        result = cos(x)
        assert abs(result.st() - (-1.0)) < 1e-10


class TestNoUpliftForZeroCoefficients:
    """Zero coefficients in multi-term composites must stay zero.
    Only whole-number zeros (tagged _expressed_zero) uplift."""

    def test_zero_coeff_stays_zero(self):
        """A composite with a zero first-derivative coefficient:
        |100|_0 + |0|_{-1} + |80|_{-2} must preserve the zero at -1."""
        c = Composite({0: 100, -2: 80})
        # coeff at -1 is 0 (absent) -- first derivative is zero
        assert c.coeff(-1) == 0.0, "zero coefficient must stay zero"

    def test_multiply_preserves_zero_coeffs(self):
        """Multiplying a composite with absent dims must not fill them."""
        c = Composite({0: 3, -2: 1})  # no dim -1 term
        result = c * R(2)
        assert result.coeff(-1) == 0.0
