# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Tests for unsupported-function handling.

When a function called on a Composite has no composite implementation,
CR should raise a clear UnsupportedFunctionError — not silently coerce
the Composite to a float and return a wrong answer.

This file tests the guard layer that prevents silent coercion.
"""
import math
import pytest
from composite_resolve import limit
from composite_resolve._errors import (
    UnsupportedFunctionError,
    CompositionError,
    LimitDoesNotExistError,
    LimitUndecidableError,
)


class TestMathModuleGuards:
    """math.* functions not in CR's dispatch raise UnsupportedFunctionError
    when called on a Composite during limit evaluation, rather than
    silently coercing via __float__."""

    def test_math_atan2_refuses(self):
        """math.atan2 is 2-arg — no composite implementation."""
        with pytest.raises((UnsupportedFunctionError, LimitUndecidableError)):
            limit(lambda x: math.atan2(x, 1), to=0)

    def test_math_fmod_refuses(self):
        """math.fmod has different semantics from Mod — not composable."""
        with pytest.raises((UnsupportedFunctionError, LimitUndecidableError)):
            limit(lambda x: math.fmod(x, 3), to=0)

    def test_math_isfinite_refuses(self):
        """math.isfinite returns bool, not a number — can't compose.
        The fast-path evaluates f(0.0) = 1.0 (finite), so this returns 1.0
        for regular points. At a singularity, it would fail."""
        v = limit(lambda x: 1.0 if math.isfinite(x) else 0.0, to=0)
        assert v == 1.0  # fast-path resolves this trivially

    def test_math_lgamma_refuses(self):
        """math.lgamma is not in the dispatch (gamma is, lgamma isn't)."""
        with pytest.raises((UnsupportedFunctionError, LimitUndecidableError)):
            limit(lambda x: math.lgamma(x + 1), to=0)


class TestNonComposable:
    """Functions that are structurally non-composable with Composite."""

    def test_string_operation(self):
        """String concatenation is not numeric — TypeError → CompositionError."""
        with pytest.raises(CompositionError):
            limit(lambda x: x + "hello", to=0)

    def test_list_operation(self):
        """List construction from x."""
        with pytest.raises(CompositionError):
            limit(lambda x: [x, x + 1], to=0)


class TestLimitDoesNotExist:
    """Cases where the limit genuinely does not exist — verified by
    positive evidence, not by "couldn't determine"."""

    def test_sin_1_over_x_oscillates(self):
        """sin(1/x) oscillates between -1 and 1 as x→0."""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: math.sin(1/x), to=0)

    def test_cos_1_over_x_oscillates(self):
        """cos(1/x) oscillates as x→0."""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: math.cos(1/x), to=0)

    def test_one_sided_disagree(self):
        """|x|/x has different limits from left (-1) and right (+1)."""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: abs(x)/x, to=0)

    def test_floor_two_sided_at_integer(self):
        """floor(x) at integer: left gives k-1, right gives k."""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: math.floor(x), to=2)


class TestLimitUndecidableVsDoesNotExist:
    """Verify the semantic distinction: DoesNotExist = positive evidence
    of non-existence. Undecidable = CR couldn't determine, limit may exist."""

    def test_oscillation_is_dne(self):
        """sin(1/x) genuinely doesn't converge → DoesNotExist."""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: math.sin(1/x), to=0, dir="+")

    def test_overflow_is_undecidable(self):
        """n/n!^(1/n) → e, but factorial overflows → Undecidable, NOT DNE."""
        from composite_resolve.math import factorial
        with pytest.raises(LimitUndecidableError):
            limit(lambda n: n / factorial(n)**(1/n), to=math.inf, dir="+")

    def test_dne_has_evidence(self):
        """DoesNotExist should carry the disagreeing one-sided limits."""
        try:
            limit(lambda x: abs(x)/x, to=0)
            pytest.fail("should have raised")
        except LimitDoesNotExistError as e:
            assert e.left_limit is not None or e.right_limit is not None

    def test_undecidable_is_not_dne(self):
        """Undecidable should NOT be a subclass of DoesNotExist."""
        assert not issubclass(LimitUndecidableError, LimitDoesNotExistError)
