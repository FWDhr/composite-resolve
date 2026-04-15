# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Error handling and edge case tests.

Malformed inputs, non-numeric returns, recursive calls, type errors,
boundary conditions, and every way a user can misuse the API.
"""
import math
import pytest
from composite_resolve import (
    resolve, limit, evaluate, safe, taylor, classify, residue,
    LimitDoesNotExistError, LimitDivergesError,
    SingularityError, CompositionError,
)
from composite_resolve.math import sin, cos, exp, log


# =============================================================================
# MALFORMED INPUTS TO limit()
# =============================================================================

class TestLimitBadInputs:

    def test_nan_target(self):
        with pytest.raises(ValueError):
            limit(lambda x: x, to=float('nan'))

    def test_none_target(self):
        with pytest.raises((TypeError, ValueError)):
            limit(lambda x: x, to=None)

    def test_string_target(self):
        with pytest.raises((TypeError, ValueError)):
            limit(lambda x: x, to="zero")

    def test_inf_dir_both(self):
        """Infinity with dir='both' — should still work (single direction)."""
        result = limit(lambda x: 1/x, to=math.inf)
        assert result == 0.0

    def test_negative_inf(self):
        result = limit(lambda x: 1/x, to=-math.inf)
        assert result == 0.0


# =============================================================================
# NON-COMPOSABLE FUNCTIONS
# =============================================================================

class TestNonComposable:

    def test_string_operation(self):
        with pytest.raises(CompositionError):
            limit(lambda x: x + "hello", to=0)

    def test_list_operation(self):
        with pytest.raises(CompositionError):
            limit(lambda x: [x, x], to=0)

    def test_none_return(self):
        """Function returns None instead of a number."""
        with pytest.raises((CompositionError, TypeError, AttributeError)):
            limit(lambda x: None, to=0)

    def test_dict_return(self):
        with pytest.raises((CompositionError, TypeError, AttributeError)):
            limit(lambda x: {"value": x}, to=0)


# =============================================================================
# DIVERGENT AND NON-EXISTENT
# =============================================================================

class TestDivergentErrors:

    def test_1_over_x_both(self):
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: 1/x, to=0)

    def test_1_over_x_right(self):
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x, to=0, dir="+")
        assert exc_info.value.value == math.inf

    def test_1_over_x_left(self):
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x, to=0, dir="-")
        assert exc_info.value.value == -math.inf

    def test_1_over_x_squared(self):
        """1/x^2 → +inf from both sides (same direction)."""
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x**2, to=0)
        assert exc_info.value.value == math.inf

    def test_exp_at_inf(self):
        with pytest.raises(LimitDivergesError):
            limit(lambda x: exp(x), to=math.inf)

    def test_sin_1_over_x(self):
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: sin(1/x), to=0)

    def test_cos_1_over_x(self):
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: cos(1/x), to=0)

    def test_diverges_has_value_attribute(self):
        """LimitDivergesError should have .value."""
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x, to=0, dir="+")
        assert hasattr(exc_info.value, 'value')

    def test_dne_has_left_right(self):
        """LimitDoesNotExistError should have .left_limit and .right_limit."""
        with pytest.raises(LimitDoesNotExistError) as exc_info:
            limit(lambda x: 1/x, to=0)
        assert hasattr(exc_info.value, 'left_limit')
        assert hasattr(exc_info.value, 'right_limit')


# =============================================================================
# evaluate() STRICTNESS
# =============================================================================

class TestEvaluateStrict:

    def test_removable_works(self):
        assert abs(evaluate(lambda x: (x**2-1)/(x-1), at=1) - 2.0) < 1e-8

    def test_pole_raises(self):
        with pytest.raises(SingularityError):
            evaluate(lambda x: 1/x, at=0)

    def test_regular_works(self):
        """Regular point — no singularity, should still work."""
        assert abs(evaluate(lambda x: exp(x), at=0) - 1.0) < 1e-8


# =============================================================================
# residue() ERRORS
# =============================================================================

class TestResidueErrors:

    def test_residue_at_pole(self):
        assert abs(residue(lambda x: 1/x, at=0) - 1.0) < 1e-4

    def test_residue_not_pole(self):
        with pytest.raises(SingularityError):
            residue(lambda x: sin(x)/x, at=0)


# =============================================================================
# @safe EDGE CASES
# =============================================================================

class TestSafeErrors:

    def test_safe_non_composable(self):
        """@safe on a function that can't be composed — should raise."""
        @safe
        def f(x):
            return x + "hello"
        with pytest.raises((CompositionError, TypeError)):
            f(0)

    def test_safe_always_fails(self):
        """Function that raises for all inputs."""
        @safe
        def f(x):
            raise RuntimeError("always fails")
        with pytest.raises(RuntimeError):
            f(0)

    def test_safe_normal_path_not_affected(self):
        """Non-singular inputs should not go through resolve."""
        call_count = [0]

        @safe
        def f(x):
            call_count[0] += 1
            return x ** 2

        f(3)
        assert call_count[0] == 1  # called once, not twice (no resolve)


# =============================================================================
# taylor() EDGE CASES
# =============================================================================

class TestTaylorErrors:

    def test_order_zero(self):
        c = taylor(lambda x: exp(x), at=0, order=0)
        assert len(c) == 1
        assert abs(c[0] - 1.0) < 1e-10

    def test_order_one(self):
        c = taylor(lambda x: sin(x), at=0, order=1)
        assert len(c) == 2
        assert abs(c[0] - 0.0) < 1e-10
        assert abs(c[1] - 1.0) < 1e-10

    def test_constant_function(self):
        c = taylor(lambda x: 5.0, at=0, order=3)
        assert abs(c[0] - 5.0) < 1e-10
        for i in range(1, 4):
            assert abs(c[i]) < 1e-10

    def test_at_nonzero(self):
        """Taylor at x=1 for x^2: [1, 2, 1, 0, ...]."""
        c = taylor(lambda x: x**2, at=1, order=3)
        assert abs(c[0] - 1.0) < 1e-10  # f(1)
        assert abs(c[1] - 2.0) < 1e-10  # f'(1)/1!
        assert abs(c[2] - 1.0) < 1e-10  # f''(1)/2!
        assert abs(c[3]) < 1e-10         # f'''(1)/3!


# =============================================================================
# classify() EDGE CASES
# =============================================================================

class TestClassifyEdge:

    def test_regular_point(self):
        from composite_resolve import Regular
        info = classify(lambda x: x**2 + 1, at=0)
        assert isinstance(info, Regular)
        assert abs(info.value - 1.0) < 1e-10

    def test_removable_at_nonzero(self):
        from composite_resolve import Removable
        info = classify(lambda x: (x**3 - 8)/(x - 2), at=2)
        assert isinstance(info, Removable)
        assert abs(info.value - 12.0) < 1e-6

    def test_higher_order_pole(self):
        from composite_resolve import Pole
        info = classify(lambda x: 1/x**3, at=0)
        assert isinstance(info, Pole)
        assert info.order == 3


# =============================================================================
# resolve() WITH DIVERGENT RESULTS
# =============================================================================

class TestResolveDivergent:

    def test_resolve_returns_inf(self):
        """resolve() should return inf, not raise."""
        result = resolve(lambda x: 1/x, at=0, dir="+")
        assert result == math.inf

    def test_resolve_returns_neg_inf(self):
        result = resolve(lambda x: 1/x, at=0, dir="-")
        assert result == -math.inf

    def test_resolve_both_diverges_same_direction(self):
        """1/x^2 diverges to +inf from both sides."""
        result = resolve(lambda x: 1/x**2, at=0)
        assert result == math.inf

    def test_resolve_both_diverges_different(self):
        """1/x from both sides — still raises because directions disagree."""
        with pytest.raises(LimitDoesNotExistError):
            resolve(lambda x: 1/x, at=0)


# =============================================================================
# RECURSIVE / NESTED LIMIT CALLS
# =============================================================================

class TestRecursive:

    def test_limit_inside_function(self):
        """A function that itself calls resolve internally."""
        def f(x):
            # At x=0, this calls resolve inside the function
            if x == 0:
                return resolve(lambda t: sin(t)/t, at=0)
            return sin(x)/x

        @safe
        def g(x):
            return f(x)

        assert abs(g(0) - 1.0) < 1e-10
        assert abs(g(0.5) - math.sin(0.5)/0.5) < 1e-10

    def test_limit_of_limit_result(self):
        """Use the result of one limit in another."""
        e = limit(lambda x: (1+x)**(1/x), to=0)  # = e
        result = limit(lambda x: (x**2 - e**2)/(x - e), to=e)  # = 2e
        assert abs(result - 2*math.e) < 1e-6


# =============================================================================
# BOUNDARY VALUES
# =============================================================================

class TestBoundary:

    def test_zero_function(self):
        """f(x) = 0 at all points."""
        assert resolve(lambda x: 0*x, at=0) == 0.0

    def test_identity(self):
        """f(x) = x. Limit at 5 is 5."""
        assert resolve(lambda x: x, at=5) == 5.0

    def test_constant(self):
        """f(x) = 42. Limit everywhere is 42."""
        assert resolve(lambda x: 42, at=0) == 42
        assert resolve(lambda x: 42, at=math.inf) == 42

    def test_very_high_order_polynomial(self):
        """x^20 / x^20 at 0 = 1."""
        assert abs(resolve(lambda x: x**20 / x**20, at=0) - 1.0) < 1e-8

    def test_negative_point(self):
        """Limit at negative point."""
        assert abs(resolve(lambda x: (x**2 - 4)/(x + 2), at=-2) - (-4.0)) < 1e-8
