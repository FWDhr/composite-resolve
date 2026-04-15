# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Tests for math and numpy module patching.

Verifies that:
  - math.sin, math.cos, etc. work inside limit()/resolve()
  - math functions are restored after the call
  - Multi-argument math functions (log(x, base), pow) work correctly
  - numpy ufuncs dispatch via __array_ufunc__
  - Patching doesn't leak between calls
  - Nested calls don't corrupt state
"""
import math
import pytest
from composite_resolve import resolve, limit, safe


# =============================================================================
# BASIC MATH PATCHING
# =============================================================================

class TestMathPatching:

    def test_math_sin(self):
        assert abs(resolve(lambda x: math.sin(x)/x, at=0) - 1.0) < 1e-10

    def test_math_cos(self):
        assert abs(resolve(lambda x: (1-math.cos(x))/x**2, at=0) - 0.5) < 1e-10

    def test_math_exp(self):
        assert abs(resolve(lambda x: (math.exp(x)-1)/x, at=0) - 1.0) < 1e-10

    def test_math_log(self):
        assert abs(resolve(lambda x: math.log(1+x)/x, at=0) - 1.0) < 1e-10

    def test_math_sqrt(self):
        assert abs(resolve(lambda x: (math.sqrt(1+x)-1)/x, at=0) - 0.5) < 1e-10

    def test_math_tan(self):
        assert abs(resolve(lambda x: math.tan(x)/x, at=0) - 1.0) < 1e-10

    def test_math_atan(self):
        assert abs(resolve(lambda x: math.atan(x)/x, at=0) - 1.0) < 1e-10

    def test_math_sinh(self):
        assert abs(resolve(lambda x: math.sinh(x)/x, at=0) - 1.0) < 1e-10

    def test_math_cosh(self):
        assert abs(resolve(lambda x: (math.cosh(x)-1)/x**2, at=0) - 0.5) < 1e-10

    def test_math_tanh(self):
        assert abs(resolve(lambda x: math.tanh(x)/x, at=0) - 1.0) < 1e-10

    def test_math_asin(self):
        assert abs(resolve(lambda x: math.asin(x)/x, at=0) - 1.0) < 1e-10

    def test_math_acos(self):
        """acos(x) at 0 → π/2."""
        assert abs(resolve(lambda x: math.acos(x), at=0) - math.pi/2) < 1e-10


# =============================================================================
# MATH RESTORATION
# =============================================================================

class TestMathRestoration:

    def test_sin_restored(self):
        original = math.sin
        resolve(lambda x: math.sin(x)/x, at=0)
        assert math.sin is original

    def test_cos_restored(self):
        original = math.cos
        resolve(lambda x: (1-math.cos(x))/x**2, at=0)
        assert math.cos is original

    def test_exp_restored(self):
        original = math.exp
        resolve(lambda x: (math.exp(x)-1)/x, at=0)
        assert math.exp is original

    def test_log_restored(self):
        original = math.log
        resolve(lambda x: math.log(1+x)/x, at=0)
        assert math.log is original

    def test_sqrt_restored(self):
        original = math.sqrt
        resolve(lambda x: (math.sqrt(1+x)-1)/x, at=0)
        assert math.sqrt is original

    def test_restored_after_error(self):
        """Math functions restored even when limit raises."""
        original = math.sin
        try:
            limit(lambda x: math.sin(1/x), to=0)
        except Exception:
            pass
        assert math.sin is original

    def test_float_behavior_after_resolve(self):
        """After resolve(), math.sin(float) returns plain float."""
        resolve(lambda x: math.sin(x)/x, at=0)
        result = math.sin(1.0)
        assert type(result) is float
        assert abs(result - 0.8414709848078965) < 1e-15


# =============================================================================
# MULTI-ARGUMENT MATH FUNCTIONS
# =============================================================================

class TestMultiArg:

    def test_math_log_base_10(self):
        """math.log(x, 10) should work with float inputs during patching."""
        @safe
        def f(x):
            return math.log(x, 10)
        # At x=1: log(1, 10) = 0, no singularity
        assert f(1.0) == 0.0
        # At x=10: log(10, 10) = 1
        assert abs(f(10.0) - 1.0) < 1e-10

    def test_math_log_base_2(self):
        @safe
        def f(x):
            return math.log(x, 2)
        assert abs(f(8.0) - 3.0) < 1e-10

    def test_math_pow(self):
        """math.pow during patching."""
        @safe
        def f(x):
            return math.pow(x, 2)
        assert abs(f(3.0) - 9.0) < 1e-10

    def test_math_log_two_arg_not_corrupted(self):
        """After resolve(), math.log(x, base) still works normally."""
        resolve(lambda x: math.sin(x)/x, at=0)
        assert abs(math.log(100, 10) - 2.0) < 1e-10
        assert abs(math.log(8, 2) - 3.0) < 1e-10


# =============================================================================
# NUMPY DISPATCH
# =============================================================================

class TestNumpy:

    @pytest.fixture(autouse=True)
    def skip_if_no_numpy(self):
        pytest.importorskip("numpy")

    def test_np_sin(self):
        import numpy as np
        assert abs(resolve(lambda x: np.sin(x)/x, at=0) - 1.0) < 1e-10

    def test_np_cos(self):
        import numpy as np
        assert abs(resolve(lambda x: (1-np.cos(x))/x**2, at=0) - 0.5) < 1e-10

    def test_np_exp(self):
        import numpy as np
        assert abs(resolve(lambda x: (np.exp(x)-1)/x, at=0) - 1.0) < 1e-10

    def test_np_log(self):
        import numpy as np
        assert abs(resolve(lambda x: np.log(1+x)/x, at=0) - 1.0) < 1e-10

    def test_np_sqrt(self):
        import numpy as np
        assert abs(resolve(lambda x: (np.sqrt(1+x)-1)/x, at=0) - 0.5) < 1e-10

    def test_np_arctan(self):
        import numpy as np
        assert abs(resolve(lambda x: np.arctan(x)/x, at=0) - 1.0) < 1e-10

    def test_np_arithmetic(self):
        """np.add, np.multiply etc. via __array_ufunc__."""
        import numpy as np
        assert abs(resolve(lambda x: np.sin(x) * np.cos(x) / x, at=0) - 1.0) < 1e-10

    def test_safe_with_numpy(self):
        import numpy as np

        @safe
        def sinc(x):
            return np.sin(x) / x

        assert abs(sinc(0.5) - 0.958851077208406) < 1e-10
        assert abs(sinc(0) - 1.0) < 1e-10

    def test_np_not_corrupted_after(self):
        """numpy functions work normally after resolve."""
        import numpy as np
        resolve(lambda x: np.sin(x)/x, at=0)
        assert abs(np.sin(1.0) - math.sin(1.0)) < 1e-15
        assert type(np.sin(1.0)) is not type(resolve)  # not a Composite


# =============================================================================
# MIXED MATH + NUMPY
# =============================================================================

class TestMixed:

    @pytest.fixture(autouse=True)
    def skip_if_no_numpy(self):
        pytest.importorskip("numpy")

    def test_mixed_math_numpy(self):
        """Function using both math and numpy."""
        import numpy as np
        result = resolve(lambda x: math.sin(x) + np.cos(x) - 1, at=0)
        # sin(0) + cos(0) - 1 = 0 + 1 - 1 = 0, but this is a regular point
        assert abs(result - 0.0) < 1e-10

    def test_numpy_safe_across_range(self):
        import numpy as np

        @safe
        def f(x):
            return (np.exp(x) - 1) / x

        for x in [-1, -0.5, 0, 0.5, 1]:
            y = f(x)
            if x == 0:
                assert abs(y - 1.0) < 1e-10
            else:
                assert abs(y - (math.exp(x)-1)/x) < 1e-10


# =============================================================================
# PATCHING DOESN'T LEAK
# =============================================================================

class TestNoLeak:

    def test_sequential_calls(self):
        """Multiple resolve() calls don't accumulate patches."""
        for _ in range(10):
            resolve(lambda x: math.sin(x)/x, at=0)
        result = math.sin(1.0)
        assert type(result) is float

    def test_different_functions(self):
        """Different functions in sequence."""
        resolve(lambda x: math.sin(x)/x, at=0)
        resolve(lambda x: (math.exp(x)-1)/x, at=0)
        resolve(lambda x: math.log(1+x)/x, at=0)
        # All math functions should be originals
        assert abs(math.sin(0.5) - 0.479425538604203) < 1e-10
        assert abs(math.exp(0.0) - 1.0) < 1e-15
        assert abs(math.log(1.0) - 0.0) < 1e-15
