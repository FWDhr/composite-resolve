# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T5: Directional (one-sided) limits."""
import math

import pytest

from composite_resolve import limit, LimitDoesNotExistError, LimitDivergesError
from composite_resolve.math import sin, cos, exp, atan


class TestOneSidedDiverges:
    def test_T5_01_1_over_x_from_right(self):
        """T5.01: 1/x → 0+: +∞"""
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x, to=0, dir="+")
        assert exc_info.value.value == math.inf

    def test_T5_02_1_over_x_from_left(self):
        """T5.02: 1/x → 0-: -∞"""
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x, to=0, dir="-")
        assert exc_info.value.value == -math.inf

    def test_T5_03_1_over_x_both(self):
        """T5.03: 1/x → 0 both: LimitDoesNotExistError"""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: 1/x, to=0, dir="both")

    def test_T5_07_abs_x_over_x_both(self):
        """T5.07: |x|/x → 0 both: LimitDoesNotExistError"""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: abs(x)/x, to=0, dir="both")

    def test_T5_08_exp_1_over_x_from_right(self):
        """T5.08: exp(1/x) → 0+: +∞"""
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: exp(1/x), to=0, dir="+")
        assert exc_info.value.value == math.inf


@pytest.mark.parametrize("name,f,at,direction,expected", [
    ("T5.04 x^x → 0+",
     lambda x: x**x,             0, "+", 1.0),

    ("T5.05 |x|/x → 0+",
     lambda x: abs(x)/x,         0, "+", 1.0),

    ("T5.06 |x|/x → 0-",
     lambda x: abs(x)/x,         0, "-", -1.0),

    ("T5.09 exp(1/x) → 0-",
     lambda x: exp(1/x),         0, "-", 0.0),
])
def test_onesided_value(name, f, at, direction, expected):
    result = limit(f, to=at, dir=direction)
    assert abs(result - expected) < 1e-4, (
        f"{name}: got {result}, expected {expected}"
    )
