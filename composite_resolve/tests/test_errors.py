# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""T8: Error handling."""
import math

import pytest

from composite_resolve import (
    limit, evaluate,
    LimitDoesNotExistError, LimitDivergesError,
    SingularityError, CompositionError,
)
from composite_resolve.math import sin


def test_T8_01_sin_1_over_x_oscillates():
    """T8.01: sin(1/x) → 0: LimitDoesNotExistError"""
    with pytest.raises(LimitDoesNotExistError):
        limit(lambda x: sin(1/x), to=0)


def test_T8_02_1_over_x_both_sides():
    """T8.02: 1/x → 0 (both): LimitDoesNotExistError"""
    with pytest.raises(LimitDoesNotExistError):
        limit(lambda x: 1/x, to=0)


def test_T8_03_1_over_x_from_right_diverges():
    """T8.03: 1/x → 0+: LimitDivergesError(value=inf)"""
    with pytest.raises(LimitDivergesError) as exc_info:
        limit(lambda x: 1/x, to=0, dir="+")
    assert exc_info.value.value == math.inf


def test_T8_04_evaluate_pole():
    """T8.04: evaluate(1/x, at=0): SingularityError"""
    with pytest.raises(SingularityError):
        evaluate(lambda x: 1/x, at=0)


def test_T8_05_non_composable():
    """T8.05: Non-composable function: CompositionError"""
    with pytest.raises(CompositionError):
        limit(lambda x: x + "hello", to=0)


def test_T8_06_limit_at_nan():
    """T8.06: limit at NaN: ValueError"""
    with pytest.raises(ValueError):
        limit(lambda x: x, to=float('nan'))
