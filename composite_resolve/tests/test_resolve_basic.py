"""Basic smoke tests for composite-resolve package."""

import math

import pytest

from composite_resolve import (
    limit, resolve, evaluate, taylor, classify, residue,
    LimitDoesNotExistError, LimitDivergesError,
    SingularityError, CompositionError,
    Regular, Removable, Pole, Essential,
)
from composite_resolve.math import sin, cos, exp, log, sqrt, atan, tanh


# --- Basic Limits ---

@pytest.mark.parametrize("name,f,at,expected", [
    ("sin(x)/x",         lambda x: sin(x)/x,               0,  1.0),
    ("(exp(x)-1)/x",     lambda x: (exp(x)-1)/x,           0,  1.0),
    ("(1-cos(x))/x²",    lambda x: (1-cos(x))/x**2,        0,  0.5),
    ("(x²-1)/(x-1)",     lambda x: (x**2-1)/(x-1),         1,  2.0),
    ("(x³-8)/(x-2)",     lambda x: (x**3-8)/(x-2),         2,  12.0),
    ("log(1+x)/x",       lambda x: log(1+x)/x,             0,  1.0),
    ("(sqrt(1+x)-1)/x",  lambda x: (sqrt(1+x)-1)/x,        0,  0.5),
    ("x/sin(x)",         lambda x: x/sin(x),               0,  1.0),
])
def test_basic_limits(name, f, at, expected):
    result = limit(f, to=at)
    assert abs(result - expected) < 1e-6, (
        f"{name} at {at}: got {result}, expected {expected}"
    )


# --- math.sin patching ---

class TestMathPatching:
    def test_math_sin_inside_limit(self):
        r = limit(lambda x: math.sin(x) / x, to=0)
        assert abs(r - 1.0) < 1e-6

    def test_math_exp_inside_limit(self):
        r = limit(lambda x: math.exp(x) - 1 - x, to=0)
        assert abs(r) < 1e-6

    def test_math_sin_restored_after_limit(self):
        _ = limit(lambda x: math.sin(x) / x, to=0)
        r = math.sin(0.0)
        assert r == 0.0
        assert type(r) is float


# --- Indeterminate Forms ---

@pytest.mark.parametrize("name,f,at,direction,expected", [
    ("x*log(x) [0×∞]",     lambda x: x*log(x),        0,  "+",  0.0),
    ("x^x [0⁰]",           lambda x: x**x,             0,  "+",  1.0),
    ("(1+x)^(1/x) [1^∞]",  lambda x: (1+x)**(1/x),    0,  None, math.e),
    ("x*sin(1/x) [0×osc]", lambda x: x*sin(1/x),       0,  None, 0.0),
])
def test_indeterminate_forms(name, f, at, direction, expected):
    kw = {"dir": direction} if direction else {}
    result = limit(f, to=at, **kw)
    assert abs(result - expected) < 1e-4, (
        f"{name}: got {result}, expected {expected}"
    )


# --- One-Sided Limits ---

class TestOneSided:
    def test_1_over_x_from_right_diverges(self):
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x, to=0, dir="+")
        assert exc_info.value.value == math.inf

    def test_1_over_x_from_left_diverges(self):
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: 1/x, to=0, dir="-")
        assert exc_info.value.value == -math.inf

    def test_1_over_x_both_does_not_exist(self):
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: 1/x, to=0, dir="both")

    def test_atan_1_over_x_from_right(self):
        r = limit(lambda x: atan(1/x), to=0, dir="+")
        assert abs(r - math.pi/2) < 1e-6


# --- Limits at Infinity ---

@pytest.mark.parametrize("name,f,at,expected", [
    ("sin(x)/x → ∞",           lambda x: sin(x)/x,            math.inf, 0.0),
    ("(x²+1)/(x²-1) → ∞",     lambda x: (x**2+1)/(x**2-1),   math.inf, 1.0),
    ("x*sin(1/x) → ∞",         lambda x: x*sin(1/x),          math.inf, 1.0),
    ("atan(x) → ∞",            lambda x: atan(x),             math.inf, math.pi/2),
])
def test_infinity(name, f, at, expected):
    result = limit(f, to=at)
    assert abs(result - expected) < 1e-4, (
        f"{name}: got {result}, expected {expected}"
    )


# --- Taylor Coefficients ---

class TestTaylor:
    def test_exp_at_0(self):
        c = taylor(lambda x: exp(x), at=0, order=4)
        expected = [1, 1, 0.5, 1/6, 1/24]
        for i in range(5):
            assert abs(c[i] - expected[i]) < 1e-10

    def test_sin_at_0(self):
        c = taylor(lambda x: sin(x), at=0, order=4)
        expected = [0, 1, 0, -1/6, 0]
        for i in range(5):
            assert abs(c[i] - expected[i]) < 1e-10

    def test_geometric_series_at_0(self):
        c = taylor(lambda x: 1/(1-x), at=0, order=4)
        expected = [1, 1, 1, 1, 1]
        for i in range(5):
            assert abs(c[i] - expected[i]) < 1e-10


# --- Singularity Classification ---

class TestClassify:
    def test_sin_x_over_x_removable(self):
        info = classify(lambda x: sin(x)/x, at=0)
        assert isinstance(info, Removable)
        assert abs(info.value - 1.0) < 1e-6

    def test_1_over_x_pole_order_1(self):
        info = classify(lambda x: 1/x, at=0)
        assert isinstance(info, Pole)
        assert info.order == 1

    def test_1_over_x2_pole_order_2(self):
        info = classify(lambda x: 1/x**2, at=0)
        assert isinstance(info, Pole)
        assert info.order == 2

    def test_exp_regular(self):
        info = classify(lambda x: exp(x), at=0)
        assert isinstance(info, Regular)
        assert abs(info.value - 1.0) < 1e-6

    def test_x2_minus_4_over_x_minus_2_removable(self):
        info = classify(lambda x: (x**2-4)/(x-2), at=2)
        assert isinstance(info, Removable)
        assert abs(info.value - 4.0) < 1e-6


# --- resolve() alias ---

def test_resolve_alias():
    r = resolve(lambda x: sin(x)/x, at=0)
    assert abs(r - 1.0) < 1e-10


# --- evaluate() ---

class TestEvaluate:
    def test_evaluate_removable(self):
        r = evaluate(lambda x: (x**2-1)/(x-1), at=1)
        assert abs(r - 2.0) < 1e-6

    def test_evaluate_pole_raises(self):
        with pytest.raises(SingularityError):
            evaluate(lambda x: 1/x, at=0)
