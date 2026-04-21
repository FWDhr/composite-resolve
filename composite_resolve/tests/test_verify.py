# composite-resolve - Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later - see LICENSE
"""Tests for the verify() automatic singularity detection."""
import math
import pytest

from composite_resolve import verify, safe


# -- Finds poles at arbitrary points --

def test_pole_at_non_integer():
    report = verify(lambda x: 1 / (x - 0.7), var_range=(-1, 2))
    assert not report.passed
    s = [s for s in report.singularities if abs(s.point - 0.7) < 1e-6]
    assert len(s) == 1
    assert s[0].kind == "pole"


def test_pole_at_half_integer():
    report = verify(lambda x: 1 / (x - 2.5), var_range=(-5, 5))
    assert not report.passed
    assert any(abs(s.point - 2.5) < 1e-6 for s in report.singularities)


def test_two_poles():
    report = verify(lambda x: 1 / ((x - 1) * (x - 3.5)), var_range=(-5, 5))
    assert not report.passed
    points = [s.point for s in report.singularities]
    assert any(abs(p - 1.0) < 1e-6 for p in points)
    assert any(abs(p - 3.5) < 1e-6 for p in points)


def test_pole_at_integer():
    report = verify(lambda x: (4 - x) / (1 - x), var_range=(-10, 10))
    assert not report.passed
    s = [s for s in report.singularities if abs(s.point - 1.0) < 1e-6]
    assert len(s) == 1
    assert s[0].kind == "pole"


def test_tan_no_computational_singularity():
    """tan(math.pi/2) is finite because math.pi/2 != exact pi/2."""
    report = verify(math.tan, var_range=(-5, 5))
    assert report.passed


def test_pole_at_pi_half():
    """1/cos(x) has a pole where cos(x)=0, but float pi/2 misses it."""
    report = verify(lambda x: 1 / (x - math.pi / 2), var_range=(0, 5))
    assert not report.passed
    assert any(abs(s.point - math.pi / 2) < 1e-10 for s in report.singularities)


# -- Finds removable singularities with correct value --

def test_sinc():
    report = verify(lambda x: math.sin(x) / x, var_range=(-10, 10))
    assert not report.passed
    s = [s for s in report.singularities if abs(s.point) < 1e-6]
    assert len(s) == 1
    assert s[0].kind == "removable"
    assert s[0].value == pytest.approx(1.0)
    assert s[0].fixable


def test_exp_decay():
    report = verify(lambda x: (math.exp(x) - 1) / x, var_range=(-5, 5))
    assert not report.passed
    s = [s for s in report.singularities if abs(s.point) < 1e-6]
    assert len(s) == 1
    assert s[0].value == pytest.approx(1.0)


def test_x_squared_over_x():
    report = verify(lambda x: x ** 2 / x, var_range=(-5, 5))
    assert not report.passed
    s = [s for s in report.singularities if abs(s.point) < 1e-6]
    assert len(s) == 1
    assert s[0].kind == "removable"
    assert s[0].value == pytest.approx(0.0, abs=1e-10)


# -- No singularities (zero false positives) --

def test_polynomial_clean():
    report = verify(lambda x: x ** 2 + 1, var_range=(-10, 10))
    assert report.passed
    assert len(report.singularities) == 0


def test_sin_clean():
    report = verify(math.sin, var_range=(-10, 10))
    assert report.passed


def test_linear_clean():
    report = verify(lambda x: 3 * x + 7, var_range=(-100, 100))
    assert report.passed


# -- Handled singularities --

def test_safe_decorator():
    @safe
    def sinc(x):
        return math.sin(x) / x

    report = verify(sinc, var_range=(-10, 10))
    assert report.passed


def test_manual_guard():
    def f(x):
        if x == 0:
            return 1.0
        return math.sin(x) / x

    report = verify(f, var_range=(-1, 1))
    assert report.passed


# -- Report interface --

def test_bool():
    assert bool(verify(lambda x: x ** 2, var_range=(-1, 1))) is True
    assert bool(verify(lambda x: 1 / x, var_range=(-1, 1))) is False


def test_str_fail():
    s = str(verify(lambda x: 1 / x, var_range=(-1, 1)))
    assert "FAIL" in s
    assert "pole" in s


def test_str_pass():
    s = str(verify(lambda x: x ** 2, var_range=(-1, 1)))
    assert "PASS" in s


def test_fixable():
    report = verify(lambda x: math.sin(x) / x, var_range=(-1, 1))
    assert len(report.fixable) == 1
    assert report.fixable[0].kind == "removable"
    assert not report.fixable[0].handled


# -- Domain boundary --

def test_entropy_at_zero():
    report = verify(lambda p: -p * math.log(p), var_range=(0, 1))
    assert not report.passed
    assert any(abs(s.point) < 0.01 for s in report.unhandled)
