# composite-resolve - Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later - see LICENSE
"""Tests for the verify() automatic singularity detection."""
import math
import pytest

from composite_resolve import verify, safe


# -- Single-variable tests --

def test_sinc_unhandled():
    report = verify(lambda x: math.sin(x) / x, var_range=(-10, 10))
    assert not report.passed
    assert len(report.unhandled) >= 1
    sing = [s for s in report.singularities if abs(s.point) < 0.01]
    assert len(sing) >= 1
    assert sing[0].correct_value == pytest.approx(1.0)


def test_sinc_safe_handled():
    @safe
    def sinc(x):
        return math.sin(x) / x

    report = verify(sinc, var_range=(-10, 10))
    assert report.passed
    assert len(report.unhandled) == 0


def test_entropy_unhandled():
    report = verify(lambda p: -p * math.log(p), var_range=(0, 1))
    assert not report.passed
    sing = [s for s in report.unhandled if abs(s.point) < 0.01]
    assert len(sing) >= 1


def test_entropy_handled():
    @safe
    def entropy(p):
        return -p * math.log(p)

    report = verify(entropy, var_range=(0, 1))
    assert report.passed


def test_log_ratio_unhandled():
    report = verify(lambda x: math.log(1 + x) / x, var_range=(-0.5, 10))
    assert not report.passed
    sing = [s for s in report.unhandled if abs(s.point) < 0.01]
    assert len(sing) >= 1
    assert sing[0].correct_value == pytest.approx(1.0)


def test_exp_decay_unhandled():
    report = verify(lambda x: (math.exp(x) - 1) / x, var_range=(-5, 5))
    assert not report.passed
    sing = [s for s in report.unhandled if abs(s.point) < 0.01]
    assert len(sing) >= 1
    assert sing[0].correct_value == pytest.approx(1.0)


def test_no_singularity():
    report = verify(lambda x: x ** 2 + 1, var_range=(-10, 10))
    assert report.passed
    assert len(report.singularities) == 0


def test_report_bool():
    good = verify(lambda x: x ** 2, var_range=(-1, 1))
    assert bool(good) is True

    bad = verify(lambda x: 1 / x, var_range=(-1, 1))
    assert bool(bad) is False


def test_report_str():
    report = verify(lambda x: math.sin(x) / x, var_range=(-1, 1))
    s = str(report)
    assert "FAIL" in s
    assert "UNHANDLED" in s


def test_fixable():
    report = verify(lambda x: math.sin(x) / x, var_range=(-1, 1))
    assert len(report.fixable) >= 1
    for s in report.fixable:
        assert s.confidence == "algebraic"
        assert not s.handled


def test_pole_detected():
    report = verify(lambda x: 1 / x, var_range=(-1, 1))
    assert not report.passed
    sing = [s for s in report.singularities if abs(s.point) < 0.01]
    assert len(sing) >= 1


def test_manually_handled():
    def f(x):
        if x == 0:
            return 1.0
        return math.sin(x) / x

    report = verify(f, var_range=(-1, 1))
    assert report.passed


# -- Multi-variable tests --

def test_multi_scan_one():
    report = verify(
        lambda x, y: (x ** 2 - y ** 2) / (x - y),
        var_ranges={"x": (-5, 5), "y": (-5, 5)},
        scan="x",
        fixed={"y": 1.0},
    )
    assert not report.passed
    sing = [s for s in report.unhandled if abs(s.point - 1.0) < 0.01]
    assert len(sing) >= 1
    assert sing[0].correct_value == pytest.approx(2.0)


def test_multi_scan_all():
    report = verify(
        lambda x, y: (x ** 2 - y ** 2) / (x - y),
        var_ranges={"x": (-5, 5), "y": (-5, 5)},
        scan="all",
    )
    assert not report.passed
    assert len(report.singularities) >= 1


def test_multi_no_singularity():
    report = verify(
        lambda x, y: x ** 2 + y ** 2,
        var_ranges={"x": (-1, 1), "y": (-1, 1)},
        scan="all",
    )
    assert report.passed


# -- CI gate pattern --

FUNCTIONS = [
    ("x^2", lambda x: x ** 2, (-10, 10)),
    ("x+1", lambda x: x + 1, (-10, 10)),
]


@pytest.mark.parametrize("name,fn,domain", FUNCTIONS)
def test_ci_gate(name, fn, domain):
    report = verify(fn, var_range=domain)
    assert not report.unhandled, f"{name}:\n{report}"
