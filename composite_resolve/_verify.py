# composite-resolve - Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later - see LICENSE
"""Automatic singularity detection and boundary verification.

Scans a callable's domain, evaluates each sample point with both plain
float and composite arithmetic to detect singularities structurally,
then computes correct limit values and returns a report.
"""
from __future__ import annotations

import math
import warnings
import time
from dataclasses import dataclass, field

from composite_resolve._core import Composite, _seeded, _has_positive_dims
from composite_resolve._limit import limit
from composite_resolve._compat import patch_math, restore_math
from composite_resolve._errors import (
    LimitDivergesError, LimitDoesNotExistError, LimitUndecidableError,
    CompositionError,
)


@dataclass
class Singularity:
    """A detected singularity in the scanned domain."""
    point: float
    kind: str
    value: object
    left_value: object = None
    right_value: object = None
    handled: bool = False
    behavior: str = ""

    @property
    def fixable(self):
        return self.kind == "removable" and not self.handled

    def __str__(self):
        status = "HANDLED" if self.handled else "UNHANDLED"
        if self.kind == "removable":
            s = f"x={self.point}: removable [{status}] value={self.value}"
            if not self.handled:
                s += f"\n      {self.behavior}"
                s += f"\n      fix: return {self.value} at x={self.point}"
        elif self.kind == "jump":
            s = f"x={self.point}: jump [{status}] left={self.left_value} right={self.right_value}"
            if not self.handled:
                s += f"\n      {self.behavior}"
        elif self.kind == "pole":
            s = f"x={self.point}: pole [{status}]"
            if not self.handled:
                s += f"\n      {self.behavior}"
        else:
            s = f"x={self.point}: {self.kind} [{status}]"
            if not self.handled:
                s += f"\n      {self.behavior}"
        return s


@dataclass
class VerifyReport:
    function_name: str
    domain: tuple
    points_sampled: int
    singularities: list = field(default_factory=list)
    scan_time: float = 0.0

    @property
    def unhandled(self):
        return [s for s in self.singularities if not s.handled]

    @property
    def fixable(self):
        return [s for s in self.singularities if s.fixable]

    @property
    def passed(self):
        return len(self.unhandled) == 0

    def __str__(self):
        lines = [f"verify({self.function_name}) over {self.domain}"]
        lines.append(f"  {self.points_sampled} points sampled in {self.scan_time:.3f}s")
        lines.append(f"  {len(self.singularities)} singularity(ies) found")
        for s in self.singularities:
            for line in str(s).split("\n"):
                lines.append(f"    {line}")
        n_un = len(self.unhandled)
        if n_un == 0:
            lines.append("  PASS: all singularities handled")
        else:
            lines.append(f"  FAIL: {n_un} unhandled singularity(ies)")
        return "\n".join(lines)

    def __bool__(self):
        return self.passed


def _snap_candidates(x):
    """Yield x itself, then nearby clean floats (rounded to N decimals)."""
    yield x
    for decimals in range(0, 16):
        r = round(x, decimals)
        if r != x:
            yield r


def _build_grid(lo, hi, n, extra=None):
    """Build sample grid: n uniform points plus critical points."""
    sample_set = set()
    for i in range(n):
        x = lo + i * (hi - lo) / (n - 1) if n > 1 else lo
        sample_set.add(x)
    if extra:
        for x in extra:
            if lo <= x <= hi:
                sample_set.add(float(x))

    if lo <= 0 <= hi:
        sample_set.add(0.0)
    int_lo = math.ceil(lo)
    int_hi = math.floor(hi)
    if int_hi - int_lo <= 100:
        for k in range(int_lo, int_hi + 1):
            sample_set.add(float(k))
    else:
        step = 10 ** max(0, int(math.log10(int_hi - int_lo + 1)) - 1)
        k = int(math.ceil(lo / step) * step)
        while k <= hi:
            sample_set.add(float(k))
            k += step
    for c in (math.e, math.pi, math.pi / 2, math.pi / 4,
              math.pi * 3 / 2, math.pi * 2, math.sqrt(2), math.sqrt(3)):
        if lo <= c <= hi:
            sample_set.add(c)
        if lo <= -c <= hi:
            sample_set.add(-c)

    return sorted(sample_set)


def _scan_domain(f, lo, hi, n, extra=None):
    """Evaluate f at grid points using composite arithmetic.

    Two detection methods:
    1. Hook: intercepts division/log/sqrt during composite eval, uses Newton's
       method to find where the denominator/argument crosses zero.
    2. Disagreement: compares f(float(x)) against f(composite(x)).st().
       If they differ, the point has a discontinuity or near-singularity.

    Returns (set of singular x values, sample count).
    """
    samples = _build_grid(lo, hi, n, extra)
    step = (hi - lo) / max(n - 1, 1)
    singular = set()
    found_zeros = set()

    patch_math()
    try:
        for x_val in samples:
            zeros = []

            def hook(op, arg):
                d0 = arg._d.get(0, 0.0)
                d1 = arg._d.get(-1, 0.0)
                if d1 == 0:
                    return
                if op == "floor":
                    nearest_int = round(d0)
                    x_disc = x_val - (d0 - nearest_int) / d1
                    if lo <= x_disc <= hi and abs(x_disc - x_val) <= step * 1.5:
                        zeros.append(x_disc)
                else:
                    x_zero = x_val - d0 / d1
                    if lo <= x_zero <= hi and abs(x_zero - x_val) <= step * 1.5:
                        zeros.append(x_zero)

            Composite._singularity_hook = hook
            composite_result = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    composite_result = f(_seeded(x_val))
            except Exception:
                pass
            Composite._singularity_hook = None
            found_zeros.update(zeros)

            # Also evaluate from the left: R(x) - ZERO
            left_result = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from composite_resolve._core import R, ZERO
                    x_left = -ZERO if x_val == 0 else R(x_val) - ZERO
                    left_result = f(x_left)
            except Exception:
                pass

            # Get dim-0 from both sides
            right_d0 = None
            left_d0 = None
            if isinstance(composite_result, Composite):
                right_d0 = composite_result._d.get(0)
            if isinstance(left_result, Composite):
                left_d0 = left_result._d.get(0)

            # Detect: float fails, or composite sides disagree, or
            # composite disagrees with float
            try:
                float_val = f(float(x_val))
                if not isinstance(float_val, (int, float)):
                    float_val = None
                elif math.isnan(float_val) or math.isinf(float_val):
                    singular.add(x_val)
                    float_val = None
            except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                singular.add(x_val)
                float_val = None

            if float_val is not None:
                # Check left vs right disagreement
                if (right_d0 is not None and left_d0 is not None
                        and math.isfinite(right_d0) and math.isfinite(left_d0)):
                    if abs(right_d0 - left_d0) > 0.01 * (abs(right_d0) + abs(left_d0) + 1e-10):
                        singular.add(x_val)
                # Check composite vs float disagreement
                if right_d0 is not None and math.isfinite(right_d0):
                    if abs(float_val - right_d0) > 0.01 * (abs(right_d0) + 1e-10):
                        singular.add(x_val)
    finally:
        Composite._singularity_hook = None
        restore_math()

    # Snap each Newton estimate to the nearest clean float
    for x_zero in found_zeros:
        best = x_zero
        for x_try in _snap_candidates(x_zero):
            if lo <= x_try <= hi:
                try:
                    f(float(x_try))
                except (ZeroDivisionError, ValueError, OverflowError, TypeError):
                    best = x_try
                    break
        singular.add(best)

    return singular, len(samples)


def _cluster(points):
    """Cluster nearby points."""
    if not points:
        return []
    sorted_p = sorted(points)
    clustered = [sorted_p[0]]
    for x in sorted_p[1:]:
        if abs(x - clustered[-1]) > 1e-8 * (abs(x) + 1.0):
            clustered.append(x)
    return clustered


def _analyze_point(f, point):
    """Analyze a singularity at a given point. Returns a single Singularity."""
    left_val = None
    right_val = None
    left_conf = None
    right_conf = None

    for direction in ("+", "-"):
        try:
            val = limit(f, to=point, dir=direction)
            if direction == "+":
                right_val, right_conf = val, "algebraic"
            else:
                left_val, left_conf = val, "algebraic"
        except LimitDivergesError as e:
            if direction == "+":
                right_val, right_conf = e.value, "divergent"
            else:
                left_val, left_conf = e.value, "divergent"
        except (LimitDoesNotExistError, LimitUndecidableError):
            if direction == "+":
                right_conf = "undecidable"
            else:
                left_conf = "undecidable"
        except Exception:
            if direction == "+":
                right_conf = "error"
            else:
                left_conf = "error"

    if left_conf == "algebraic" and right_conf == "algebraic":
        if _values_equal(left_val, right_val):
            kind = "removable"
            value = left_val
        else:
            kind = "jump"
            value = None
    elif left_conf == "divergent" and right_conf == "divergent":
        kind = "pole"
        value = None
    elif left_conf == "divergent" and right_conf == "algebraic":
        kind = "jump"
        value = None
    elif left_conf == "algebraic" and right_conf == "divergent":
        kind = "jump"
        value = None
    elif left_conf == "algebraic":
        kind = "removable"
        value = left_val
    elif right_conf == "algebraic":
        kind = "removable"
        value = right_val
    elif left_conf == "divergent" or right_conf == "divergent":
        kind = "pole"
        value = None
    else:
        kind = "undecidable"
        value = None

    behavior = _check_handling(f, point, value)
    handled = (behavior == "handled_correctly")

    return Singularity(
        point=point,
        kind=kind,
        value=value,
        left_value=left_val,
        right_value=right_val,
        handled=handled,
        behavior=behavior,
    )


def _values_equal(a, b):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) < 1e-10 * (abs(a) + abs(b) + 1e-100)


def _check_handling(f, point, correct_value):
    """Does the function already handle this singularity correctly?"""
    try:
        actual = f(float(point))
        if not isinstance(actual, (int, float)):
            return "returns_non_numeric"
        if math.isnan(actual):
            return "returns_NaN"
        if math.isinf(actual):
            if correct_value is not None and math.isinf(correct_value):
                if (actual > 0) == (correct_value > 0):
                    return "handled_correctly"
            return "returns_inf"
        if correct_value is not None:
            if math.isinf(correct_value):
                return "returns_finite_expected_divergent"
            if math.isfinite(correct_value):
                if abs(actual - correct_value) < 1e-6 * (abs(correct_value) + 1e-10):
                    return "handled_correctly"
                return f"returns_wrong_value({actual:.6g})"
        return "handled_correctly"
    except ZeroDivisionError:
        return "raises_ZeroDivisionError"
    except ValueError:
        return "raises_ValueError"
    except OverflowError:
        return "raises_OverflowError"
    except TypeError:
        return "raises_TypeError"
    except Exception as e:
        return f"raises_{type(e).__name__}"


def verify(f, var_range, points=1000, check_points=None):
    """Scan a function's domain for singularities and verify boundary behavior.

    Evaluates f at evenly spaced points plus critical points (0, integers,
    e, pi/2, etc.). Points where f raises, returns NaN, or returns inf are
    flagged as singularities. Poles between grid points are found via
    bisection. Each singularity is analyzed with composite arithmetic.

    Pass check_points to test specific x values you suspect may be singular:

        verify(f, var_range=(-1, 1), check_points=[1e-6, 0.7])

    Usage:
        report = verify(f, var_range=(0, 1))
        assert report  # passes if all singularities are handled

    Returns a VerifyReport with:
        .singularities  - list of Singularity objects found
        .unhandled      - subset that the function doesn't handle
        .fixable        - subset that are removable and can be fixed
        .passed         - True if no unhandled singularities
    """
    t0 = time.perf_counter()
    fname = getattr(f, "__name__", repr(f))

    lo, hi = var_range
    singular_points, n_sampled = _scan_domain(f, lo, hi, points, check_points)
    failure_points = _cluster(singular_points)

    singularities = []
    for pt in failure_points:
        singularities.append(_analyze_point(f, pt))

    return VerifyReport(
        function_name=fname,
        domain=var_range,
        points_sampled=n_sampled,
        singularities=singularities,
        scan_time=time.perf_counter() - t0,
    )
