# composite-resolve - Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later - see LICENSE
"""Automatic singularity detection and boundary verification.

Scans a callable's domain, finds where it breaks, classifies each
singularity, computes the correct limit value, and returns a structured
report. Usable standalone during development and as a CI gate via pytest.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from composite_resolve._limit import limit
from composite_resolve._classify import classify, Regular, Removable, Pole, Essential
from composite_resolve._errors import (
    LimitDivergesError, LimitDoesNotExistError, LimitUndecidableError,
)


@dataclass
class SingularityReport:
    point: float
    direction: str
    classification: object
    correct_value: object
    confidence: str
    function_behavior: str

    @property
    def handled(self):
        return self.function_behavior == "handled_correctly"

    @property
    def fixable(self):
        return self.confidence == "algebraic" and not self.handled

    def __str__(self):
        status = "HANDLED" if self.handled else "UNHANDLED"
        cls_name = type(self.classification).__name__ if not isinstance(self.classification, str) else self.classification
        if self.correct_value is not None:
            val = f"value={self.correct_value:.6g}"
        else:
            val = self.confidence
        s = f"x={self.point} dir={self.direction}: {cls_name} [{status}] {val}"
        if not self.handled:
            s += f"\n      function {self.function_behavior}"
            if self.fixable:
                s += f"\n      fix: wrap with @safe or return {self.correct_value} at x={self.point}"
        return s


@dataclass
class VerifyReport:
    function_name: str
    domain: tuple
    points_sampled: int
    singularities: list[SingularityReport] = field(default_factory=list)
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


def _scan_domain(f, lo, hi, n):
    """Sample f at n points in [lo, hi], plus critical points (0, integers).
    Returns list of (x, status, value_or_error) sorted by x."""
    sample_set = set()
    for i in range(n):
        x = lo + i * (hi - lo) / (n - 1) if n > 1 else lo
        sample_set.add(x)

    if lo <= 0 <= hi:
        sample_set.add(0.0)
    int_lo = math.ceil(lo)
    int_hi = math.floor(hi)
    if int_hi - int_lo <= 100:
        for k in range(int_lo, int_hi + 1):
            sample_set.add(float(k))

    samples = sorted(sample_set)
    results = []
    for x in samples:
        try:
            y = f(float(x))
            if isinstance(y, (int, float)):
                if math.isnan(y):
                    results.append((x, "nan", y))
                elif math.isinf(y):
                    results.append((x, "inf", y))
                else:
                    results.append((x, "ok", float(y)))
            else:
                results.append((x, "ok", float(y)))
        except (ZeroDivisionError, ValueError, OverflowError, TypeError) as e:
            results.append((x, "raises", e))
    return results


def _find_failure_points(scan_results, lo, hi):
    """Extract failure points: where f raises, returns NaN/inf,
    or has a discontinuity. Also check domain boundaries."""
    failures = set()

    for x, status, _ in scan_results:
        if status != "ok":
            failures.add(x)

    # Detect discontinuities: flag only when the local derivative is
    # vastly larger than the global average, avoiding false positives
    # at smooth zero-crossings.
    ok_vals = [(x, v) for x, s, v in scan_results if s == "ok"]
    if len(ok_vals) >= 3:
        all_y = [v for _, v in ok_vals]
        y_range = max(all_y) - min(all_y) if all_y else 0.0
        domain = hi - lo + 1e-100
        if y_range > 0:
            avg_deriv = y_range / domain
            for i in range(1, len(ok_vals)):
                x0, y0 = ok_vals[i - 1]
                x1, y1 = ok_vals[i]
                dy = abs(y1 - y0)
                dx = abs(x1 - x0) + 1e-100
                local_deriv = dy / dx
                if local_deriv > 200 * avg_deriv and dy > 0.1 * y_range:
                    midpoint = (x0 + x1) / 2
                    failures.add(midpoint)

    # Cluster nearby points (within 1e-8 relative)
    if not failures:
        return []
    sorted_f = sorted(failures)
    clustered = [sorted_f[0]]
    for x in sorted_f[1:]:
        if abs(x - clustered[-1]) > 1e-8 * (abs(x) + 1.0):
            clustered.append(x)
        else:
            pass
    return clustered


def _analyze_singularity(f, point):
    """Classify and compute limit at a failure point.
    Returns a SingularityReport for each relevant direction."""
    reports = []

    for direction in ("+", "-"):
        cls = None
        try:
            cls = classify(f, at=point, dir=direction)
        except Exception:
            cls = "unclassifiable"

        correct_value = None
        confidence = None
        try:
            correct_value = limit(f, to=point, dir=direction)
            confidence = "algebraic"
        except LimitDivergesError as e:
            correct_value = e.value
            confidence = "divergent"
        except LimitDoesNotExistError:
            confidence = "does_not_exist"
        except LimitUndecidableError:
            confidence = "undecidable"
        except Exception:
            confidence = "error"

        behavior = _check_handling(f, point, correct_value)

        reports.append(SingularityReport(
            point=point,
            direction=direction,
            classification=cls,
            correct_value=correct_value,
            confidence=confidence,
            function_behavior=behavior,
        ))

    # If both directions agree and are handled the same way, merge into one
    if (len(reports) == 2
            and reports[0].function_behavior == reports[1].function_behavior
            and reports[0].confidence == reports[1].confidence
            and _values_equal(reports[0].correct_value, reports[1].correct_value)):
        reports = [SingularityReport(
            point=point,
            direction="both",
            classification=reports[0].classification,
            correct_value=reports[0].correct_value,
            confidence=reports[0].confidence,
            function_behavior=reports[0].function_behavior,
        )]

    return reports


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
                return f"returns_finite({actual:.6g})_expected_divergent"
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


def verify(f, var_range=None, var_ranges=None, scan=None, fixed=None, points=1000):
    """Scan a function's domain for singularities and verify boundary behavior.

    Single variable:
        report = verify(f, var_range=(0, 1))

    Multi-variable (scan one, fix others):
        report = verify(f, var_ranges={"x": (0,1), "y": (0,1)},
                        scan="x", fixed={"y": 0.5})

    Multi-variable (scan all, one at a time):
        report = verify(f, var_ranges={"x": (0,1), "y": (0,1)}, scan="all")

    Returns a VerifyReport. Use `assert report` or `assert not report.unhandled`
    in tests for CI gating.
    """
    t0 = time.perf_counter()
    fname = getattr(f, "__name__", repr(f))

    # Single variable
    if var_range is not None:
        lo, hi = var_range
        scan_results = _scan_domain(f, lo, hi, points)
        failure_points = _find_failure_points(scan_results, lo, hi)

        all_reports = []
        for pt in failure_points:
            all_reports.extend(_analyze_singularity(f, pt))

        return VerifyReport(
            function_name=fname,
            domain=var_range,
            points_sampled=len(scan_results),
            singularities=all_reports,
            scan_time=time.perf_counter() - t0,
        )

    # Multi-variable
    if var_ranges is not None:
        if scan == "all":
            # Scan each variable with others at midpoint
            all_reports = []
            for var_name, (lo, hi) in var_ranges.items():
                midpoints = {}
                for other_name, (other_lo, other_hi) in var_ranges.items():
                    if other_name != var_name:
                        midpoints[other_name] = (other_lo + other_hi) / 2
                actual_fixed = dict(midpoints)
                if fixed:
                    actual_fixed.update({k: v for k, v in fixed.items() if k != var_name})

                wrapped = _make_single_var(f, var_name, list(var_ranges.keys()), actual_fixed)
                scan_results = _scan_domain(wrapped, lo, hi, points)
                failure_points = _find_failure_points(scan_results, lo, hi)
                for pt in failure_points:
                    reports = _analyze_singularity(wrapped, pt)
                    for r in reports:
                        r.direction = f"{var_name}={r.direction}"
                    all_reports.extend(reports)

            return VerifyReport(
                function_name=fname,
                domain=var_ranges,
                points_sampled=points * len(var_ranges),
                singularities=all_reports,
                scan_time=time.perf_counter() - t0,
            )

        elif scan is not None:
            # Scan one variable
            lo, hi = var_ranges[scan]
            actual_fixed = dict(fixed) if fixed else {}
            for name, (vlo, vhi) in var_ranges.items():
                if name != scan and name not in actual_fixed:
                    actual_fixed[name] = (vlo + vhi) / 2

            wrapped = _make_single_var(f, scan, list(var_ranges.keys()), actual_fixed)
            scan_results = _scan_domain(wrapped, lo, hi, points)
            failure_points = _find_failure_points(scan_results, lo, hi)

            all_reports = []
            for pt in failure_points:
                reports = _analyze_singularity(wrapped, pt)
                for r in reports:
                    r.direction = f"{scan}={r.direction}"
                all_reports.extend(reports)

            return VerifyReport(
                function_name=fname,
                domain=var_ranges,
                points_sampled=points,
                singularities=all_reports,
                scan_time=time.perf_counter() - t0,
            )

    raise ValueError("Provide var_range for single-variable or var_ranges for multi-variable")


def _make_single_var(f, var_name, all_var_names, fixed_values):
    """Create a single-variable wrapper for a multi-variable function."""
    def wrapped(x):
        args = []
        for name in all_var_names:
            if name == var_name:
                args.append(x)
            else:
                args.append(fixed_values[name])
        return f(*args)
    wrapped.__name__ = f"{getattr(f, '__name__', 'f')}({var_name})"
    return wrapped
