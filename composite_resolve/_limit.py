# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Core limit computation API.

limit(f, to, dir)   — compute exact limit of f(x) as x → to
resolve(f, at, dir)  — alias for limit
evaluate(f, at)      — evaluate at removable singularity only
"""

import math
import warnings

from composite_resolve._core import (
    Composite, R, ZERO, INF,
    _seeded, _min_terms, _is_nothing,
)
from composite_resolve._errors import (
    LimitDoesNotExistError, LimitDivergesError,
    CompositionError, SingularityError,
)
from composite_resolve._compat import patch_math, restore_math
from composite_resolve._classify import classify, Regular, Removable


def limit(f, to=0, dir="both", truncation=20):
    """Compute the exact limit of f(x) as x → to.

    Args:
        f:          Function f(x). Can use math.sin, numpy.sin, or
                    composite_resolve.math.sin — all are handled.
        to:         Target point. float, math.inf, or -math.inf.
        dir:        Direction: "both" (default), "+" (right), "-" (left).
        truncation: Max composite order (default 20).

    Returns:
        float — the exact limit value.

    Raises:
        LimitDoesNotExistError: limit does not exist (oscillatory,
            or one-sided limits disagree when dir="both").
        LimitDivergesError: limit is ±∞ (with .value attribute).
        CompositionError: function not composable with composite arithmetic.

    Examples:
        >>> limit(lambda x: math.sin(x) / x, to=0)
        1.0
        >>> limit(lambda x: (1 + x) ** (1/x), to=0)
        2.718281828459045
        >>> limit(lambda x: x * math.sin(1/x), to=0)
        0.0
    """
    if not math.isfinite(to) and to != math.inf and to != -math.inf:
        raise ValueError(f"Invalid limit point: {to}")

    # Handle dir="both": compute left and right, compare
    if dir == "both" and math.isfinite(to):
        try:
            right = _limit_one_sided(f, to, "+", truncation)
        except LimitDivergesError as e:
            # Check if left also diverges in same direction
            try:
                left = _limit_one_sided(f, to, "-", truncation)
            except LimitDivergesError as e2:
                if e.value == e2.value:
                    raise  # both diverge same way
                raise LimitDoesNotExistError(
                    "One-sided limits diverge in different directions",
                    left_limit=e2.value, right_limit=e.value)
            except LimitDoesNotExistError:
                raise
            raise LimitDoesNotExistError(
                "Right limit diverges but left limit is finite",
                left_limit=left, right_limit=e.value)
        except LimitDoesNotExistError:
            raise

        try:
            left = _limit_one_sided(f, to, "-", truncation)
        except LimitDivergesError as e:
            raise LimitDoesNotExistError(
                "Left limit diverges but right limit is finite",
                left_limit=e.value, right_limit=right)
        except LimitDoesNotExistError:
            raise

        if abs(right - left) < 1e-10 * (abs(right) + abs(left) + 1e-100):
            return right
        raise LimitDoesNotExistError(
            f"One-sided limits disagree: left={left}, right={right}",
            left_limit=left, right_limit=right)

    # Single direction or infinity
    return _limit_one_sided(f, to, dir, truncation)


def _limit_one_sided(f, to, dir, truncation):
    """Compute a one-sided limit (or limit at infinity)."""

    # Infinity: evaluate at composite INF directly (algebraic first),
    # fall back to 1/t substitution if algebraic fails.
    if to == math.inf:
        x = INF
    elif to == -math.inf:
        x = Composite({1: -1.0})  # -INF
    elif dir == "-":
        x = -ZERO if to == 0 else R(to) - ZERO
    else:
        x = _seeded(to)

    # Set truncation for transcendentals
    old_min = _min_terms[0]
    _min_terms[0] = truncation

    try:
        # Patch math module, evaluate, restore
        patch_math()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result = f(x)
            except (TypeError, AttributeError):
                raise CompositionError(
                    "Function is not composable with Composite objects. "
                    "Ensure it uses standard arithmetic and math functions.")
            except LimitDoesNotExistError:
                # Division by nothing — propagate
                raise
            except (ValueError, ZeroDivisionError):
                # Domain error — try fallback
                _at_inf = (to == math.inf or to == -math.inf)
                if _at_inf:
                    extrap = _extrapolate_inf(f, to, truncation)
                else:
                    extrap = _extrapolate(f, to, dir, truncation)
                if extrap is not None:
                    return extrap
                raise LimitDoesNotExistError(
                    "Function has a domain error at the limit point "
                    "and extrapolation did not converge.")
    finally:
        restore_math()
        _min_terms[0] = old_min

    # Process result
    if not isinstance(result, Composite):
        try:
            return float(result)
        except (TypeError, ValueError):
            raise CompositionError(
                "Function returned a non-numeric value. "
                "Ensure it returns a number, not a list/dict/other.")

    _at_inf = (to == math.inf or to == -math.inf)

    # Nothing → indeterminate. Try fallbacks.
    if _is_nothing(result):
        return _recover(f, to, dir, truncation, _at_inf,
                        "Result is indeterminate (oscillatory). Limit does not exist.")

    st_val = result.st()

    # NaN/Inf contamination
    if not math.isfinite(st_val):
        return _recover(f, to, dir, truncation, _at_inf,
                        "Algebraic evaluation produced NaN/Inf.")

    # Positive dims → divergence (filter out near-zero noise from Taylor cancellation)
    max_pos = result.max_positive_dim()
    if max_pos is not None:
        pos_coeffs = {d: c for d, c in result.coeffs_dict().items()
                      if d > 0 and abs(c) > 1e-10}
        if not pos_coeffs:
            # All positive dims are noise — treat as clean result
            return st_val
        signs = [c > 0 for c in pos_coeffs.values()]
        if all(signs):
            raise LimitDivergesError(math.inf)
        elif not any(signs):
            raise LimitDivergesError(-math.inf)
        # Mixed — try fallbacks
        return _recover(f, to, dir, truncation, _at_inf,
                        "Result has mixed-sign positive dimensions.")

    return st_val


def _recover(f, to, dir, truncation, at_inf, error_msg):
    """Try fallback when algebraic evaluation fails.

    For infinity: numerical extrapolation at large values.
    For finite: composite extrapolation from nearby points.
    """
    if at_inf:
        extrap = _extrapolate_inf(f, to, truncation)
        if extrap is not None:
            return extrap
    else:
        extrap = _extrapolate(f, to, dir, truncation)
        if extrap is not None:
            return extrap

    raise LimitDoesNotExistError(error_msg)


def _extrapolate(f, to, dir, truncation, n_probes=6):
    """Extrapolate limit from nearby composite evaluations.

    Evaluates f at decreasing distances from the limit point.
    Uses Taylor extrapolation (exact) or value convergence (fallback).
    """
    if dir == "-":
        signs = [-1]
    elif dir == "+":
        signs = [1]
    else:
        signs = [1]

    old_min = _min_terms[0]
    _min_terms[0] = truncation

    try:
        patch_math()
        for sign in signs:
            taylor_candidates = []
            value_candidates = []

            for k in range(n_probes):
                eps = 10 ** (-(k + 2))

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        x_comp = _seeded(to + sign * eps)
                        result = f(x_comp)
                except (ValueError, ZeroDivisionError, OverflowError,
                        LimitDoesNotExistError):
                    continue

                if not isinstance(result, Composite):
                    result = R(float(result))

                st = result.st()
                if not math.isfinite(st):
                    continue

                value_candidates.append(st)

                try:
                    extrap = st + result.eval_taylor(-sign * eps)
                    if math.isfinite(extrap):
                        taylor_candidates.append(extrap)
                except (OverflowError, ValueError):
                    pass

            # Taylor converged
            if len(taylor_candidates) >= 2:
                if abs(taylor_candidates[-1] - taylor_candidates[-2]) < \
                        1e-6 * (abs(taylor_candidates[-1]) + 1e-100):
                    return taylor_candidates[-1]

            # Value convergence
            if len(value_candidates) >= 3:
                v = value_candidates
                d1 = abs(v[-1] - v[-2])
                d2 = abs(v[-2] - v[-3])
                if d1 < d2:
                    if d1 < 1e-3 * (abs(v[-1]) + 1e-100):
                        return v[-1]
                if len(v) >= 4 and all(abs(vi) < 1e-3 for vi in v[-3:]):
                    return 0.0
    finally:
        restore_math()
        _min_terms[0] = old_min

    return None


def _extrapolate_inf(f, to, truncation, n_probes=6):
    """Extrapolate limit at infinity by evaluating at increasing large values."""
    sign = 1 if to == math.inf else -1

    old_min = _min_terms[0]
    _min_terms[0] = truncation

    try:
        patch_math()
        candidates = []
        for k in range(n_probes):
            x_val = sign * 10 ** (k + 2)  # 100, 1000, ..., 10^7
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = f(_seeded(x_val))
            except (ValueError, ZeroDivisionError, OverflowError,
                    LimitDoesNotExistError):
                continue

            if not isinstance(result, Composite):
                result = R(float(result))

            st = result.st()
            if math.isfinite(st):
                candidates.append(st)

        if len(candidates) >= 3:
            v = candidates
            d1 = abs(v[-1] - v[-2])
            d2 = abs(v[-2] - v[-3])
            if d1 < d2:
                if d1 < 1e-3 * (abs(v[-1]) + 1e-100):
                    return v[-1]
            if len(v) >= 4 and all(abs(vi) < 1e-3 for vi in v[-3:]):
                return 0.0
    finally:
        restore_math()
        _min_terms[0] = old_min

    return None


def resolve(f, at=0, dir="both", truncation=20):
    """Evaluate f at a point where it would normally return NaN.

    Alias for limit() with engineering-oriented naming.

    Examples:
        >>> resolve(lambda x: math.sin(x) / x, at=0)
        1.0
    """
    try:
        return limit(f, to=at, dir=dir, truncation=truncation)
    except LimitDivergesError as e:
        return e.value


def evaluate(f, at=0):
    """Evaluate f at a removable singularity. Raises if not removable.

    Examples:
        >>> evaluate(lambda x: (x**2 - 1) / (x - 1), at=1)
        2.0
    """
    info = classify(f, at=at)
    if isinstance(info, (Regular, Removable)):
        return info.value
    raise SingularityError(f"Not a removable singularity: {info}")


def safe(f):
    """Decorator: automatically resolve singularities.

    Normal inputs: runs f directly, zero overhead.
    Singularities: catches NaN/Inf/errors and resolves algebraically.

    Examples:
        @safe
        def sinc(x):
            return math.sin(x) / x

        sinc(0.5)  # → 0.9588... (normal)
        sinc(0)    # → 1.0 (resolved)
    """
    import math as _math
    import functools

    @functools.wraps(f)
    def wrapper(x):
        try:
            y = f(x)
            if isinstance(y, float) and (_math.isnan(y) or _math.isinf(y)):
                return resolve(f, at=x)
            return y
        except (ZeroDivisionError, ValueError, OverflowError):
            return resolve(f, at=x)

    return wrapper
