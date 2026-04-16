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

    # Fast path for regular points: if f(float(to)) returns a finite float
    # cleanly, that IS the limit. Composite arithmetic is only load-bearing
    # when dimensions actually cancel (×0 / ×∞ at the limit point). For
    # regular points it's pure overhead — a single float evaluation suffices.
    if math.isfinite(to):
        import warnings as _w
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                y = f(float(to))
            if isinstance(y, (int, float)) and math.isfinite(y):
                return float(y)
        except (ZeroDivisionError, ValueError, OverflowError, TypeError,
                AttributeError):
            pass  # fall through; composite path handles errors / raises CompositionError

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
            # Right side undefined: fall back to left side if that works.
            if _function_undefined_on_side(f, to, "+"):
                try:
                    return _limit_one_sided(f, to, "-", truncation)
                except LimitDivergesError as e:
                    raise
            raise

        try:
            left = _limit_one_sided(f, to, "-", truncation)
        except LimitDivergesError as e:
            raise LimitDoesNotExistError(
                "Left limit diverges but right limit is finite",
                left_limit=e.value, right_limit=right)
        except LimitDoesNotExistError:
            # Left side is undefined or can't be resolved. If the function is
            # only defined on the right (entropy `-p·log p` at 0, `sqrt(x)` at
            # 0, etc.), fall back to the right-sided limit.
            if _function_undefined_on_side(f, to, "-"):
                return right
            raise

        # Tolerance that covers both algebraic evaluation (exact to floating
        # precision) and numerical extrapolation fallbacks (typically ~1e-6).
        if abs(right - left) < 1e-5 * (abs(right) + abs(left)) + 1e-8:
            return 0.5 * (right + left)
        raise LimitDoesNotExistError(
            f"One-sided limits disagree: left={left}, right={right}",
            left_limit=left, right_limit=right)

    # Single direction or infinity
    return _limit_one_sided(f, to, dir, truncation)


def _function_undefined_on_side(f, to, side, n_probes=4):
    """Heuristic: does f raise a domain error at every probe on `side` of `to`?

    Used to distinguish "function genuinely undefined here" from "oscillates
    / extrapolation failed". When one side is undefined everywhere near the
    point (e.g. sqrt(x) on the left of 0, log(x) on the left of 0), the
    two-sided limit is conventionally taken as the defined-side limit.
    """
    import warnings as _w
    sign = -1 if side == "-" else 1
    base = to if math.isfinite(to) else 0.0
    scale = abs(base) if abs(base) > 1e-9 else 1.0
    failures = 0
    for k in range(n_probes):
        eps = 10 ** (-(k + 2)) * scale
        x = base + sign * eps
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", RuntimeWarning)
                y = f(x)
            if isinstance(y, float) and (math.isnan(y) or math.isinf(y)):
                failures += 1
        except (ValueError, ZeroDivisionError, OverflowError):
            failures += 1
        except Exception:
            pass
    return failures == n_probes


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
            except (ValueError, ZeroDivisionError, CompositionError):
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

    # Positive dims → divergence. The HIGHEST positive dimension dominates
    # all lower ones (∞² beats ∞ beats finite), so the sign of the leading
    # coefficient is the limit's sign. Lower-dim positive coefficients —
    # whatever their sign — are dominated and don't matter.
    max_pos = result.max_positive_dim()
    if max_pos is not None:
        pos_coeffs = {d: c for d, c in result.coeffs_dict().items()
                      if d > 0 and abs(c) > 1e-10}
        if not pos_coeffs:
            # All positive dims are floating-point noise — clean result.
            return st_val
        leading_dim = max(pos_coeffs)
        leading = pos_coeffs[leading_dim]
        raise LimitDivergesError(math.inf if leading > 0 else -math.inf)

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

            # Probe eps schedule ordered gentle → aggressive. value_candidates
            # ends up sorted by closeness-to-limit. When aggressive probes
            # overflow (e.g. `5^(1/x)` at x ≤ 1e-3), only the gentle ones
            # remain and the convergence check still sees "the closest
            # surviving samples" as the latest entries.
            _probe_eps = [1e-1, 3e-2, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

            for k in range(len(_probe_eps)):
                eps = _probe_eps[k]
                x_val = to + sign * eps

                # Try composite-seeded probe first (enables Taylor extrapolation).
                # If it hits a non-representable composite op, fall back to plain
                # float — the function is evaluated numerically at the probe.
                result = None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = f(_seeded(x_val))
                except (ValueError, ZeroDivisionError, OverflowError,
                        LimitDoesNotExistError, CompositionError):
                    pass

                if result is None:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            result = f(x_val)
                    except (ValueError, ZeroDivisionError, OverflowError,
                            LimitDoesNotExistError, CompositionError, TypeError):
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
                if d1 <= d2 and d1 < 1e-3 * (abs(v[-1]) + 1e-100):
                    return v[-1]
                if len(v) >= 4 and all(abs(vi) < 1e-3 for vi in v[-3:]):
                    return 0.0

                # Divergence detection: same-sign, monotonically growing
                # magnitudes reaching a large value → ±∞.
                mags = [abs(vi) for vi in v]
                same_sign = all(vi > 0 for vi in v) or all(vi < 0 for vi in v)
                monotone = all(mags[i+1] > mags[i] for i in range(len(mags)-1))
                incs = [mags[i+1] - mags[i] for i in range(len(mags)-1)]
                not_shrinking = len(incs) > 0 and incs[-1] > 0.5 * incs[0]
                if (same_sign and monotone and not_shrinking
                        and mags[-1] > 5 and mags[-1] > 3 * mags[0]):
                    raise LimitDivergesError(math.inf if v[-1] > 0 else -math.inf)

                # Convergence to zero: magnitudes monotonically shrinking,
                # last one much smaller than the first.
                monotone_down = all(mags[i+1] < mags[i] for i in range(len(mags)-1))
                if monotone_down and mags[-1] < 0.01 * mags[0] and mags[-1] < 0.1:
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
            result = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    result = f(_seeded(x_val))
            except (ValueError, ZeroDivisionError, OverflowError,
                    LimitDoesNotExistError, CompositionError):
                pass

            if result is None:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        result = f(x_val)
                except (ValueError, ZeroDivisionError, OverflowError,
                        LimitDoesNotExistError, CompositionError, TypeError):
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
            # Tight convergence: last step is already small vs. the value
            # itself. Use `<=` so exactly-equal probes (d1 == d2 == 0) count
            # as converged.
            if d1 <= d2 and d1 < 1e-3 * (abs(v[-1]) + 1e-100):
                return v[-1]
            if len(v) >= 4 and all(abs(vi) < 1e-3 for vi in v[-3:]):
                return 0.0

            # Divergence: same-sign, magnitudes monotonically growing with
            # non-shrinking increments (so no convergence in sight). Modest
            # threshold so logarithmic divergence is caught.
            mags = [abs(vi) for vi in v]
            same_sign = all(vi > 0 for vi in v) or all(vi < 0 for vi in v)
            monotone = all(mags[i+1] > mags[i] for i in range(len(mags)-1))
            incs = [mags[i+1] - mags[i] for i in range(len(mags)-1)]
            not_shrinking = len(incs) > 0 and incs[-1] > 0.5 * incs[0]
            if (same_sign and monotone and not_shrinking
                    and mags[-1] > 5 and mags[-1] > 3 * mags[0]):
                raise LimitDivergesError(math.inf if v[-1] > 0 else -math.inf)

            # Convergence to zero: magnitudes monotonically shrinking to tiny.
            monotone_down = all(mags[i+1] < mags[i] for i in range(len(mags)-1))
            if monotone_down and mags[-1] < 0.01 * mags[0] and mags[-1] < 0.1:
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
