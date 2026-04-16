# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Singularity classification via composite dimensional analysis.

One composite evaluation reveals the singularity type:
  - No positive dims, dim 0 exists → Regular or Removable
  - Positive dims → Pole (order = highest positive dim)
  - Empty composite (nothing) → Essential (oscillatory)
"""

import math
import warnings

from composite_resolve._core import Composite, _seeded, _min_terms, _is_nothing
from composite_resolve._compat import patch_math, restore_math
from composite_resolve._errors import SingularityError


class Regular:
    """No singularity. f is well-defined at this point."""
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Regular(value={self.value})"


class Removable:
    """Removable singularity. Limit exists and is finite."""
    def __init__(self, value, order=1, indeterminate_form="0/0"):
        self.value = value
        self.order = order
        self.indeterminate_form = indeterminate_form

    def __repr__(self):
        return (f"Removable(value={self.value}, order={self.order}, "
                f"form='{self.indeterminate_form}')")


class Pole:
    """Pole. Function diverges."""
    def __init__(self, order=1, residue=0.0):
        self.order = order
        self.residue = residue

    def __repr__(self):
        return f"Pole(order={self.order}, residue={self.residue})"


class Essential:
    """Essential singularity. No Laurent series with finite principal part."""
    def __repr__(self):
        return "Essential()"


def classify(f, at=0, dir="both"):
    """Classify the singularity of f at the given point.

    Args:
        f:   Function f(x).
        at:  Point to classify (default 0).
        dir: Direction: "both", "+", or "-".

    Returns:
        Regular, Removable, Pole, or Essential.

    Examples:
        >>> classify(lambda x: math.sin(x)/x, at=0)
        Removable(value=1.0, order=1, form='0/0')
        >>> classify(lambda x: 1/x, at=0)
        Pole(order=1, residue=1.0)
        >>> classify(lambda x: math.exp(x), at=0)
        Regular(value=1.0)
    """
    # First: try plain float evaluation to check if it's regular
    try:
        plain_val = float(f(at))
        if math.isfinite(plain_val) and abs(plain_val) < 1e12:
            return Regular(value=plain_val)
        # Huge finite value (e.g. tan(pi/2) ≈ 1.6e16) → treat as singular
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        pass

    # Composite evaluation
    old_min = _min_terms[0]
    _min_terms[0] = 20

    try:
        patch_math()
        if dir == "-":
            x = Composite({0: float(at), -1: -1.0}) if at != 0 else -Composite.zero()
        else:
            x = _seeded(at)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                result = f(x)
            except (TypeError, ValueError, ZeroDivisionError,
                    OverflowError, CompositionError) as e:
                from composite_resolve._errors import LimitDoesNotExistError
                if isinstance(e, LimitDoesNotExistError):
                    return Essential()
                # Domain error or non-representable composite — fall back to probing.
                return _classify_from_probes(f, at, dir)
    finally:
        restore_math()
        _min_terms[0] = old_min

    if not isinstance(result, Composite):
        return Regular(value=float(result))

    # Nothing → essential singularity (oscillatory)
    if _is_nothing(result):
        return Essential()

    st_val = result.st()
    max_pos = result.max_positive_dim()

    # Positive dims → pole
    if max_pos is not None:
        residue_val = result.coeff(1) if result.coeff(1) != 0 else 0.0
        return Pole(order=max_pos, residue=residue_val)

    # No positive dims, finite st → removable singularity
    # (we already checked it's not regular above)
    if math.isfinite(st_val):
        # Determine order: how many negative dims have significant coefficients
        coeffs = result.coeffs_dict()
        neg_nonzero = [d for d, c in coeffs.items()
                       if d < 0 and abs(c) > 1e-15]
        order = len(neg_nonzero) if neg_nonzero else 1
        return Removable(value=st_val, order=min(order, 10))

    return Essential()


def _classify_from_probes(f, at, dir):
    """Fallback classification from nearby evaluations."""
    sign = -1 if dir == "-" else 1
    for eps in [1e-3, 1e-5, 1e-7]:
        try:
            val = float(f(at + sign * eps))
            if not math.isfinite(val):
                return Pole(order=1)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    return Essential()


def residue(f, at=0):
    """Compute the residue of f at a pole.

    Args:
        f:  Function f(x).
        at: Pole location (default 0).

    Returns:
        float — the residue (coefficient of 1/(x-a) term).

    Raises:
        SingularityError: if the point is not a pole.

    Examples:
        >>> residue(lambda x: 1/x, at=0)
        1.0
    """
    info = classify(f, at=at)
    if not isinstance(info, Pole):
        raise SingularityError(f"Not a pole: {info}")
    return info.residue
