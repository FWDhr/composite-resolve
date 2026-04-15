# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Taylor series coefficient extraction via composite arithmetic.

One composite evaluation at the expansion point gives all coefficients
simultaneously — no symbolic differentiation, no finite differences.
"""

import warnings

from composite_resolve._core import Composite, _seeded, _min_terms
from composite_resolve._compat import patch_math, restore_math


def taylor(f, at=0, order=10):
    """Return exact Taylor coefficients [f(a), f'(a)/1!, f''(a)/2!, ...].

    Args:
        f:     Function f(x).
        at:    Expansion point (default 0).
        order: Number of derivative orders (default 10).

    Returns:
        List of floats: [c_0, c_1, ..., c_order] where
        f(x) ≈ c_0 + c_1·(x-a) + c_2·(x-a)² + ... + c_order·(x-a)^order

    Examples:
        >>> taylor(lambda x: math.exp(x), at=0, order=4)
        [1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664]
        >>> taylor(lambda x: math.sin(x), at=0, order=4)
        [0.0, 1.0, 0.0, -0.16666666666666666, 0.0]
    """
    old_min = _min_terms[0]
    _min_terms[0] = order + 2

    try:
        patch_math()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x = _seeded(at)
            result = f(x)
    finally:
        restore_math()
        _min_terms[0] = old_min

    if not isinstance(result, Composite):
        # Function returned a plain number — constant
        return [float(result)] + [0.0] * order

    return [result.coeff(-k) for k in range(order + 1)]
