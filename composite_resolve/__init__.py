# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""
composite-resolve — Exact limit computation via composite arithmetic.

Resolves indeterminate forms (0/0, 0×∞, ∞−∞, 0⁰, 1^∞, ∞⁰) algebraically,
without symbolic engines, approximation, or special-casing.

Primary API:
    limit(f, to, dir)     — compute the exact limit of f(x) as x → to
    resolve(f, at, dir)   — alias for limit (engineering naming)
    evaluate(f, at)       — evaluate f at a removable singularity

Analysis:
    taylor(f, at, order)  — extract Taylor coefficients
    classify(f, at, dir)  — classify singularity type
    residue(f, at)        — compute residue at a pole

Math functions (composite-aware, accept plain floats):
    from composite_resolve.math import sin, cos, exp, log, sqrt, ...
"""

from composite_resolve._limit import limit, resolve, evaluate, safe
from composite_resolve._taylor import taylor
from composite_resolve._classify import classify, residue
from composite_resolve._errors import (
    CompositeResolveError,
    LimitDoesNotExistError,
    LimitUndecidableError,
    LimitDivergesError,
    SingularityError,
    CompositionError,
    UnsupportedFunctionError,
)
from composite_resolve._classify import Regular, Removable, Pole, Essential

__version__ = "0.1.0"
__all__ = [
    "limit", "resolve", "evaluate", "safe",
    "taylor", "classify", "residue",
    "CompositeResolveError", "LimitDoesNotExistError",
    "LimitUndecidableError",
    "LimitDivergesError", "SingularityError", "CompositionError",
    "UnsupportedFunctionError",
    "Regular", "Removable", "Pole", "Essential",
]
