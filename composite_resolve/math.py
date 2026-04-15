# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Composite-aware math functions.

These accept both float and Composite inputs:
  - float → float (delegates to math module)
  - Composite → Composite (full derivative propagation)

Use these in functions passed to limit()/resolve() for guaranteed
composite propagation. Alternatively, use math.sin/numpy.sin —
limit() patches them automatically during evaluation.
"""

from composite_resolve._core import (
    sin, cos, tan, exp, ln, sqrt,
    atan, asin, acos, sinh, cosh, tanh,
)

# Alias log → ln for math module compatibility
log = ln

__all__ = [
    "sin", "cos", "tan", "exp", "log", "sqrt",
    "sinh", "cosh", "tanh", "asin", "acos", "atan",
]
