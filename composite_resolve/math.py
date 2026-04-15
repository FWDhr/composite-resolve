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

import math as _pymath

from composite_resolve._core import (
    sin, cos, tan, exp, ln, sqrt,
    atan, asin, acos, sinh, cosh, tanh,
)

# Alias log → ln for math module compatibility
log = ln


# ---------------------------------------------------------------------------
# Reciprocal trig / hyperbolic. Implemented as compositions of existing
# primitives — Composite-aware automatically.
# ---------------------------------------------------------------------------

def cot(x):  return 1 / tan(x)
def sec(x):  return 1 / cos(x)
def csc(x):  return 1 / sin(x)

def coth(x): return 1 / tanh(x)
def sech(x): return 1 / cosh(x)
def csch(x): return 1 / sinh(x)


# ---------------------------------------------------------------------------
# Inverse hyperbolic (real domain).
#   asinh(x) = ln(x + √(x²+1))
#   acosh(x) = ln(x + √(x²−1))     (x ≥ 1)
#   atanh(x) = ½·ln((1+x)/(1−x))    (|x| < 1)
# ---------------------------------------------------------------------------

def asinh(x): return ln(x + sqrt(x*x + 1))
def acosh(x): return ln(x + sqrt(x*x - 1))
def atanh(x): return 0.5 * ln((1 + x) / (1 - x))


# ---------------------------------------------------------------------------
# Inverse reciprocal trig / hyperbolic.
# ---------------------------------------------------------------------------

def acot(x):  return _pymath.pi / 2 - atan(x)
def asec(x):  return acos(1 / x)
def acsc(x):  return asin(1 / x)

def asech(x): return acosh(1 / x)
def acsch(x): return asinh(1 / x)
def acoth(x): return atanh(1 / x)


__all__ = [
    "sin", "cos", "tan", "exp", "log", "ln", "sqrt",
    "sinh", "cosh", "tanh",
    "asin", "acos", "atan",
    "cot", "sec", "csc", "coth", "sech", "csch",
    "asinh", "acosh", "atanh",
    "acot", "asec", "acsc", "asech", "acsch", "acoth",
]
