# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Math module patching for transparent composite interception.

During limit()/resolve() calls, math.sin/cos/exp/etc. are temporarily
replaced with composite-aware versions. This lets user functions written
with math.sin(x) work without any code changes.

Not thread-safe. For concurrent use, import from composite_resolve.math
or use numpy (handled via __array_ufunc__).
"""

import math as _math

from composite_resolve._core import (
    sin as _c_sin, cos as _c_cos, tan as _c_tan,
    exp as _c_exp, ln as _c_ln, sqrt as _c_sqrt,
    atan as _c_atan, asin as _c_asin, acos as _c_acos,
    sinh as _c_sinh, cosh as _c_cosh, tanh as _c_tanh,
)

# Save original math functions before any patching
_originals = {
    'sin': _math.sin, 'cos': _math.cos, 'tan': _math.tan,
    'exp': _math.exp, 'log': _math.log, 'sqrt': _math.sqrt,
    'asin': _math.asin, 'acos': _math.acos, 'atan': _math.atan,
    'sinh': _math.sinh, 'cosh': _math.cosh, 'tanh': _math.tanh,
}

# Composite replacements
_composite = {
    'sin': _c_sin, 'cos': _c_cos, 'tan': _c_tan,
    'exp': _c_exp, 'log': _c_ln, 'sqrt': _c_sqrt,
    'asin': _c_asin, 'acos': _c_acos, 'atan': _c_atan,
    'sinh': _c_sinh, 'cosh': _c_cosh, 'tanh': _c_tanh,
}


def _wrap_composite(name, composite_fn):
    """Create a wrapper that handles float→float and Composite→Composite.

    Uses the saved original math function for float inputs,
    composite function for Composite inputs. Avoids recursion.
    """
    original = _originals[name]
    from composite_resolve._core import Composite

    def wrapper(x, *args, **kwargs):
        if isinstance(x, Composite):
            return composite_fn(x, *args, **kwargs)
        return original(x)

    wrapper.__name__ = f"composite_{name}"
    return wrapper


# Wrappers that dispatch: float→math.original, Composite→composite_fn
_wrappers = {name: _wrap_composite(name, fn) for name, fn in _composite.items()}


def patch_math():
    """Replace math module functions with composite-aware wrappers."""
    for name, fn in _wrappers.items():
        setattr(_math, name, fn)


def restore_math():
    """Restore original math module functions."""
    for name, fn in _originals.items():
        setattr(_math, name, fn)
