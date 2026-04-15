# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Math module patching for transparent composite interception.

During limit()/resolve() calls, every callable in the `math` module is
temporarily replaced:
  - Supported ops (sin, cos, exp, log, sqrt, ...) dispatch to composite
    implementations when given a Composite, original math function otherwise.
  - Unsupported ops (floor, gamma, atan2, ...) raise UnsupportedFunctionError
    if called with a Composite — we refuse to silently coerce to a float,
    which would give a plausible but mathematically wrong answer.

Not thread-safe. For concurrent use, import from composite_resolve.math
or use numpy (handled via __array_ufunc__).
"""

import math as _math

from composite_resolve._core import (
    Composite,
    sin as _c_sin, cos as _c_cos, tan as _c_tan,
    exp as _c_exp, ln as _c_ln, sqrt as _c_sqrt,
    atan as _c_atan, asin as _c_asin, acos as _c_acos,
    sinh as _c_sinh, cosh as _c_cosh, tanh as _c_tanh,
)
from composite_resolve._errors import UnsupportedFunctionError


# Supported functions: name → composite implementation.
# `log` is aliased to the natural log (ln); the one-arg form matches math.log
# when no base is supplied.
_SUPPORTED = {
    'sin': _c_sin, 'cos': _c_cos, 'tan': _c_tan,
    'exp': _c_exp, 'log': _c_ln, 'sqrt': _c_sqrt,
    'asin': _c_asin, 'acos': _c_acos, 'atan': _c_atan,
    'sinh': _c_sinh, 'cosh': _c_cosh, 'tanh': _c_tanh,
}


# Snapshot every callable in `math` at import time, before any patching.
_originals: dict[str, object] = {
    name: getattr(_math, name)
    for name in dir(_math)
    if callable(getattr(_math, name)) and not name.startswith('_')
}


def _make_supported_wrapper(name: str, composite_fn):
    """float → original, Composite → composite_fn."""
    original = _originals[name]

    def wrapper(x, *args, **kwargs):
        if isinstance(x, Composite):
            return composite_fn(x, *args, **kwargs)
        return original(x, *args, **kwargs)

    wrapper.__name__ = f"composite_{name}"
    return wrapper


def _make_unsupported_guard(name: str):
    """float → original, Composite → raise UnsupportedFunctionError.

    Refuses to silently coerce a Composite to its standard part. That coercion
    path is what caused `math.floor(composite)` to return a numerically
    plausible but mathematically wrong answer instead of erroring out.
    """
    original = _originals[name]

    def guard(*args, **kwargs):
        for a in args:
            if isinstance(a, Composite):
                raise UnsupportedFunctionError(f"math.{name}")
        for v in kwargs.values():
            if isinstance(v, Composite):
                raise UnsupportedFunctionError(f"math.{name}")
        return original(*args, **kwargs)

    guard.__name__ = f"composite_guard_{name}"
    return guard


# Build the wrapper table once at import.
_wrappers: dict[str, object] = {}
for _name, _fn in _originals.items():
    if _name in _SUPPORTED:
        _wrappers[_name] = _make_supported_wrapper(_name, _SUPPORTED[_name])
    else:
        _wrappers[_name] = _make_unsupported_guard(_name)


def patch_math():
    """Replace math module callables with composite-aware wrappers."""
    for name, fn in _wrappers.items():
        setattr(_math, name, fn)


def restore_math():
    """Restore original math module callables."""
    for name, fn in _originals.items():
        setattr(_math, name, fn)
