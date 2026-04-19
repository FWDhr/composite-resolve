# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Composite arithmetic core — pure Python, zero dependencies.

Minimal implementation of composite numbers for limit computation.
Dict-based: {dimension: coefficient}. Direct convolution for multiplication.
No numpy, no backends, no overhead.

This replaces the development shim that imported from composite_lib.
All composite_resolve modules import from here.
"""

import math

from composite_resolve._errors import CompositionError

# Maximum number of active (non-zero) dimensions kept after convolution
# and polynomial long division. Prevents O(n^2) dimension explosion in
# deeply nested function chains. Keeps dims closest to 0 (most significant
# for limits and derivatives). Discarded tails have coefficients of order
# 1/n! and contribute negligibly within float precision.
MAX_ACTIVE_DIMS = 60

# Optional numpy acceleration for convolution.
try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_NUMPY_THRESHOLD = 25
_FFT_THRESHOLD = 128
_CLUSTER_GAP = 5


def _fft_convolve(a, b):
    n = len(a) + len(b) - 1
    return _np.fft.irfft(_np.fft.rfft(a, n) * _np.fft.rfft(b, n), n)


def _dict_to_sparse(d):
    dims = sorted(d.keys())
    return _np.array(dims, dtype=_np.int64), _np.array([d[k] for k in dims])


def _find_clusters(indices):
    if len(indices) == 0:
        return []
    clusters = []
    start = 0
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > _CLUSTER_GAP:
            clusters.append((start, i))
            start = i
    clusters.append((start, len(indices)))
    return clusters


def _cluster_to_dense(indices, values, start_idx, end_idx):
    cl_idx = indices[start_idx:end_idx]
    cl_val = values[start_idx:end_idx]
    min_d, max_d = int(cl_idx[0]), int(cl_idx[-1])
    dense = _np.zeros(max_d - min_d + 1)
    for i in range(len(cl_idx)):
        dense[int(cl_idx[i]) - min_d] = cl_val[i]
    return dense, min_d


def _convolve_sparse(d_a, d_b):
    idx_a, val_a = _dict_to_sparse(d_a)
    idx_b, val_b = _dict_to_sparse(d_b)
    clusters_a = _find_clusters(idx_a)
    clusters_b = _find_clusters(idx_b)
    result = {}
    for sa, ea in clusters_a:
        da, oa = _cluster_to_dense(idx_a, val_a, sa, ea)
        for sb, eb in clusters_b:
            db, ob = _cluster_to_dense(idx_b, val_b, sb, eb)
            conv = _fft_convolve(da, db) if len(da) + len(db) > _FFT_THRESHOLD else _np.convolve(da, db)
            off = oa + ob
            for i in range(len(conv)):
                c = conv[i]
                if abs(c) > 1e-15:
                    dim = off + i
                    result[dim] = result.get(dim, 0.0) + float(c)
    return result


def _truncate(d):
    if len(d) <= MAX_ACTIVE_DIMS:
        return d
    sorted_dims = sorted(d.keys(), key=lambda x: abs(x))
    keep = set(sorted_dims[:MAX_ACTIVE_DIMS])
    # Always preserve nonfinite coefficients (lossy-infinity markers).
    # These are signals that the algebraic path produced an indeterminate
    # result. Discarding them would mask the error and return a wrong
    # answer instead of triggering the fallback/recovery path.
    for dim, c in d.items():
        if not math.isfinite(c):
            keep.add(dim)
    return {dim: c for dim, c in d.items() if dim in keep}


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LimitDoesNotExistError(ValueError):
    """Raised when a limit provably does not exist."""
    pass

# =============================================================================
# COMPOSITE NUMBER
# =============================================================================

class Composite:
    """Composite number: sparse dict {dimension: coefficient}.

    Dimension 0  = real value (standard part)
    Dimension -1 = infinitesimal (first order)
    Dimension -2 = second-order infinitesimal
    Dimension +1 = infinity (first order)
    """

    __slots__ = ['_d', '_expressed_zero']

    def __init__(self, data=None):
        self._expressed_zero = False
        if data is None:
            self._d = {}
        elif isinstance(data, dict):
            self._d = {d: c for d, c in data.items() if c != 0}
        elif isinstance(data, (int, float)):
            self._d = {0: float(data)} if data != 0 else {}
        else:
            raise TypeError(f"Cannot create Composite from {type(data)}")

    # --- Constructors ---

    @classmethod
    def zero(cls):
        """Structural zero: |1|₋₁"""
        return cls({-1: 1.0})

    @classmethod
    def infinity(cls):
        """Structural infinity: |1|₊₁"""
        return cls({1: 1.0})

    @classmethod
    def real(cls, value):
        """Real number: |value|₀"""
        if value == 0:
            return cls.zero()
        return cls({0: float(value)})

    # --- Representation ---

    def __repr__(self):
        if not self._d:
            return "∅"
        sub = "₀₁₂₃₄₅₆₇₈₉"
        def fmt_dim(n):
            if n >= 0:
                return ''.join(sub[int(d)] for d in str(n))
            return "₋" + ''.join(sub[int(d)] for d in str(-n))
        def fmt_coeff(c):
            if not math.isfinite(c):
                return "∞" if c > 0 else "-∞" if c < 0 else "NaN"
            if c == int(c):
                return str(int(c))
            return f"{c:.6g}"
        parts = [f"|{fmt_coeff(v)}|{fmt_dim(d)}"
                 for d, v in sorted(self._d.items(), reverse=True)]
        return " + ".join(parts)

    # --- Properties ---

    @property
    def c(self):
        """Dict view {dim: coeff}."""
        return dict(self._d)

    def st(self):
        """Standard part: coefficient at dimension 0."""
        return self._d.get(0, 0.0)

    def coeff(self, dim):
        """Coefficient at specific dimension."""
        return self._d.get(dim, 0.0)

    def d(self, n=1):
        """Extract nth derivative: coeff at dim -n times n!"""
        return self._d.get(-n, 0.0) * math.factorial(n)

    def max_positive_dim(self):
        """Highest positive dimension, or None."""
        pos = [d for d in self._d if d > 0]
        return max(pos) if pos else None

    def coeffs_dict(self):
        """Return {dim: coeff} for all dimensions."""
        return dict(self._d)

    def eval_taylor(self, h):
        """Evaluate Taylor expansion: sum c_n * h^|n| for dim < 0."""
        total = 0.0
        for dim, coeff in self._d.items():
            if dim < 0:
                total += coeff * (h ** (-dim))
        return total

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                if 0 not in self._d and self._d:
                    return self + Composite({-1: 1.0})
                return self
            other = Composite(other)
        result = dict(self._d)
        for d, c in other._d.items():
            result[d] = result.get(d, 0.0) + c
        return Composite(result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                if 0 not in self._d and self._d:
                    return self - Composite({-1: 1.0})
                return self
            other = Composite(other)
        result = dict(self._d)
        for d, c in other._d.items():
            result[d] = result.get(d, 0.0) - c
        # Check for exact cancellation
        vals = [v for v in result.values() if v != 0]
        if not vals:
            has_real_content = any(d >= 0 for d in self._d) or any(d >= 0 for d in other._d)
            if has_real_content:
                r = Composite({})
                r._expressed_zero = True
                return r
            else:
                all_dims = list(self._d.keys()) + list(other._d.keys())
                if all_dims:
                    min_dim = min(all_dims)
                    power = max(2, 1 - min_dim)
                    z = Composite({-1: 1.0})
                    for _ in range(power - 1):
                        z = z * Composite({-1: 1.0})
                    return z
        return Composite(result)

    def __rsub__(self, other):
        return Composite(other).__sub__(self)

    def __neg__(self):
        return Composite({d: -c for d, c in self._d.items()})

    def __mul__(self, other):
        """Multiplication via direct convolution.

        Integer-dimension arithmetic: `|a|ₘ · |b|ₙ = |a·b|_{m+n}`.

        Non-finite coefficients (`±∞`, `NaN`) are markers of a "lossy"
        fallback value produced when a composite operation dropped to
        real-valued math (e.g. `ln(|1|₋₁) → |-∞|₁`). When such a marker
        ends up at a dim ≤ 0 after convolution, it's the signature of a
        `0·∞` cancellation whose rate was lost — refuse and let the
        outer evaluator (`limit`) fall back to numerical extrapolation.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                return self * Composite({-1: 1.0})
            return Composite({d: c * other for d, c in self._d.items()})
        if isinstance(other, Composite):
            _ZERO_D = {-1: 1.0}
            self_d = _ZERO_D if self._expressed_zero else self._d
            other_d = _ZERO_D if other._expressed_zero else other._d
            if (_HAS_NUMPY
                    and len(self_d) > _NUMPY_THRESHOLD
                    and len(other_d) > _NUMPY_THRESHOLD):
                result = _convolve_sparse(self_d, other_d)
            else:
                result = {}
                for d1, c1 in self_d.items():
                    for d2, c2 in other_d.items():
                        d = d1 + d2
                        result[d] = result.get(d, 0.0) + c1 * c2
            for d, c in result.items():
                if not math.isfinite(c) and d <= 0:
                    raise CompositionError(
                        "0*inf indeterminate: lossy-infinity coefficient "
                        "cancelled into a finite/infinitesimal dimension")
            return Composite(_truncate(result))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return self.__truediv__(Composite({-1: 1.0}))
            return Composite({d: c / other for d, c in self._d.items()})
        if isinstance(other, Composite):
            if not other._d:
                raise LimitDoesNotExistError(
                    "Division by nothing (empty composite). "
                    "Denominator is indeterminate.")
            # Single-term divisor: fast path
            if len(other._d) == 1:
                div_dim, div_coeff = next(iter(other._d.items()))
                # Refuse when dividing by a lossy infinity (nonfinite
                # coefficient at positive dim). Otherwise the lossy-ness is
                # silently absorbed into a numerically-zero quotient and we
                # lose the information that the result is indeterminate.
                if not math.isfinite(div_coeff) and div_dim > 0:
                    raise CompositionError(
                        "Division by lossy-infinity composite; quotient "
                        "would silently lose rate information")
                return Composite({d - div_dim: c / div_coeff
                                  for d, c in self._d.items()})
            # Multi-term: polynomial long division (deconvolution)
            return _deconvolve(self, other)
        return NotImplemented

    def __rtruediv__(self, other):
        return Composite(other).__truediv__(self)

    def __pow__(self, n):
        if isinstance(n, int):
            if n == 0:
                return Composite({0: 1.0})
            if n < 0:
                return Composite({0: 1.0}) / (self ** (-n))
            result = Composite({0: 1.0})
            for _ in range(n):
                result = result * self
            return result

        if isinstance(n, float):
            if n == int(n):
                return self ** int(n)
            # Non-integer float exponent.
            # (a) self is purely dim 0 → plain scalar power, no composite log.
            if set(self._d) <= {0}:
                c = self._d.get(0, 0.0)
                if c > 0:
                    return Composite({0: math.pow(c, n)})
                if c == 0:
                    if n > 0: return Composite({})
                    raise ValueError("0 ** non-positive non-integer is indeterminate")
                raise ValueError(f"Non-integer power of negative scalar: {c}^{n}")
            # (b) self is structurally infinite → scalar 0 / inf at dim 0.
            pos = {d: c for d, c in self._d.items() if d > 0 and abs(c) > 1e-15}
            if pos:
                leading = pos[max(pos)]
                if leading > 0:
                    return Composite({0: math.inf}) if n > 0 else Composite({})
                raise CompositionError(
                    "Non-integer power of structurally negative-infinite base")
            # (c) Finite positive st + infinitesimals → call ln on self; its st > 0
            #     so no CompositionError.
            return exp(Composite(n) * ln(self))

        if isinstance(n, Composite):
            # Reduce to simpler cases whenever possible.
            # (a) Exponent is purely dim 0 → use the float-exponent path.
            if set(n._d) <= {0}:
                return self ** n.coeff(0)
            # (b) Exponent has structural infinity → scalar 0/inf depending on
            #     self's magnitude class. Doesn't invoke composite log.
            n_pos = {d: c for d, c in n._d.items() if d > 0 and abs(c) > 1e-15}
            if n_pos:
                leading_exp = n_pos[max(n_pos)]
                exp_to_plus_inf = leading_exp > 0
                # Classify self's magnitude
                self_pos = {d: c for d, c in self._d.items() if d > 0 and abs(c) > 1e-15}
                if self_pos:
                    # self → ±∞.  (+∞)^(+∞) = +∞, (+∞)^(-∞) = 0.
                    if self_pos[max(self_pos)] > 0:
                        return Composite({0: math.inf}) if exp_to_plus_inf else Composite({})
                    raise CompositionError(
                        "Structurally negative-infinite base raised to composite exponent")
                a = self.st()
                # If self has non-dim-0 components AND st == 1, this is the
                # classical 1^∞ indeterminate form — the infinitesimal part
                # is what determines the limit. Refuse; let extrapolation
                # (or the Taylor path via exp(n · ln self)) resolve it.
                has_nontrivial = any(d != 0 and abs(c) > 1e-15
                                     for d, c in self._d.items())
                if a == 1 and has_nontrivial:
                    raise CompositionError(
                        "1^∞ indeterminate form: base's infinitesimal part "
                        "determines the limit; refusing algebraic shortcut")
                if a == 1:
                    return Composite({0: 1.0})      # exact 1 ^ anything = 1
                if a > 1:
                    return Composite({0: math.inf}) if exp_to_plus_inf else Composite({})
                if 0 < a < 1:
                    return Composite({}) if exp_to_plus_inf else Composite({0: math.inf})
                # a <= 0: sign-dependent / indeterminate
                raise CompositionError(
                    "Non-positive base raised to structurally infinite composite exponent")
            # (c) Exponent = integer-valued st + infinitesimals.
            n_st = n.st()
            h = Composite({d: c for d, c in n._d.items() if d != 0 and abs(c) > 1e-15})
            # Self has a scalar-infinite coefficient at dim 0 (e.g. from 5^INF)?
            # Then base^exponent is an ∞^(composite) form; not resolvable
            # purely algebraically. Refuse and let extrapolation handle it.
            st_val = self.st()
            if not math.isfinite(st_val):
                raise CompositionError(
                    "Scalar-infinite base raised to composite exponent is "
                    "indeterminate algebraically; falling back is required")
            if n_st == int(n_st) and abs(n_st) < 100:
                int_part = int(n_st)
                base_pow = self ** int_part
                if not h._d:
                    return base_pow
                if {d for d, c in self._d.items() if d > 0 and abs(c) > 1e-15}:
                    return base_pow
                return base_pow * exp(h * ln(self))
            # (d) Non-integer st in exponent. ln(self) is only valid if self has
            #     finite positive st.
            if {d for d, c in self._d.items() if d > 0 and abs(c) > 1e-15}:
                raise CompositionError(
                    "Structurally infinite base with non-integer composite exponent")
            return exp(n * ln(self))

        raise TypeError(f"Power exponent must be int, float, or Composite, got {type(n)}")

    def __rpow__(self, base):
        """base ** self  —  plain float/int base with composite exponent.

        No composite `ln` is ever invoked: `log(base)` is a plain scalar.
        Cases:
          * exponent has structural infinity → scalar 0 / math.inf at dim 0
            (Principle 3: no ×0/×∞ multiplication happened, no dim shift)
          * exponent has finite st + infinitesimals → base^st · exp(h · log base)
          * base = 1 short-circuit, base ≤ 0 refused
        """
        if not isinstance(base, (int, float)):
            return NotImplemented
        if base == 1:
            return Composite({0: 1.0})
        if base <= 0:
            if base == 0:
                st = self.st()
                if st > 0:
                    return Composite({})
                raise ValueError("0 ** composite with non-positive exponent is indeterminate")
            raise ValueError(f"Cannot compute {base}**composite for negative base")

        pos = {d: c for d, c in self._d.items() if d > 0 and abs(c) > 1e-15}
        if pos:
            exp_to_plus_inf = pos[max(pos)] > 0
            base_gt_1 = base > 1
            if exp_to_plus_inf == base_gt_1:
                return Composite({0: math.inf})
            return Composite({})

        a = self.st()
        h = Composite({d: c for d, c in self._d.items() if d < 0 and abs(c) > 1e-15})
        base_a = math.pow(base, a)
        if not h._d:
            return Composite({0: base_a})
        return Composite({0: base_a}) * exp(h * math.log(base))

    def __abs__(self):
        if not self._d:
            return Composite({})
        top_dim = max(self._d.keys())
        if self._d[top_dim] >= 0:
            return self
        return -self

    def __float__(self):
        return self.st()

    def __format__(self, fmt):
        if fmt:
            return format(self.st(), fmt)
        return repr(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy ufunc dispatch. Makes np.sin(composite) just work."""
        from composite_resolve._errors import UnsupportedFunctionError
        try:
            import numpy as np
        except ImportError:
            raise UnsupportedFunctionError(f"numpy.{ufunc.__name__}")

        _dispatch = {
            np.sin: sin, np.cos: cos, np.tan: tan,
            np.exp: exp, np.log: ln, np.sqrt: sqrt,
            np.arctan: atan, np.arcsin: asin, np.arccos: acos,
            np.sinh: sinh, np.cosh: cosh, np.tanh: tanh,
            np.abs: abs, np.absolute: abs,
            np.expm1: expm1, np.log1p: log1p,
            np.floor: floor, np.ceil: ceiling,
        }

        if ufunc in _dispatch:
            return _dispatch[ufunc](*inputs, **kwargs)

        # Arithmetic ufuncs
        if ufunc == np.add:
            return inputs[0] + inputs[1]
        if ufunc == np.subtract:
            return inputs[0] - inputs[1]
        if ufunc == np.multiply:
            return inputs[0] * inputs[1]
        if ufunc == np.true_divide:
            return inputs[0] / inputs[1]
        if ufunc == np.negative:
            return -inputs[0]
        if ufunc == np.power:
            return inputs[0] ** inputs[1]

        raise UnsupportedFunctionError(f"numpy.{ufunc.__name__}")


def _deconvolve(a, b):
    """Polynomial long division in composite space."""
    b_dims = sorted(b._d.keys(), reverse=True)
    lead_dim = b_dims[0]
    lead_coeff = b._d[lead_dim]

    remainder = dict(a._d)
    result = {}

    for _ in range(60):  # max iterations
        if not remainder:
            break
        r_dims = sorted(remainder.keys(), reverse=True)
        r_dim = r_dims[0]
        r_coeff = remainder[r_dim]
        if abs(r_coeff) < 1e-100:
            del remainder[r_dim]
            continue

        q_dim = r_dim - lead_dim
        q_coeff = r_coeff / lead_coeff
        result[q_dim] = result.get(q_dim, 0.0) + q_coeff

        for bd, bc in b._d.items():
            sub_dim = q_dim + bd
            remainder[sub_dim] = remainder.get(sub_dim, 0.0) - q_coeff * bc
        # Clean near-zeros
        remainder = {d: c for d, c in remainder.items() if abs(c) > 1e-50}

    return Composite(_truncate(result))


# =============================================================================
# CONSTANTS
# =============================================================================

ZERO = Composite.zero()       # |1|₋₁
INF = Composite.infinity()    # |1|₊₁


def R(x):
    """Create real composite. R(0) = ZERO."""
    if x == 0:
        return Composite.zero()
    return Composite({0: float(x)})


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

# Global minimum terms for transcendentals
_min_terms = [0]

def _effective_terms(default):
    return max(default, _min_terms[0])

def _seeded(at):
    """Evaluation point seeded with infinitesimal."""
    x = R(at)
    if at == 0:
        return x
    return x + ZERO

def _is_nothing(x):
    """Check if x is the empty composite (nothing)."""
    return isinstance(x, Composite) and not x._d

def _has_positive_dims(x):
    return x.max_positive_dim() is not None


# =============================================================================
# BOUNDED TRANSCENDENTAL HANDLING
# =============================================================================

def _bounded_at_inf(func, x):
    """Evaluate bounded transcendental at infinite argument.

    Monotonic (atan, tanh): math.func(±inf) → correct asymptotic.
    Oscillatory (sin, cos): math.func(±inf) → ValueError → return ∅.
    """
    max_d = x.max_positive_dim()
    sign = 1.0 if x.coeff(max_d) > 0 else -1.0
    try:
        return R(func(sign * float('inf')))
    except (ValueError, OverflowError):
        return Composite({})


# =============================================================================
# TRANSCENDENTAL FUNCTIONS
# =============================================================================

def _sincos_h_series(h, terms):
    """Build exact rational Taylor series for sin(h) and cos(h) via the
    derivative recurrence.  Invariant: `math.sin`/`math.cos` are called only
    at the real base point `a` by callers — never on a composite.  Coefficients
    here are exact rationals (unit scalars divided by consecutive integers).

    Recurrence (from `sin' = cos`, `cos' = -sin`): each new term = prev · h / n,
    routed to sin_h on odd n, cos_h on even n, with alternating signs.
    """
    sin_h = Composite({})
    cos_h = Composite({0: 1.0})
    term = Composite({0: 1.0})
    for n in range(1, terms):
        term = (term * h) / n
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
        else:
            sign = (-1) ** (n // 2)
        signed = term if sign > 0 else -term
        if n % 2 == 1:
            sin_h = sin_h + signed
        else:
            cos_h = cos_h + signed
    return sin_h, cos_h


def sin(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.sin(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.sin, x)
    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0})
    if not h._d:
        return R(math.sin(a))
    sin_a, cos_a = math.sin(a), math.cos(a)
    sin_h, cos_h = _sincos_h_series(h, terms)
    # sin(a + h) = sin(a)·cos(h) + cos(a)·sin(h)
    # Skip zero-coefficient terms to avoid scalar-0 uplift in __mul__
    result = Composite({})
    if sin_a != 0:
        result = result + sin_a * cos_h
    if cos_a != 0:
        result = result + cos_a * sin_h
    return result


def cos(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.cos(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.cos, x)
    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0})
    if not h._d:
        return R(math.cos(a))
    sin_a, cos_a = math.sin(a), math.cos(a)
    sin_h, cos_h = _sincos_h_series(h, terms)
    # cos(a + h) = cos(a)·cos(h) - sin(a)·sin(h)
    # Skip zero-coefficient terms to avoid scalar-0 uplift in __mul__
    result = Composite({})
    if cos_a != 0:
        result = result + cos_a * cos_h
    if sin_a != 0:
        result = result - sin_a * sin_h
    return result


def exp(x, terms=15):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.exp(float(x))
    if _is_nothing(x):
        return Composite({})
    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0 and abs(c) > 1e-15})
    if not h._d:
        return Composite({0: math.exp(a)})

    # Asymptotic handling when the argument has unbounded (positive-dim)
    # components. The Taylor expansion of exp around x=∞ diverges — it's the
    # wrong tool for an unbounded input. The correct behavior:
    #   exp(value that grows to +∞) → +∞
    #   exp(value that falls to −∞) → 0
    # Whether the dominant unbounded term is 1st-order infinity or higher
    # doesn't matter for limit purposes — both mean "diverges in that sign".
    pos_dims = {d: c for d, c in h._d.items() if d > 0}
    if pos_dims:
        leading_coeff = pos_dims[max(pos_dims)]
        if leading_coeff > 0:
            # exp dominates every polynomial: place it at a dimension higher
            # than any finite-order polynomial will reach. `terms` is the
            # Taylor truncation bound, which effectively sets the ceiling of
            # polynomial dims the user's code can produce.
            return Composite({terms: math.exp(a)})   # → +∞ (exp-dominant)
        return Composite({})                          # → 0

    # Only infinitesimal components: Taylor series converges.
    # Derivative identity `exp' = exp` gives the recurrence  t_n = t_{n-1} · h / n
    # starting from t_0 = 1.  One composite op per term instead of two.
    base = math.exp(a)
    term = Composite({0: 1.0})
    exp_h = Composite({0: 1.0})
    for n in range(1, terms):
        term = (term * h) / n
        exp_h = exp_h + term
    return base * exp_h


def ln(x, terms=15):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.log(float(x))
    if _is_nothing(x):
        return Composite({})

    coeffs = x._d
    if not coeffs:
        raise ValueError("ln of empty composite")

    dims_sorted = sorted(coeffs.keys(), reverse=True)
    top_dim = dims_sorted[0]
    top_coeff = coeffs[top_dim]

    # Structurally unbounded (input → +∞): ln(+∞) = +∞. Lossy fallback —
    # embed the real-valued answer as a nonfinite coefficient at dim 1 so
    # downstream `__mul__` can detect 0·∞ cancellations and refuse.
    if top_dim > 0:
        if top_coeff > 0:
            return Composite({1: math.inf})      # ln(+∞) = +∞ (lossy)
        raise ValueError("ln requires positive input")   # -∞ → log undef

    # Standard: finite positive standard part → Taylor around `a`.
    # Derivative identity ln' = 1/x gives:
    #   ln(a + h) = ln(a) + h/a - h²/(2a²) + h³/(3a³) - ...
    # Recurrence on `(h/a)` powers: term_k = term_{k-1} · (h/a).
    # `math.log(a)` is the single float rounding at dim 0; all correction
    # coefficients 1, -1/2, 1/3, -1/4, ... are exact rationals.
    a = x.st()
    if a > 0:
        ratio = (x - R(a)) / R(a)                     # h / a
        result = Composite({0: math.log(a)})
        term = Composite({0: 1.0})
        for n in range(1, terms):
            term = term * ratio                        # (h/a)^n
            sign = 1 if n % 2 == 1 else -1             # alternating, + at n=1
            result = result + (sign / n) * term
        return result

    # a == 0 with top_dim ≤ 0: input → 0⁺ (if leading infinitesimal positive).
    # ln(0⁺) = -∞. Lossy fallback at dim 1.
    if a == 0:
        neg_dims = {d: c for d, c in coeffs.items() if d < 0 and abs(c) > 1e-15}
        if neg_dims:
            top_neg = neg_dims[max(neg_dims)]
            if top_neg > 0:
                return Composite({1: -math.inf})   # ln(0⁺) = -∞ (lossy)
        raise ValueError("ln requires positive input")

    raise ValueError("ln requires positive input")


def expm1(x, terms=15):
    """exp(x) − 1, numerically stable for small x.

    Decomposes as `expm1(a+h) = expm1(a) + exp(a)·(exp(h) − 1)`. The leading
    `1` in the Taylor series never materializes: the series for `exp(h)−1`
    starts at `h` itself (dim −1 for an infinitesimal probe), skipping the
    dim-0 cancellation entirely. For plain floats, delegates to `math.expm1`.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.expm1(float(x))
    if _is_nothing(x):
        return Composite({})

    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0 and abs(c) > 1e-15})

    # Asymptotic: input → +∞ → expm1 → +∞ (exp-dominant); input → −∞ → expm1 → −1.
    pos = {d: c for d, c in h._d.items() if d > 0}
    if pos:
        if pos[max(pos)] > 0:
            return Composite({terms: math.exp(a)})
        return Composite({0: -1.0})

    if not h._d:
        return Composite({0: math.expm1(a)})

    # series(h) = h + h²/2 + h³/6 + … — starts at `h` itself, no "1".
    term = h
    series = Composite(dict(h._d))
    for n in range(2, terms):
        term = (term * h) / n
        series = series + term

    base_expm1 = math.expm1(a)
    base_exp = math.exp(a)
    result = base_exp * series
    if base_expm1 != 0:
        result = result + Composite({0: base_expm1})
    return result


def _step_function(x, real_fn, integer_right, integer_left):
    """Shared machinery for `floor` and `ceil` on composites.

    Both are piecewise-constant real functions. At a non-integer `a = st(x)`
    they're locally constant; at an integer they jump — the direction of
    approach decides which side. Composite arithmetic already carries the
    direction in the sign of the leading negative-dim coefficient.
    """
    if isinstance(x, (int, float)):
        return real_fn(float(x))
    if _is_nothing(x):
        return Composite({})

    coeffs = x._d
    if not coeffs:
        return Composite({})

    # Positive-dim (structurally infinite) input: floor/ceil of ±∞ → ±∞.
    pos = {d: c for d, c in coeffs.items() if d > 0 and abs(c) > 1e-15}
    if pos:
        leading = pos[max(pos)]
        return Composite({1: math.inf if leading > 0 else -math.inf})

    a = x.st()
    # Non-integer standard part: locally constant, no direction needed.
    if a != int(a):
        return Composite({0: float(real_fn(a))})

    # At an integer: direction of infinitesimal decides the side.
    neg = {d: c for d, c in coeffs.items() if d < 0 and abs(c) > 1e-15}
    if not neg:
        # Exact integer, no perturbation — floor(k) = ceil(k) = k.
        return Composite({0: float(a)})

    top_neg = max(neg)
    if neg[top_neg] > 0:
        return Composite({0: float(integer_right(a))})
    return Composite({0: float(integer_left(a))})


def floor(x):
    """Largest integer ≤ x, composite-aware.

    At a non-integer, locally constant. At an integer k, direction matters:
    `floor(k + 0⁺) = k`, `floor(k − 0⁺) = k − 1`.
    """
    return _step_function(x, math.floor,
                          integer_right=lambda a: int(a),
                          integer_left=lambda a: int(a) - 1)


def ceiling(x):
    """Smallest integer ≥ x, composite-aware.

    Dual to `floor`: `ceil(k + 0⁺) = k + 1`, `ceil(k − 0⁺) = k`.
    """
    return _step_function(x, math.ceil,
                          integer_right=lambda a: int(a) + 1,
                          integer_left=lambda a: int(a))


def frac(x):
    """Fractional part: `frac(x) = x − floor(x)`.

    At non-integer x, a smooth "sawtooth" locally. At integer k, jumps
    from 1 (left limit) to 0 (right limit). Directional handling flows
    from `floor`'s direction-aware behavior.
    """
    if isinstance(x, (int, float)):
        return float(x) - math.floor(float(x))
    return x - floor(x)


def cbrt(x):
    """Real cube root, defined for negative inputs (unlike `x**(1/3)`).

    For positive st: defers to `__pow__` with exponent 1/3.
    For negative st: uses reflection `cbrt(-y) = -cbrt(y)`.
    For composite ±∞: cube root of coefficient at same dim (if exact).
    """
    if isinstance(x, (int, float)):
        v = float(x)
        if v >= 0:
            return v ** (1.0 / 3.0)
        return -((-v) ** (1.0 / 3.0))
    if _is_nothing(x):
        return Composite({})
    a = x.st()
    if a >= 0:
        return x ** (1.0 / 3.0)
    # Reflect through -1 to stay on the real branch.
    return Composite({0: -1.0}) * ((Composite({0: -1.0}) * x) ** (1.0 / 3.0))


def Mod(x, n):
    """Mathematical modulo: `Mod(x, n) = x − n·floor(x/n)`.

    Uses the composite-aware `floor`, so directional behavior at boundaries
    flows through automatically.
    """
    if isinstance(x, (int, float)) and isinstance(n, (int, float)):
        return float(x) - float(n) * math.floor(float(x) / float(n))
    return x - n * floor(x / n)


# ---------------------------------------------------------------------------
# Error-function family and Fresnel integrals.
# All are defined by Taylor series with simple coefficient recurrences.
# Positive-dim inputs (structural ±∞) saturate to the known limits.
# ---------------------------------------------------------------------------

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


def _erf_like_series(x, sign_pattern, terms, scale=_TWO_OVER_SQRT_PI):
    """Shared Taylor series for erf / erfi.

    erf(x)  = (2/√π) · Σ  (-1)^n · x^(2n+1) / (n! · (2n+1))
    erfi(x) = (2/√π) · Σ          x^(2n+1) / (n! · (2n+1))

    `sign_pattern('alt')` toggles the alternating sign; `sign_pattern('plus')`
    keeps everything positive. Coefficient recurrence:
        c_n = c_{n-1} · (±1) · (2n-1) / (n · (2n+1))
    Actually simpler: `term = term · x² / n`, and coefficient = c_{n-1} · ±1 · (2n-1)/(2n+1).
    """
    x_sq = x * x
    term = x                           # c_0 · x^1 = x
    # first term's coefficient is 1 (so result_0 = scale · x)
    result = scale * term
    coeff = 1.0
    for n in range(1, terms):
        term = (term * x_sq) / n       # accumulates x^(2n+1), includes 1/n!
        if sign_pattern == 'alt':
            coeff = -coeff             # (-1)^n
        # series factor is 1/(2n+1) per term
        result = result + (scale * coeff / (2 * n + 1)) * term
    return result


def _saturate(x, plus_val, minus_val):
    """For a structurally-infinite input, return the saturation value.

    Zero saturations use `ZERO = |1|₋₁` (structural infinitesimal) rather
    than `|0|₀` — the latter gets pruned to an empty composite which the
    limit evaluator interprets as "indeterminate" and tries to recover
    numerically.
    """
    pos = {d: c for d, c in x._d.items() if d > 0 and abs(c) > 1e-15}
    val = plus_val if pos[max(pos)] > 0 else minus_val
    if val == 0:
        return Composite({-1: 1.0})                # structural zero
    return Composite({0: val})


def erf(x, terms=15):
    """Error function: erf(x) = (2/√π) ∫₀ˣ e^(−t²) dt.

    `erf(+∞) = 1`, `erf(−∞) = −1`. Direct Taylor series around 0 — converges
    everywhere; for finite `x` the convergence is fast when |x| ≲ 3.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.erf(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _saturate(x, 1.0, -1.0)
    return _erf_like_series(x, 'alt', terms)


def erfc(x, terms=15):
    """Complementary error function: `erfc(x) = 1 − erf(x)`."""
    if isinstance(x, (int, float)):
        return math.erfc(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        # erfc(+∞) = 0, erfc(−∞) = 2
        return _saturate(x, 0.0, 2.0)
    return Composite({0: 1.0}) - erf(x, terms)


def erfi(x, terms=15):
    """Imaginary error function: erfi(x) = (2/√π) ∫₀ˣ e^(+t²) dt.

    `erfi(+∞) = +∞`, `erfi(−∞) = −∞`. Same series as `erf` but with no sign
    alternation.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        # No stdlib erfi; compute via series for small |x|, saturate for large.
        if abs(x) < 1e-300:
            return 0.0
        return _erfi_float(float(x), 50)
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        # erfi(+∞) = +∞ (lossy); erfi(−∞) = −∞
        pos = {d: c for d, c in x._d.items() if d > 0 and abs(c) > 1e-15}
        return Composite({1: math.inf if pos[max(pos)] > 0 else -math.inf})
    return _erf_like_series(x, 'plus', terms)


def _erfi_float(x, terms):
    """erfi(x) for plain float, direct Taylor series around 0."""
    factor = _TWO_OVER_SQRT_PI
    x_sq = x * x
    total = x
    term = x
    for n in range(1, terms):
        term = term * x_sq / n
        total = total + term / (2 * n + 1)
    return factor * total


_HALF_PI = math.pi / 2.0


def fresnels(x, terms=15):
    """Fresnel S: S(x) = ∫₀ˣ sin(π t² / 2) dt.

    Series: `S(x) = Σ (-1)^n · (π/2)^(2n+1) · x^(4n+3) / ((2n+1)! · (4n+3))`.
    Saturates to ±1/2 at ±∞.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return _fresnels_float(float(x), 40)
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _saturate(x, 0.5, -0.5)
    # General composite: Taylor around 0. x^(4n+3) built incrementally.
    x_sq = x * x                       # x²
    x_cubed = x * x_sq                 # x³ (leading term)
    term = x_cubed                     # x^(4·0+3)
    scale = _HALF_PI / 3.0             # (π/2)^1 / ((2·0+1)! · (4·0+3)) = π/6
    result = scale * term
    # Recurrence: going from x^(4n+3) to x^(4(n+1)+3) multiplies by x⁴,
    # and the coefficient multiplies by  (-1)·(π/2)² / ((2n+2)(2n+1)) · (4n+3)/(4n+7)
    coeff = scale
    x4 = x_sq * x_sq                   # x⁴
    for n in range(1, terms):
        factor = (-(_HALF_PI * _HALF_PI)) / ((2 * n) * (2 * n + 1))
        factor *= (4 * n - 1) / (4 * n + 3)
        coeff = coeff * factor
        term = term * x4
        result = result + coeff * term
    return result


def fresnelc(x, terms=15):
    """Fresnel C: C(x) = ∫₀ˣ cos(π t² / 2) dt.

    Series: `C(x) = Σ (-1)^n · (π/2)^(2n) · x^(4n+1) / ((2n)! · (4n+1))`.
    Saturates to ±1/2 at ±∞.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return _fresnelc_float(float(x), 40)
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _saturate(x, 0.5, -0.5)
    x_sq = x * x
    term = x                           # x^(4·0+1) = x
    coeff = 1.0                        # n=0: (π/2)^0 / (0! · 1) = 1
    result = coeff * term
    x4 = x_sq * x_sq
    for n in range(1, terms):
        factor = (-(_HALF_PI * _HALF_PI)) / ((2 * n - 1) * (2 * n))
        factor *= (4 * n - 3) / (4 * n + 1)
        coeff = coeff * factor
        term = term * x4
        result = result + coeff * term
    return result


def _fresnels_float(x, terms):
    """S(x) for plain float via Taylor series."""
    if x == 0:
        return 0.0
    x_sq = x * x
    x_cubed = x * x_sq
    term = x_cubed
    coeff = _HALF_PI / 3.0
    total = coeff * term
    x4 = x_sq * x_sq
    for n in range(1, terms):
        factor = (-(_HALF_PI * _HALF_PI)) / ((2 * n) * (2 * n + 1))
        factor *= (4 * n - 1) / (4 * n + 3)
        coeff = coeff * factor
        term = term * x4
        total = total + coeff * term
    return total


def gamma(x):
    """Γ(z), with composite-aware pole handling.

    Γ has simple poles at 0, −1, −2, …, with residue `(−1)ⁿ/n!` at −n.
    Near a pole, `Γ(−n + h) ~ (−1)ⁿ / (n! · h)` — a structural ∞ scaled by
    the residue coefficient. At regular points we return only the dim-0
    value (Γ(a)) — the full Taylor expansion would require digamma /
    polygamma functions, which stdlib doesn't provide. For most limit
    evaluations `Γ(a + small h) ≈ Γ(a)` at dim 0 suffices.
    """
    if isinstance(x, (int, float)):
        return math.gamma(float(x))
    if _is_nothing(x):
        raise CompositionError("gamma of empty composite is undefined")
    if _has_positive_dims(x):
        pos = {d: c for d, c in x._d.items() if d > 0 and abs(c) > 1e-15}
        leading = pos[max(pos)]
        if leading > 0:
            # Γ(+∞) is a scalar-infinite value, not a structural dim-1 ∞
            # (Principle 3: no ×0/×∞ happened). Placing it at dim 0 makes
            # downstream `__pow__` trigger its scalar-∞ refusal and fall
            # back to numerical extrapolation for Stirling-like cases
            # (e.g. `n / factorial(n)^(1/n) → e`).
            return Composite({0: math.inf})
        raise CompositionError("gamma(−∞) is oscillatory, undefined")

    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0 and abs(c) > 1e-15})

    # Pole at non-positive integer: Γ(−n + h) ~ (−1)ⁿ / (n! · h).
    if a == int(a) and a <= 0:
        n = -int(a)
        residue = ((-1) ** n) / math.factorial(n)
        if not h._d:
            raise CompositionError("gamma at exact pole (integer ≤ 0)")
        # h's sign determines direction into the pole.
        neg = {d: c for d, c in h._d.items() if d < 0 and abs(c) > 1e-15}
        top_neg = neg[max(neg)] if neg else 0.0
        sign = 1.0 if top_neg * residue > 0 else -1.0
        return Composite({1: sign * math.inf})

    # Regular point: base value at dim 0.
    return Composite({0: math.gamma(a)})


def factorial(n):
    """n! for non-negative integer n, Γ(n+1) otherwise.

    Returns `math.inf` for n > 170 to stay representable in double precision
    (171! already overflows). Callers can then detect overflow as a finite
    inf rather than triggering OverflowError.
    """
    if isinstance(n, (int, float)):
        if n == int(n) and n >= 0:
            ni = int(n)
            if ni > 170:
                raise OverflowError("factorial exceeds double precision")
            return float(math.factorial(ni))
        return math.gamma(float(n) + 1.0)   # may raise OverflowError
    # Composite: Γ(n + 1)
    return gamma(n + Composite({0: 1.0}))


def binomial(n, k):
    """Generalized binomial coefficient C(n, k) = Γ(n+1) / (Γ(k+1)·Γ(n-k+1))."""
    if isinstance(n, (int, float)) and isinstance(k, (int, float)):
        ni, ki = int(n), int(k)
        if (n == ni and k == ki and 0 <= ki <= ni):
            return math.comb(ni, ki)
        return (math.gamma(float(n) + 1.0)
                / (math.gamma(float(k) + 1.0)
                   * math.gamma(float(n) - float(k) + 1.0)))
    one = Composite({0: 1.0})
    return gamma(n + one) / (gamma(k + one) * gamma(n - k + one))


def _fresnelc_float(x, terms):
    """C(x) for plain float via Taylor series."""
    if x == 0:
        return 0.0
    x_sq = x * x
    term = x
    coeff = 1.0
    total = coeff * term
    x4 = x_sq * x_sq
    for n in range(1, terms):
        factor = (-(_HALF_PI * _HALF_PI)) / ((2 * n - 1) * (2 * n))
        factor *= (4 * n - 3) / (4 * n + 1)
        coeff = coeff * factor
        term = term * x4
        total = total + coeff * term
    return total


def cosm1(x, terms=12):
    """cos(x) − 1, numerically stable for small x.

    Decomposes as `cosm1(a+h) = (cos(a)−1) + cos(a)·(cos(h)−1) − sin(a)·sin(h)`.
    The leading `1` from `cos(h)`'s Taylor series never materializes:
    `cos(h) − 1 = −h²/2 + h⁴/24 − …` starts at dim −2 (for infinitesimal h).
    No stdlib counterpart; useful for `(1 − cos(x)) / x² → 1/2` patterns.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        # Stable small-x form: cos(x) - 1 = -2·sin(x/2)²
        return -2.0 * math.sin(float(x) / 2.0) ** 2
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        # cos oscillates at infinity → AccumBounds-like; -1 offset is finite.
        inner = _bounded_at_inf(math.cos, x)
        return inner - Composite({0: 1.0})

    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0 and abs(c) > 1e-15})
    if not h._d:
        return Composite({0: -2.0 * math.sin(a / 2.0) ** 2})

    # sin_h = h − h³/6 + …  (starts at dim −1)
    # cosm1_h = cos(h) − 1 = −h²/2 + h⁴/24 − …  (starts at dim −2)
    sin_h = Composite({})
    cosm1_h = Composite({})
    term = Composite({0: 1.0})
    for n in range(1, terms):
        term = (term * h) / n
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (term if sign > 0 else -term)
        else:
            sign = (-1) ** (n // 2)
            cosm1_h = cosm1_h + (term if sign > 0 else -term)

    # cosm1(a+h) = (cos(a)-1) + cos(a)·cosm1_h - sin(a)·sin_h
    # Skip zero-coefficient terms to avoid scalar-0 uplift in __mul__
    base_cosm1 = -2.0 * math.sin(a / 2.0) ** 2
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    result = Composite({})
    if cos_a != 0:
        result = result + cos_a * cosm1_h
    if sin_a != 0:
        result = result - sin_a * sin_h
    if base_cosm1 != 0:
        result = result + Composite({0: base_cosm1})
    return result


def log1p(x, terms=15):
    """ln(1 + x), numerically stable for small x.

    Uses `log1p(a+h) = log1p(a) + sum_{n≥1} (-1)^{n+1} · (h/(1+a))^n / n`.
    `math.log1p(a)` is the single float rounding at dim 0. For `a = 0` the
    leading `ln(1)` term vanishes and the series starts at `h` itself.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.log1p(float(x))
    if _is_nothing(x):
        return Composite({})

    a = x.st()
    if 1 + a <= 0:
        raise ValueError("log1p requires 1 + x > 0")

    h = Composite({d: c for d, c in x._d.items() if d != 0 and abs(c) > 1e-15})

    # Asymptotic: x → +∞ → log1p → +∞ (lossy); x → −1⁺ separately.
    pos = {d: c for d, c in h._d.items() if d > 0}
    if pos:
        if pos[max(pos)] > 0:
            return Composite({1: math.inf})   # lossy +∞ marker
        raise ValueError("log1p of argument tending to −∞")

    if not h._d:
        return Composite({0: math.log1p(a)})

    ratio = h / R(1.0 + a)                     # h / (1+a)
    result = Composite({0: math.log1p(a)}) if a != 0 else Composite({})
    term = Composite({0: 1.0})
    for n in range(1, terms):
        term = term * ratio                    # (h/(1+a))^n
        sign = 1 if n % 2 == 1 else -1
        result = result + (sign / n) * term
    return result


def sqrt(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.sqrt(float(x))
    if _is_nothing(x):
        return Composite({})

    coeffs = x._d
    if not coeffs:
        raise ValueError("sqrt of empty composite")

    dims_sorted = sorted(coeffs.keys(), reverse=True)
    top_dim = dims_sorted[0]
    top_coeff = coeffs[top_dim]

    if top_dim > 0 and top_coeff > 0:
        if top_dim % 2 != 0:
            raise CompositionError(
                "sqrt is not composable with odd-dim infinite composite input")
        result_dim = top_dim // 2
        sqrt_top = math.sqrt(top_coeff)
        if len(coeffs) == 1:
            return Composite({result_dim: sqrt_top})
        dominant = Composite({top_dim: top_coeff})
        ratio_comp = (x - dominant) / dominant
        one_plus = Composite({0: 1.0}) + ratio_comp
        sqrt_1h = _sqrt_binomial(one_plus, 1.0, terms)
        return Composite({result_dim: sqrt_top}) * sqrt_1h

    if top_dim == 0:
        a = x.st()
        if a > 0:
            return _sqrt_binomial(x, a, terms)
        raise ValueError("sqrt requires positive standard part")

    # Negative dims (infinitesimal) — same rule: halve the dimension
    if top_coeff > 0:
        if top_dim % 2 != 0:
            raise CompositionError(
                "sqrt is not composable with odd-dim infinitesimal composite input")
        result_dim = top_dim // 2  # -2 → -1, -4 → -2
        sqrt_top = math.sqrt(top_coeff)
        if len(coeffs) == 1:
            return Composite({result_dim: sqrt_top})
        # Multi-term: factor out dominant, binomial expand remainder
        dominant = Composite({top_dim: top_coeff})
        ratio_comp = (x - dominant) / dominant
        one_plus = Composite({0: 1.0}) + ratio_comp
        sqrt_1h = _sqrt_binomial(one_plus, 1.0, terms)
        return Composite({result_dim: sqrt_top}) * sqrt_1h
    raise ValueError("sqrt requires positive coefficient")


def _sqrt_binomial(x, a, terms):
    """sqrt(a + h) via generalized binomial series with derivative recurrence.

    Identity:  sqrt(a + h) = sqrt(a) · (1 + h/a)^(1/2)
    Binomial coefficients for exponent α = 1/2 satisfy the recurrence
        c_n = c_{n-1} · (α - (n-1)) / n
    so  c_0 = 1, c_1 = 1/2, c_2 = -1/8, c_3 = 1/16, ...
    `math.sqrt(a)` is the single float rounding at dim 0; all correction
    coefficients are exact rationals.
    """
    sqrt_a = math.sqrt(a)
    ratio = (x - R(a)) / R(a)                 # h / a
    result = Composite({0: sqrt_a})
    coeff = 1.0                                 # binomial c_0
    term = Composite({0: 1.0})                  # (h/a)^0
    for n in range(1, terms):
        coeff = coeff * (0.5 - (n - 1)) / n     # c_n = c_{n-1} · (α-(n-1))/n
        term = term * ratio                     # (h/a)^n
        result = result + (sqrt_a * coeff) * term
    return result


def tan(x, terms=12):
    if isinstance(x, (int, float)):
        return math.tan(float(x))
    if _is_nothing(x):
        return Composite({})
    return sin(x, terms) / cos(x, terms)


def atan(x, terms=15):
    """Arctangent via Taylor expansion: atan(a+h) = atan(a) + sum of h-powers.

    Uses direct h-power expansion (not derivative integration) to
    preserve sign of h. The nth derivative of atan at a is computed
    from the recurrence of 1/(1+a^2) derivatives.
    """
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.atan(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.atan, x)
    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0})
    if not h._d:
        return R(math.atan(a))
    # Taylor: atan(a+h) = atan(a) + sum_{n=1}^{terms} (atan^(n)(a)/n!) * h^n
    # Compute derivatives of atan at a via 1/(1+x^2) derivative tower
    # d/dx atan = 1/(1+x^2). Higher derivatives via composite evaluation.
    t = Composite({0: a, -1: 1.0})  # t = a + epsilon
    deriv_tower = _reciprocal(R(1) + t * t, terms)
    # deriv_tower.coeff(-k) = (1/(1+x^2))^(k) / k! evaluated at a
    # These are the derivatives of atan' = 1/(1+x^2), so atan^(n+1)(a)/(n+1)!
    # We need atan^(n)(a)/n!, which is the antiderivative shift
    result = Composite({0: math.atan(a)})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        # coeff at dim -n in deriv_tower = f^(n)(a)/n! where f = 1/(1+x^2)
        # atan^(n+1)(a)/(n+1)! = f^(n)(a)/n! ... no, atan'=f, so atan^(n+1)=f^(n)
        # We want atan^(n)(a)/n! = f^(n-1)(a)/(n-1)! / n = deriv_tower.coeff(-(n-1)) / n
        if n == 1:
            cn = 1.0 / (1 + a*a)  # atan'(a)
        else:
            cn = deriv_tower.coeff(-(n-1))  # f^(n-1)(a)/(n-1)!
            cn = cn / n  # atan^(n)(a)/n!
        if cn != 0:
            result = result + cn * h_power
    return result


def asin(x, terms=15):
    """Arcsine via Taylor expansion through h, preserving sign."""
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.asin(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.asin, x)
    a = x.st()
    if abs(a) >= 1:
        raise ValueError("asin requires |standard part| < 1")
    h = Composite({d: c for d, c in x._d.items() if d != 0})
    if not h._d:
        return R(math.asin(a))
    # asin'(x) = 1/sqrt(1-x^2). Compute derivative tower at a.
    t = Composite({0: a, -1: 1.0})
    inner = R(1) - t * t
    deriv_tower = _reciprocal(sqrt(inner, terms), terms)
    result = Composite({0: math.asin(a)})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        if n == 1:
            cn = 1.0 / math.sqrt(1 - a*a)
        else:
            cn = deriv_tower.coeff(-(n-1)) / n
        if cn != 0:
            result = result + cn * h_power
    return result


def acos(x, terms=15):
    if isinstance(x, (int, float)):
        return math.acos(float(x))
    if _is_nothing(x):
        return Composite({})
    return R(math.pi / 2) - asin(x, terms)


def sinh(x, terms=15):
    if isinstance(x, (int, float)):
        return math.sinh(float(x))
    if _is_nothing(x):
        return Composite({})
    return (exp(x, terms) - exp(-x, terms)) / 2


def cosh(x, terms=15):
    if isinstance(x, (int, float)):
        return math.cosh(float(x))
    if _is_nothing(x):
        return Composite({})
    return (exp(x, terms) + exp(-x, terms)) / 2


def tanh(x, terms=15):
    if isinstance(x, (int, float)):
        return math.tanh(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.tanh, x)
    return sinh(x, terms) / cosh(x, terms)


def _reciprocal(x, terms=15):
    a = x.st()
    if abs(a) < 1e-14:
        raise ZeroDivisionError("Cannot compute 1/x at x=0")
    h_part = x - R(a)
    ratio = h_part / R(-a)
    result = Composite({0: 1/a})
    power = Composite({0: 1.0})
    for n in range(1, terms):
        power = power * ratio
        result = result + power / a
    return result


# Alias
log = ln
