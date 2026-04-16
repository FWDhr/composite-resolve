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

    __slots__ = ['_d']

    def __init__(self, data=None):
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
            other = Composite(other)
        result = dict(self._d)
        for d, c in other._d.items():
            result[d] = result.get(d, 0.0) + c
        return Composite(result)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Composite(other)
        result = dict(self._d)
        for d, c in other._d.items():
            result[d] = result.get(d, 0.0) - c
        # Check for exact cancellation of zero-valued composites
        vals = [v for v in result.values() if v != 0]
        if not vals and self.st() == 0 and (isinstance(other, Composite) and other.st() == 0):
            # Both zero-valued, exact cancel → shift to higher zero power
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
        """Multiplication via direct convolution."""
        if isinstance(other, (int, float)):
            if other == 0:
                return Composite({})
            return Composite({d: c * other for d, c in self._d.items()})
        if isinstance(other, Composite):
            result = {}
            for d1, c1 in self._d.items():
                for d2, c2 in other._d.items():
                    d = d1 + d2
                    result[d] = result.get(d, 0.0) + c1 * c2
            return Composite(result)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by Python zero.")
            return Composite({d: c / other for d, c in self._d.items()})
        if isinstance(other, Composite):
            if not other._d:
                raise LimitDoesNotExistError(
                    "Division by nothing (empty composite). "
                    "Denominator is indeterminate.")
            # Single-term divisor: fast path
            if len(other._d) == 1:
                div_dim, div_coeff = next(iter(other._d.items()))
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

    return Composite(result)


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

def sin(x, terms=12):
    terms = _effective_terms(terms)
    if isinstance(x, (int, float)):
        return math.sin(x) if isinstance(x, float) and not isinstance(x, bool) else math.sin(float(x))
    if _is_nothing(x):
        return Composite({})
    if _has_positive_dims(x):
        return _bounded_at_inf(math.sin, x)
    a = x.st()
    h = Composite({d: c for d, c in x._d.items() if d != 0})
    if not h._d:
        return R(math.sin(a))
    sin_a, cos_a = math.sin(a), math.cos(a)
    sin_h = Composite({})
    cos_h = Composite({0: 1.0})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    return sin_a * cos_h + cos_a * sin_h


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
    sin_h = Composite({})
    cos_h = Composite({0: 1.0})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        if n % 2 == 1:
            sign = (-1) ** ((n - 1) // 2)
            sin_h = sin_h + (sign / math.factorial(n)) * h_power
        else:
            sign = (-1) ** (n // 2)
            cos_h = cos_h + (sign / math.factorial(n)) * h_power
    return cos_a * cos_h - sin_a * sin_h


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

    # Only infinitesimal components: Taylor series converges, use it.
    base = math.exp(a)
    exp_h = Composite({0: 1.0})
    h_power = Composite({0: 1.0})
    for n in range(1, terms):
        h_power = h_power * h
        exp_h = exp_h + (1.0 / math.factorial(n)) * h_power
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

    # Structurally unbounded (input → ±∞) or infinitesimal (input → 0).
    # Integer dimensions can't faithfully represent ln's sub-polynomial
    # growth rate, so rather than pick a wrong answer or a numerical-fallback
    # patch, refuse explicitly.
    if top_dim > 0:
        raise CompositionError(
            "ln is not composable with structurally unbounded composite input")

    # Standard: finite positive standard part → Taylor expansion around st.
    a = x.st()
    if a > 0:
        h_part = x - R(a)
        ratio = h_part / R(a)
        result = Composite({0: math.log(a)})
        power = Composite({0: 1.0})
        for n in range(1, terms):
            power = power * ratio
            sign = (-1) ** (n + 1)
            result = result + sign * power / n
        return result

    if a == 0:
        raise CompositionError(
            "ln is not composable with infinitesimal composite input")

    raise ValueError("ln requires positive input")


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
    sqrt_a = math.sqrt(a)
    h_part = x - R(a)
    ratio = h_part / R(a)
    def binom(n):
        if n == 0:
            return 1
        r = 1
        for k in range(n):
            r *= (0.5 - k)
        return r / math.factorial(n)
    result = Composite({0: sqrt_a})
    power = Composite({0: 1.0})
    for n in range(1, terms):
        power = power * ratio
        result = result + binom(n) * sqrt_a * power
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
