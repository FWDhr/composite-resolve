# composite-resolve

Evaluate Python functions at points where they're undefined.

Pass a plain numeric Python function and get exact limits via algebraic infinitesimal arithmetic. To my knowledge, this is the first library that does this directly on plain Python functions - no symbolic expressions, no approximation.

Note however, this is a mathematical correctness and continuity analysis tool, not a general execution safety mechanism. It is usefull for verifying the mathematical correctness of functions before they become production code.

It can only be used as general execution safety mechamism at runtime in some very limited and specific cases like in physics-style modeling and signal processing math (interpolation in DSP is the reason this tool emerged after all), but you have to know what you are doing in such cases.

Do not use use this as a general execution safety mechanism unless you are aware of consequences of using the limit results at the points where function is genuinely undefined.

```python
import math
from composite_resolve import safe

@safe
def sinc(x):
    return math.sin(x) / x

sinc(0.5)  # → 0.9589 (normal computation)
sinc(0)    # → 1.0 (singularity resolved)
```

Uses composite arithmetic to resolve singularities algebraically. Works with any callable that uses standard Python arithmetic and `math` module functions. Pure Python, zero dependencies.

## Install

```bash
pip install composite-resolve
```

## The `@safe` Decorator

Write your math as-is. The decorator handles singularities automatically:

```python
import math
from composite_resolve import safe

@safe
def f(x):
    return (x**2 - 1) / (x - 1)

f(3)   # → 4.0 (normal)
f(1)   # → 2.0 (resolved - no ZeroDivisionError)

@safe
def entropy(p):
    return -p * math.log(p)

entropy(0.5)  # → 0.347 (normal)
entropy(0)    # → 0.0 (resolved - no ValueError)
```

Normal inputs run the original function directly with zero overhead. Only when the function fails (ZeroDivisionError, NaN, Inf) does the resolver kick in.

## Direct API

For more control, use `resolve`, `limit`, and `classify` directly:

```python
import math
from composite_resolve import resolve, limit, classify, taylor

# Evaluate at removable singularities
resolve(lambda x: math.sin(x) / x, at=0)                # → 1.0
resolve(lambda x: (math.exp(x) - 1) / x, at=0)          # → 1.0
resolve(lambda x: (x**2 - 1) / (x - 1), at=1)           # → 2.0

# Indeterminate forms
limit(lambda x: x * math.log(x), to=0, dir="+")          # → 0.0    (0 × ∞)
limit(lambda x: x**x, to=0, dir="+")                      # → 1.0    (0⁰)
limit(lambda x: (1 + x)**(1/x), to=0)                     # → e      (1^∞)
limit(lambda x: 1/x - 1/math.sin(x), to=0)                # → 0.0    (∞ − ∞)

# Limits at infinity
limit(lambda x: (1 + 1/x)**x, to=math.inf)                # → e
limit(lambda x: math.sin(x) / x, to=math.inf)              # → 0.0

# One-sided limits
limit(lambda x: 1/x, to=0, dir="+")   # raises LimitDivergesError (+∞)
limit(lambda x: 1/x, to=0, dir="-")   # raises LimitDivergesError (-∞)
limit(lambda x: 1/x, to=0)            # raises LimitDoesNotExistError

# Singularity classification
classify(lambda x: math.sin(x)/x, at=0)   # → Removable(value=1.0)
classify(lambda x: 1/x, at=0)             # → Pole(order=1, residue=1.0)
classify(lambda x: math.exp(x), at=0)     # → Regular(value=1.0)

# Taylor coefficients
taylor(lambda x: math.exp(x), at=0, order=4)
# → [1.0, 1.0, 0.5, 0.16667, 0.04167]
```

## How It Works

The library evaluates functions using composite arithmetic. Instead of symbolic manipulation, the library substitutes a concrete algebraic infinitesimal into your function. The result carries enough structure to resolve 0/0, 0×∞, and all other indeterminate forms through ordinary arithmetic.



The function is treated as a black box. No expression tree, no symbolic manipulation.

### Evaluation layers

1. **Fast path** - `f(float(to))` evaluated directly. If the point is regular (no singularity), this returns instantly. A continuity check at `to ± ε` prevents silent mis-evaluation at discontinuities.

2. **Composite arithmetic** - the core primitive. Substitutes a seeded composite number at the limit point and evaluates `f` through the full derivative tower. Handles removable singularities, indeterminate forms, and pole detection algebraically.

3. **Numerical fallback** - when composite arithmetic hits a representation it can't handle (e.g., `ln(∞)`, `floor` at infinity, factorial overflow), falls back to evaluating `f` at nearby float probe points and extrapolating convergence/divergence.

4. **Clear refusal** - when none of the above can determine the answer, raises `LimitUndecidableError` (not `LimitDoesNotExistError`) to distinguish "we couldn't determine this" from "the limit genuinely does not exist. (I'm deliberating should I move this above numerical fallback and limit this library only to Composite resolvable limits. TBD.)"

## API

### `safe(f) -> wrapped function`

Decorator. Normal inputs run `f` directly. Singularities are resolved automatically.

### `resolve(f, at, dir="both", truncation=20) -> float`

Evaluate `f` at a point where it would normally fail. Returns `math.inf` or `-math.inf` for divergent limits.

### `limit(f, to, dir="both", truncation=20) -> float`

Compute the limit of `f(x)` as `x → to`. Raises:
- `LimitDivergesError` - limit is ±∞ (access `.value` for the sign)
- `LimitDoesNotExistError` - limit genuinely does not exist (oscillation, one-sided limits disagree). Carries evidence in `.left_limit` / `.right_limit`.
- `LimitUndecidableError` - CR could not determine the limit. The mathematical limit may still exist. Typical causes: double-precision overflow in probes, sub-polynomial growth rates the integer-dimension system can't represent, or expressions requiring log-space computation.

### `evaluate(f, at) -> float`

Strict: only returns a value if the singularity is removable. Raises `SingularityError` otherwise.

### `taylor(f, at=0, order=10) -> list[float]`

Extract Taylor coefficients `[f(a), f'(a)/1!, f''(a)/2!, ...]`.

### `classify(f, at=0, dir="both") -> SingularityType`

Returns `Regular`, `Removable`, `Pole`, or `Essential`.

### `residue(f, at=0) -> float`

Residue at a pole.

### `verify(f, var_range, points=1000) -> VerifyReport`

Scan a function's domain for singularities and verify boundary behavior. Returns a `VerifyReport` with detected singularities, their classifications, correct limit values, and whether the function handles them. Use `assert report` or `assert not report.unhandled` in tests for CI gating.

```python
report = verify(f, var_range=(0, 1))
report = verify(f, var_ranges={"x": (0,1), "y": (0,1)}, scan="x", fixed={"y": 0.5})
report = verify(f, var_ranges={"x": (0,1), "y": (0,1)}, scan="all")
```

## Supported Math Functions

`composite_resolve.math` provides composite-aware versions of 42 functions. These accept both plain floats and Composite objects - use them in functions passed to `limit()`/`resolve()` for guaranteed composite propagation, or use `math.*` / `numpy.*` which are patched automatically during evaluation.

**Core transcendentals:**
`sin` `cos` `tan` `exp` `log` `ln` `sqrt`

**Inverse trig / hyperbolic:**
`asin` `acos` `atan` `sinh` `cosh` `tanh`
`asinh` `acosh` `atanh`

**Reciprocal trig / hyperbolic:**
`cot` `sec` `csc` `coth` `sech` `csch`

**Inverse reciprocal:**
`acot` `asec` `acsc` `asech` `acsch` `acoth`

**Early-cancellation primitives** (numerically stable near cancellation points):
`expm1` (= exp(x)−1), `log1p` (= ln(1+x)), `cosm1` (= cos(x)−1)

**Step / piecewise** (direction-aware at discontinuities):
`floor` `ceil` `ceiling` `frac`

**Arithmetic:**
`cbrt` (real cube root, handles negatives), `Mod` (mathematical modulo)

**Error functions:**
`erf` `erfc` `erfi`

**Fresnel integrals:**
`fresnels` `fresnelc`

**Gamma family:**
`gamma` (with pole handling at 0, −1, −2, …), `factorial`, `binomial`

**Not yet supported:** Bessel functions (J, Y, I, K), exponential integral (Ei), Lambert W, elliptic integrals. These raise `NameError` if used in expressions passed to `limit()`.

## Examples

```python
# Evaluate a function across its full domain, including singularities
from composite_resolve import resolve

f = lambda x: (x**2 - 1) / (x - 1)
for x in range(-5, 6):
    print(f"x={x:>2d}  f(x)={resolve(f, at=x):.1f}")
# x=1 gives 2.0 - no special case needed
```

```python
# Cross-entropy loss at boundary
resolve(lambda p: -(0*math.log(p) + 1*math.log(1-p)), at=0, dir="+")  # → 0.0

# Continuous compounding
limit(lambda n: (1 + 0.05/n)**n, to=math.inf)  # → 1.05127
```

```python
# Directional limits at discontinuities
from composite_resolve.math import floor, ceiling

limit(lambda x: floor(x), to=2, dir="+")   # → 2
limit(lambda x: floor(x), to=2, dir="-")   # → 1
limit(lambda x: ceiling(x), to=2, dir="+") # → 3
```

## Boundary Verification

`verify()` scans a function's domain, finds where it breaks (raises, NaN, inf, discontinuities), classifies each singularity, computes the correct limit value, and reports whether the function already handles it. Use it during development to find boundary bugs, or in CI to prevent regressions.

```python
import math
from composite_resolve import verify

report = verify(lambda x: math.sin(x) / x, var_range=(-10, 10))
print(report)
```

```
verify(<lambda>) over (-10, 10)
  1019 points sampled in 0.002s
  1 singularity(ies) found
    x=0.0 dir=both: Removable [UNHANDLED] value=1
          function raises_ZeroDivisionError
          fix: wrap with @safe or return 1.0 at x=0.0
  FAIL: 1 unhandled singularity(ies)
```

After fixing the function (e.g. with `@safe`), the report passes:

```python
from composite_resolve import safe

@safe
def sinc(x):
    return math.sin(x) / x

report = verify(sinc, var_range=(-10, 10))
assert report  # PASS
```

### CI integration

```python
# tests/test_boundaries.py
import math
import pytest
from composite_resolve import verify

FUNCTIONS = [
    ("sinc",      lambda x: math.sin(x)/x,       (-10, 10)),
    ("entropy",   lambda p: -p * math.log(p),     (0, 1)),
    ("log_ratio", lambda x: math.log(1+x)/x,     (-0.5, 10)),
    ("exp_decay", lambda x: (math.exp(x)-1)/x,   (-5, 5)),
]

@pytest.mark.parametrize("name,fn,domain", FUNCTIONS)
def test_boundary(name, fn, domain):
    report = verify(fn, var_range=domain)
    assert not report.unhandled, f"{name}:\n{report}"
```

### Multi-variable functions

Scan one variable at a time with others held fixed:

```python
report = verify(
    lambda x, y: (x**2 - y**2) / (x - y),
    var_ranges={"x": (-5, 5), "y": (-5, 5)},
    scan="x",
    fixed={"y": 1.0},
)
# Finds removable singularity at x=1 with correct value 2.0
```

Or scan all variables one at a time (others at midpoint):

```python
report = verify(f, var_ranges={"x": (-5, 5), "y": (-5, 5)}, scan="all")
```

### What verify does NOT do

- Does not fix the function. Only reports.
- Does not check numerical stability away from singularities.
- Does not handle symbolic expressions, only plain Python callables.
- Does not prove absence of singularities, only checks sampled points.

## Math Library Support

Functions can use `math`, `numpy`, or `composite_resolve.math` - all work transparently:

```python
import math
import numpy as np
from composite_resolve import safe

@safe
def f(x):
    return math.sin(x) / x    # works

@safe
def g(x):
    return np.sin(x) / x      # also works

f(0)  # → 1.0
g(0)  # → 1.0
```

`math` functions are patched during resolution. `numpy` functions dispatch via `__array_ufunc__`.

**Important:** use `import math` not `from math import sin`. The `from` form captures the original function at import time - patching the module later doesn't reach it.

## Error Semantics

composite-resolve distinguishes three failure modes:

| Error | Meaning | User action |
|---|---|---|
| `LimitDivergesError` | Limit is ±∞ | Access `.value` for the sign |
| `LimitDoesNotExistError` | Limit genuinely doesn't exist - positive evidence (oscillation, one-sided limits disagree) | Check `.left_limit` / `.right_limit` for the one-sided values |
| `LimitUndecidableError` | CR couldn't determine the limit - no claim about existence | Try a symbolic engine (SymPy) or higher-precision tool (mpmath) |

`LimitUndecidableError` is NOT a subclass of `LimitDoesNotExistError`. They represent fundamentally different situations: evidence of non-existence vs. insufficient machinery.

## Performance

### Dimension truncation

Deeply nested function chains (like `sin(atan(sin(atan(x))))`) cause dimension counts to grow quadratically through convolution. `MAX_ACTIVE_DIMS` (default 60) caps the number of active dimensions after each multiplication, keeping the closest to dim 0. Accuracy loss is negligible (~1/60! = 10^-82) because discarded tails carry factorially tiny coefficients.

```python
import composite_resolve._core as core
core.MAX_ACTIVE_DIMS = 100  # raise for higher-order derivatives
```

### NumPy acceleration (optional)

When numpy is installed, large convolutions automatically use C-speed `np.convolve` instead of Python dict loops. For very large composites (128+ combined segment size), FFT-based convolution via `np.fft` provides O(N log N) scaling. NumPy is optional - without it, everything works identically via pure Python.

Three-tier dispatch (automatic, no configuration needed):

| Composite size | Method | Typical speedup |
|---|---|---|
| < 25 dims | Python dict loop | baseline |
| 25-127 dims | `np.convolve` (direct) | 2-5x |
| 128+ dims | `np.fft` (FFT) | 10-50x |

The internal representation stays as a sparse dict. NumPy arrays are transient - created for the convolution segment, discarded after. No storage overhead when numpy is not used.

```bash
pip install numpy  # optional, for acceleration
```

## Limitations

- **Single-variable functions only.** Multi-variable limits are out of scope.
- **Use `import math` not `from math import sin`** - the `from` form captures the original function; patching can't reach it.
- **Float-precision evaluation points.** `math.pi/2` is not exactly π/2 - limits at transcendental points may lose precision.
- **Not thread-safe** during `limit()`/`resolve()`/`@safe` calls (the `math` module is temporarily patched).
- **Integer-dimension limitation.** The composite number system uses integer dimensions. Functions whose growth rate sits between polynomial orders (like `log(x)`, which grows slower than `x^ε` for any ε > 0) can't be faithfully represented. Limits involving log/polynomial rate comparisons may fall to numerical extrapolation or raise `LimitUndecidableError`.
- **Double-precision overflow.** Numerical fallback probes are plain Python floats. Functions like `factorial(n)` overflow at n > 170; `exp(x)` at x > 709. Limits requiring probe values beyond these ranges may be undecidable.
- **Unsupported functions** (`jax`, `torch`, Bessel, Ei, Lambert W, etc.) raise `UnsupportedFunctionError` or `NameError` - not silent wrong answers.

## License

AGPL-3.0. Commercial licensing available: tmilovan@fwd.hr

## Author

Toni Milovan - tmilovan@fwd.hr
