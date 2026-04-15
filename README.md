# composite-resolve

Evaluate Python functions at points where they're undefined.

The library where you pass a plain numeric Python function and get exact limits via algebraic infinitesimal arithmetic, with provenance tracking. To my knowledge, this is the first library that does this directly on plain Python functions.

Eliminates edge-case handling and numerical instability at singularities — just write the function and evaluate it everywhere.

```python
import math
from composite_resolve import safe

@safe
def sinc(x):
    return math.sin(x) / x

sinc(0.5)  # → 0.9589 (normal computation)
sinc(0)    # → 1.0 (singularity resolved)
```

Uses composite arithmetic to resolve singularities algebraically. Works with any callable that uses standard Python arithmetic and `math` module functions. No symbolic expressions, no approximation. Pure Python, zero dependencies.

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
f(1)   # → 2.0 (resolved — no ZeroDivisionError)

@safe
def entropy(p):
    return -p * math.log(p)

entropy(0.5)  # → 0.347 (normal)
entropy(0)    # → 0.0 (resolved — no ValueError)
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
limit(lambda x: x * math.log(x), to=0, dir="+")          # → 0.0    (0 * inf)
limit(lambda x: x**x, to=0, dir="+")                      # → 1.0    (0^0)
limit(lambda x: (1 + x)**(1/x), to=0)                     # → e      (1^inf)
limit(lambda x: 1/x - 1/math.sin(x), to=0)                # → 0.0    (inf - inf)

# Limits at infinity
limit(lambda x: (1 + 1/x)**x, to=math.inf)                # → e
limit(lambda x: math.sin(x) / x, to=math.inf)              # → 0.0

# One-sided limits
limit(lambda x: 1/x, to=0, dir="+")   # raises LimitDivergesError (+inf)
limit(lambda x: 1/x, to=0, dir="-")   # raises LimitDivergesError (-inf)
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

## API

### `safe(f) -> wrapped function`

Decorator. Normal inputs run `f` directly. Singularities are resolved automatically.

### `resolve(f, at, dir="both", truncation=20) -> float`

Evaluate `f` at a point where it would normally fail. Returns `math.inf` or `-math.inf` for divergent limits.

### `limit(f, to, dir="both", truncation=20) -> float`

Compute the limit of `f(x)` as `x -> to`. Raises `LimitDivergesError` for infinite limits, `LimitDoesNotExistError` when the limit doesn't exist.

### `evaluate(f, at) -> float`

Strict: only returns a value if the singularity is removable. Raises `SingularityError` otherwise.

### `taylor(f, at=0, order=10) -> list[float]`

Extract Taylor coefficients `[f(a), f'(a)/1!, f''(a)/2!, ...]`.

### `classify(f, at=0, dir="both") -> SingularityType`

Returns `Regular`, `Removable`, `Pole`, or `Essential`.

### `residue(f, at=0) -> float`

Residue at a pole.

## Examples

```python
# Evaluate a function across its full domain, including singularities
from composite_resolve import resolve

f = lambda x: (x**2 - 1) / (x - 1)
for x in range(-5, 6):
    print(f"x={x:>2d}  f(x)={resolve(f, at=x):.1f}")
# x=1 gives 2.0 — no special case needed
```

```python
# Cross-entropy loss at boundary
resolve(lambda p: -(0*math.log(p) + 1*math.log(1-p)), at=0, dir="+")  # → 0.0

# Continuous compounding
limit(lambda n: (1 + 0.05/n)**n, to=math.inf)  # → 1.05127
```

## Math Library Support

Functions can use `math`, `numpy`, or `composite_resolve.math` — all work transparently:

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

## Limitations

- Single-variable functions only
- Functions must use `math` or `numpy` transcendentals (not `jax`, `torch`, etc.)
- Not thread-safe during `limit()`/`resolve()`/`@safe` calls
- Float-precision evaluation points (e.g., `math.pi/2` is not exactly pi/2)

## License

AGPL-3.0. Commercial licensing available: tmilovan@fwd.hr

## Author

Toni Milovan — tmilovan@fwd.hr
