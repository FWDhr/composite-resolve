# Known Limitations

## What works

composite-resolve handles:

- All removable singularities (0/0 forms) — exact
- All seven indeterminate forms: 0/0, 0*inf, inf-inf, 0^0, 1^inf, inf^0, inf/inf
- Higher-order cancellation up to ~20th order (configurable via `truncation`)
- Limits at +/- infinity
- One-sided limits with direction detection
- Singularity classification (Regular, Removable, Pole, Essential)
- Taylor coefficient extraction to arbitrary order
- Functions using `math` or `numpy` transcendentals
- Compositions of arbitrary depth (sin(sin(sin(x)))/x, etc.)

## What doesn't work

### Fractional dimensions

Functions involving `sqrt(x)` or fractional powers at x=0 have limited support. The underlying composite arithmetic uses integer dimensions. `sqrt` of an even-dimension infinitesimal works (dim -2 halves to dim -1), but odd dimensions (dim -1 halves to dim -0.5) cannot be represented.

**Works:**
```python
resolve(lambda x: x * sqrt(x), at=0)         # → 0.0
resolve(lambda x: sqrt(x**2 + x) - x, at=0)  # → 0.5 (via infinity)
```

**Limited:**
```python
resolve(lambda x: sqrt(x), at=0)  # Returns ZERO (st=0), correct limit
                                    # but intermediate is dim -1, not dim -0.5
```

### Float precision at irrational points

`math.pi/2` is not exactly pi/2 in IEEE 754. Functions with singularities at irrational points may not be detected:

```python
classify(lambda x: tan(x), at=math.pi/2)
# Returns Removable (huge value) instead of Pole
# Because math.tan(math.pi/2) ≈ 1.6e16, not infinity
```

### ln(x) at 0

`ln(ZERO)` returns ZERO (via the coefficient evaluation approach). This makes `x*ln(x) → 0` and `x^x → 1` work exactly, but `ln(x)` alone at 0 returns 0 instead of -infinity. The true fix requires fractional dimensions.

```python
limit(lambda x: x * log(x), to=0, dir="+")    # → 0.0 (correct)
limit(lambda x: x**x, to=0, dir="+")           # → 1.0 (correct)
limit(lambda x: log(x), to=0, dir="+")         # → 0.0 (should be -inf)
```

### Thread safety

`limit()`, `resolve()`, and `@safe` temporarily patch the `math` module during evaluation. This is not thread-safe. Concurrent calls from different threads may corrupt each other's math functions.

For concurrent use:
- Import from `composite_resolve.math` directly
- Or use `numpy` functions (dispatched via `__array_ufunc__`, thread-safe)

### Non-existent limits that look convergent

Functions with infinitely dense poles near the limit point may not be detected as non-existent:

```python
# x/sin(1/x) has poles at x = 1/(n*pi)
# Probes may miss the poles and report convergence
limit(lambda x: x/sin(1/x), to=0)  # May return 0.0 instead of raising
```

### `from math import *` breaks resolution

`from math import sin` captures `math.sin` as a local name. When `resolve()` patches the `math` module during evaluation, the captured name still points to the original — the patch doesn't reach it.

```python
from math import sin    # captures original math.sin

@safe
def sinc(x):
    return sin(x) / x   # uses the captured math.sin, not the patched one

sinc(0)  # → 0.9999999999999983 (approximate, not exact)
```

**Fix:** use `import math` (module lookup, patchable) or `from composite_resolve.math import sin` (already composite-aware):

```python
import math

@safe
def sinc(x):
    return math.sin(x) / x  # module lookup — patching works

sinc(0)  # → 1.0 (exact)
```

### Unsupported math libraries

Only `math` and `numpy` transcendentals are intercepted. Functions using `jax.numpy`, `torch`, `scipy.special`, or other libraries will raise `CompositionError`.

### Single variable only

v1.0 supports single-variable functions only. Multivariate limits (f(x,y) as (x,y) → (0,0)) are not supported.

## Error behavior summary

| Situation | Behavior |
|---|---|
| Removable singularity | Returns exact value |
| Pole (one-sided) | Raises `LimitDivergesError` with `.value = ±inf` |
| Pole (both sides, same direction) | Raises `LimitDivergesError` |
| Pole (both sides, different) | Raises `LimitDoesNotExistError` |
| Oscillatory (sin(1/x)) | Raises `LimitDoesNotExistError` |
| Non-composable function | Raises `CompositionError` |
| Domain error (ln(0), sqrt(-x)) | Attempts extrapolation, raises `LimitUndecidableError` if fails |
| Regular point (no singularity) | Returns function value directly |
