# API Reference

## Functions

### `resolve(f, at=0, dir="both", truncation=20) → float`

Evaluate a function at a point where it would normally fail.

**Parameters:**
- `f` — callable, `f(x) → number`. Can use `math.sin`, `numpy.sin`, etc.
- `at` — evaluation point. `float`, `int`, `math.inf`, or `-math.inf`.
- `dir` — direction: `"both"` (default), `"+"` (from right), `"-"` (from left).
- `truncation` — maximum Taylor series order (default 20).

**Returns:** `float`. Returns `math.inf` or `-math.inf` for divergent limits.

**Raises:** `LimitDoesNotExistError` if one-sided limits disagree or limit is oscillatory.

```python
resolve(lambda x: math.sin(x) / x, at=0)          # → 1.0
resolve(lambda x: (x**2 - 1) / (x - 1), at=1)     # → 2.0
resolve(lambda x: 1/x, at=0, dir="+")              # → inf
resolve(lambda x: 1/x**2, at=0)                    # → inf
```

---

### `limit(f, to=0, dir="both", truncation=20) → float`

Compute the limit of `f(x)` as `x → to`.

Same parameters as `resolve()`, except the target is named `to` instead of `at`.

**Raises:**
- `LimitDivergesError` — limit is +/- infinity. Access `.value` for the direction.
- `LimitDoesNotExistError` — limit does not exist. Access `.left_limit` and `.right_limit` when one-sided limits disagree.
- `CompositionError` — function cannot be evaluated with composite arithmetic.

```python
limit(lambda x: (1 + x)**(1/x), to=0)              # → 2.718281828459045
limit(lambda x: x * math.log(x), to=0, dir="+")     # → 0.0
limit(lambda x: (1 + 1/x)**x, to=math.inf)           # → e
limit(lambda x: 1/x, to=0, dir="+")                  # raises LimitDivergesError
limit(lambda x: 1/x, to=0)                           # raises LimitDoesNotExistError
```

The difference between `resolve()` and `limit()`: `resolve()` returns `inf`/`-inf` for divergent limits. `limit()` raises `LimitDivergesError`. Use `resolve()` when you want a value regardless. Use `limit()` when you want to distinguish finite limits from divergence.

---

### `safe(f) → wrapped function`

Decorator. Wraps a function so that singularities are resolved automatically.

Normal inputs run the original function directly with zero overhead. Only when the function raises (`ZeroDivisionError`, `ValueError`, `OverflowError`) or returns `NaN`/`Inf` does the resolver activate.

```python
from composite_resolve import safe

@safe
def sinc(x):
    return math.sin(x) / x

sinc(0.5)  # → 0.9589 (normal path, no overhead)
sinc(0)    # → 1.0 (resolved)
```

The wrapped function preserves `__name__` and `__doc__` via `functools.wraps`.

---

### `evaluate(f, at=0) → float`

Strict evaluation. Returns a value only if the singularity is removable or the point is regular. Raises `SingularityError` for poles and essential singularities.

```python
evaluate(lambda x: (x**2 - 4)/(x - 2), at=2)  # → 4.0
evaluate(lambda x: math.exp(x), at=0)           # → 1.0
evaluate(lambda x: 1/x, at=0)                   # raises SingularityError
```

---

### `taylor(f, at=0, order=10) → list[float]`

Extract Taylor series coefficients around a point.

Returns `[c_0, c_1, c_2, ..., c_order]` where `f(x) ≈ c_0 + c_1*(x-a) + c_2*(x-a)^2 + ...`

- `c_0 = f(a)`
- `c_1 = f'(a) / 1!`
- `c_2 = f''(a) / 2!`
- `c_n = f^(n)(a) / n!`

```python
taylor(lambda x: math.exp(x), at=0, order=4)
# → [1.0, 1.0, 0.5, 0.16667, 0.04167]

taylor(lambda x: math.sin(x), at=0, order=4)
# → [0.0, 1.0, 0.0, -0.16667, 0.0]

taylor(lambda x: 1/(1-x), at=0, order=4)
# → [1.0, 1.0, 1.0, 1.0, 1.0]
```

---

### `classify(f, at=0, dir="both") → SingularityType`

Classify the singularity at a point.

Returns one of:

- `Regular(value)` — no singularity, function is well-defined
- `Removable(value, order, indeterminate_form)` — singularity is removable, limit exists
- `Pole(order, residue)` — function diverges
- `Essential()` — essential singularity (oscillatory, no limit)

```python
classify(lambda x: math.sin(x)/x, at=0)
# → Removable(value=1.0, order=8, form='0/0')

classify(lambda x: 1/x, at=0)
# → Pole(order=1, residue=1.0)

classify(lambda x: 1/x**2, at=0)
# → Pole(order=2, residue=0.0)

classify(lambda x: math.exp(x), at=0)
# → Regular(value=1.0)
```

---

### `residue(f, at=0) → float`

Compute the residue of a function at a pole. Raises `SingularityError` if the point is not a pole.

```python
residue(lambda x: 1/x, at=0)     # → 1.0
residue(lambda x: 3/x**2, at=0)  # → 0.0 (double pole, residue is 0)
```

---

## Math Functions

`composite_resolve.math` provides composite-aware versions of standard math functions. These are used internally by `limit()`/`resolve()` and can also be imported directly.

```python
from composite_resolve.math import sin, cos, tan, exp, log, sqrt
from composite_resolve.math import sinh, cosh, tanh, asin, acos, atan
```

Each function accepts both `float` and `Composite` inputs. Float inputs return float results. These are the same functions that `math.sin` etc. are temporarily patched to during `resolve()`/`limit()` calls.

---

## Exceptions

All exceptions inherit from `CompositeResolveError`.

### `LimitDoesNotExistError`

The limit does not exist. Raised when:
- One-sided limits disagree (access `.left_limit`, `.right_limit`)
- Function oscillates (e.g., `sin(1/x)`)
- Division by an indeterminate value

### `LimitDivergesError`

The limit is +/- infinity. Access `.value` for `math.inf` or `-math.inf`.

### `SingularityError`

Operation invalid for this singularity type. Raised by `evaluate()` for poles and `residue()` for non-poles.

### `CompositionError`

Function cannot be evaluated with composite arithmetic. Raised when the function uses unsupported operations (string concatenation, list construction, etc.) or unsupported math libraries.

---

## Types

### `Regular(value)`

No singularity. `value` is `f(at)`.

### `Removable(value, order, indeterminate_form)`

Removable singularity. `value` is the limit. `order` indicates cancellation depth. `indeterminate_form` is a string like `"0/0"`.

### `Pole(order, residue)`

Pole of given order. `residue` is the coefficient of `1/(x-a)`.

### `Essential()`

Essential singularity. No finite Laurent expansion.
