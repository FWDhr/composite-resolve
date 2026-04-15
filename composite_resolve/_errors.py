"""Exception classes for composite-resolve."""

import math


class CompositeResolveError(Exception):
    """Base exception for composite-resolve."""
    pass


class LimitDoesNotExistError(CompositeResolveError):
    """The limit does not exist (one-sided limits disagree or oscillatory)."""

    def __init__(self, message="Limit does not exist",
                 left_limit=None, right_limit=None):
        self.left_limit = left_limit
        self.right_limit = right_limit
        super().__init__(message)


class LimitDivergesError(CompositeResolveError):
    """The limit diverges to ±∞."""

    def __init__(self, value=math.inf, message=None):
        self.value = value
        if message is None:
            sign = "+∞" if value > 0 else "−∞"
            message = f"Limit diverges to {sign}"
        super().__init__(message)


class SingularityError(CompositeResolveError):
    """Operation not valid for this singularity type."""
    pass


class CompositionError(CompositeResolveError):
    """Function is not composable with Composite objects."""
    pass


class UnsupportedFunctionError(CompositeResolveError):
    """A math / numpy function called on a Composite has no composite implementation.

    Raised instead of silently coercing the Composite to its standard part,
    which would produce a numerically plausible but mathematically wrong
    answer (e.g. `math.floor(composite) → math.floor(st())`).

    The function name is available as the ``function`` attribute so callers
    can decide whether to extend the dispatch table or route around the call.
    """

    def __init__(self, function: str, message: str | None = None):
        self.function = function
        if message is None:
            message = (
                f"composite-resolve: {function!r} has no composite implementation. "
                f"Called with a Composite argument; refusing to silently coerce to a float."
            )
        super().__init__(message)
