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
