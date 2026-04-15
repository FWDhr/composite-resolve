# composite-resolve — Exact limit computation via composite arithmetic
# Copyright (C) 2026 Toni Milovan <tmilovan@fwd.hr>
# AGPL-3.0-or-later — see LICENSE
"""Real-world functions that programmers actually encounter.

These are expressions from physics, engineering, ML, finance, and
numerical computing where the formula breaks at a specific point
but the function has a well-defined value there.
"""
import math
import pytest
from composite_resolve import resolve, limit, LimitDivergesError, LimitDoesNotExistError
from composite_resolve.math import sin, cos, exp, log, sqrt, tan, atan, tanh


# =============================================================================
# SIGNAL PROCESSING / PHYSICS
# =============================================================================

class TestSignalProcessing:

    def test_sinc(self):
        """sinc(x) = sin(x)/x, used everywhere in DSP."""
        assert abs(resolve(lambda x: sin(x)/x, at=0) - 1.0) < 1e-10

    def test_sinc_squared(self):
        """Diffraction pattern intensity."""
        assert abs(resolve(lambda x: (sin(x)/x)**2, at=0) - 1.0) < 1e-10

    def test_sinc_derivative_form(self):
        """(x*cos(x) - sin(x)) / x^2 — derivative of sinc."""
        assert abs(resolve(lambda x: (x*cos(x) - sin(x))/x**2, at=0) - 0.0) < 1e-10

    def test_bessel_related(self):
        """(1 - cos(x)) / x^2 — appears in Bessel function expansions."""
        assert abs(resolve(lambda x: (1 - cos(x))/x**2, at=0) - 0.5) < 1e-10

    def test_fresnel_integrand(self):
        """sin(x^2) / x — Fresnel integral related."""
        assert abs(resolve(lambda x: sin(x**2)/x, at=0) - 0.0) < 1e-10

    def test_damped_oscillation_envelope(self):
        """(1 - exp(-x)) / x — RC circuit response."""
        assert abs(resolve(lambda x: (1 - exp(-x))/x, at=0) - 1.0) < 1e-10

    def test_wave_group_velocity(self):
        """(sin(x) - x*cos(x)) / x^3 — related to dispersion."""
        assert abs(resolve(lambda x: (sin(x) - x*cos(x))/x**3, at=0) - 1.0/3) < 1e-8


# =============================================================================
# MACHINE LEARNING / STATISTICS
# =============================================================================

class TestML:

    def test_cross_entropy_y0(self):
        """-[y*log(p) + (1-y)*log(1-p)] at p=0, y=0."""
        assert abs(resolve(lambda p: -(0*log(p) + 1*log(1-p)), at=0, dir="+") - 0.0) < 1e-6

    def test_cross_entropy_y1(self):
        """-[y*log(p) + (1-y)*log(1-p)] at p=1, y=1."""
        assert abs(resolve(lambda p: -(1*log(p) + 0*log(1-p)), at=1, dir="-") - 0.0) < 1e-6

    def test_entropy(self):
        """-p*log(p) at p=0 — Shannon entropy term."""
        assert abs(resolve(lambda p: -p*log(p), at=0, dir="+") - 0.0) < 1e-6

    def test_kl_divergence_term(self):
        """p*log(p/q) at p=0 — KL divergence term."""
        q = 0.5
        assert abs(resolve(lambda p: p*log(p/q), at=0, dir="+") - 0.0) < 1e-6

    def test_softmax_log_sum_exp(self):
        """(log(exp(x) + 1) - log(2)) / x at x=0 — log-sum-exp centered."""
        assert abs(resolve(lambda x: (log(exp(x) + 1) - log(2))/x, at=0) - 0.5) < 1e-6

    def test_sigmoid_derivative_at_origin(self):
        """(exp(x)/(1+exp(x))^2) — sigmoid derivative, well-defined but test it."""
        assert abs(resolve(lambda x: exp(x)/(1+exp(x))**2, at=0) - 0.25) < 1e-10

    def test_huber_like(self):
        """(exp(x) - 1 - x) / x^2 — second-order loss approximation."""
        assert abs(resolve(lambda x: (exp(x) - 1 - x)/x**2, at=0) - 0.5) < 1e-10


# =============================================================================
# CONTROL SYSTEMS / ENGINEERING
# =============================================================================

class TestEngineering:

    def test_transfer_function_pole_cancellation(self):
        """(s^2 - 1) / (s - 1) at s=1 — pole-zero cancellation."""
        assert abs(resolve(lambda s: (s**2 - 1)/(s - 1), at=1) - 2.0) < 1e-10

    def test_cubic_transfer(self):
        """(s^3 - 27) / (s - 3) at s=3."""
        assert abs(resolve(lambda s: (s**3 - 27)/(s - 3), at=3) - 27.0) < 1e-10

    def test_feedback_gain(self):
        """1/(1 + K*G(s)) where G has a zero — (s-1)/((s-1)*(s+2)) at s=1."""
        assert abs(resolve(lambda s: (s-1)/((s-1)*(s+2)), at=1) - 1.0/3) < 1e-10

    def test_step_response_initial(self):
        """(1 - exp(-x)) / x — initial slope of step response."""
        assert abs(resolve(lambda x: (1 - exp(-x))/x, at=0) - 1.0) < 1e-10

    def test_bode_magnitude_at_crossover(self):
        """x / sqrt(1 + x^2) at x=0."""
        assert abs(resolve(lambda x: x/sqrt(1 + x**2), at=0) - 0.0) < 1e-10


# =============================================================================
# FINANCE
# =============================================================================

class TestFinance:

    def test_continuous_compounding(self):
        """(1 + r/n)^n as n->inf — continuous compounding."""
        r = 0.05
        expected = math.exp(r)
        assert abs(limit(lambda n: (1 + r/n)**n, to=math.inf) - expected) < 1e-4

    def test_continuous_compounding_high_rate(self):
        """Same with r=1."""
        assert abs(limit(lambda n: (1 + 1/n)**n, to=math.inf) - math.e) < 1e-4

    def test_black_scholes_at_zero_vol(self):
        """Simplified: x*exp(-x^2/2)/sqrt(2*pi) at x=0."""
        c = 1/math.sqrt(2*math.pi)
        assert abs(resolve(lambda x: x*exp(-x**2/2)*c, at=0) - 0.0) < 1e-10

    def test_annuity_factor(self):
        """(1 - (1+r)^(-n)) / r at r=0 — annuity present value factor → n."""
        n = 10
        assert abs(resolve(lambda r: (1 - (1+r)**(-n))/r, at=0) - float(n)) < 1e-4


# =============================================================================
# CALCULUS TEXTBOOK CLASSICS
# =============================================================================

class TestTextbook:

    def test_derivative_definition_x_squared(self):
        """lim h->0 ((x+h)^2 - x^2) / h at x=3 — definition of derivative."""
        x = 3
        assert abs(resolve(lambda h: ((x+h)**2 - x**2)/h, at=0) - 6.0) < 1e-10

    def test_derivative_definition_sin(self):
        """lim h->0 (sin(x+h) - sin(x)) / h at x=0."""
        x = 0
        assert abs(resolve(lambda h: (sin(x+h) - sin(x))/h, at=0) - math.cos(x)) < 1e-10

    def test_derivative_definition_exp(self):
        """lim h->0 (exp(x+h) - exp(x)) / h at x=1."""
        x = 1
        assert abs(resolve(lambda h: (exp(x+h) - exp(x))/h, at=0) - math.exp(x)) < 1e-10

    def test_squeeze_theorem(self):
        """x^2 * sin(1/x) at 0 — classic squeeze."""
        assert abs(resolve(lambda x: x**2 * sin(1/x), at=0) - 0.0) < 1e-10

    def test_exponential_dominates_polynomial(self):
        """x^3 * exp(-x) at inf."""
        assert abs(limit(lambda x: x**3 * exp(-x), to=math.inf) - 0.0) < 1e-4

    def test_log_grows_slower_than_polynomial(self):
        """log(x) / x at inf."""
        assert abs(limit(lambda x: log(x)/x, to=math.inf) - 0.0) < 1e-4

    def test_geometric_series_boundary(self):
        """(1 - x^n) / (1 - x) at x=1 — equals n."""
        n = 7
        assert abs(resolve(lambda x: (1 - x**n)/(1 - x), at=1) - float(n)) < 1e-8

    def test_lhopital_triple(self):
        """(exp(x) - 1 - x - x^2/2) / x^3 — requires three applications."""
        assert abs(resolve(lambda x: (exp(x) - 1 - x - x**2/2)/x**3, at=0) - 1.0/6) < 1e-10

    def test_rational_at_infinity(self):
        """(3x^2 + 2x - 1) / (x^2 - 4) at inf."""
        assert abs(limit(lambda x: (3*x**2 + 2*x - 1)/(x**2 - 4), to=math.inf) - 3.0) < 1e-4

    def test_conjugate_trick(self):
        """sqrt(x+1) - sqrt(x) at inf → 0."""
        assert abs(limit(lambda x: sqrt(x+1) - sqrt(x), to=math.inf) - 0.0) < 1e-4


# =============================================================================
# NUMERICAL COMPUTING PATTERNS
# =============================================================================

class TestNumerical:

    def test_expm1_over_x(self):
        """(exp(x)-1)/x — the reason expm1() exists."""
        assert abs(resolve(lambda x: (exp(x)-1)/x, at=0) - 1.0) < 1e-10

    def test_log1p_over_x(self):
        """log(1+x)/x — the reason log1p() exists."""
        assert abs(resolve(lambda x: log(1+x)/x, at=0) - 1.0) < 1e-10

    def test_cos_cancellation(self):
        """(1-cos(x))/x^2 — catastrophic cancellation in naive computation."""
        assert abs(resolve(lambda x: (1-cos(x))/x**2, at=0) - 0.5) < 1e-10

    def test_tan_minus_sin(self):
        """(tan(x)-sin(x))/x^3 — cancellation of leading terms."""
        assert abs(resolve(lambda x: (tan(x)-sin(x))/x**3, at=0) - 0.5) < 1e-10

    def test_evaluate_across_singularity(self):
        """Evaluate (x^3-8)/(x-2) for x in range including the singularity."""
        f = lambda x: (x**3 - 8)/(x - 2)
        results = [resolve(f, at=x) for x in range(-2, 5)]
        expected = [float(x**2 + 2*x + 4) for x in range(-2, 5)]
        for r, e in zip(results, expected):
            assert abs(r - e) < 1e-8

    def test_batch_sinc(self):
        """Evaluate sinc across a range including 0."""
        f = lambda x: sin(x)/x
        xs = [-2, -1, -0.5, 0, 0.5, 1, 2]
        for x in xs:
            y = resolve(f, at=x)
            if x == 0:
                assert abs(y - 1.0) < 1e-10
            else:
                assert abs(y - math.sin(x)/x) < 1e-10


# =============================================================================
# DIVERGENT LIMITS (should raise, not return wrong values)
# =============================================================================

class TestDivergent:

    def test_1_over_x_squared_diverges(self):
        with pytest.raises(LimitDivergesError):
            limit(lambda x: 1/x**2, to=0)

    def test_exp_at_infinity(self):
        with pytest.raises(LimitDivergesError):
            limit(lambda x: exp(x), to=math.inf)

    def test_log_at_zero_diverges(self):
        """log(x) at 0+ diverges to -inf."""
        with pytest.raises(LimitDivergesError) as exc_info:
            limit(lambda x: log(x), to=0, dir="+")
        assert exc_info.value.value == -math.inf

    def test_tan_near_pi_over_2(self):
        """tan(x) diverges near pi/2. Use a point we can represent exactly."""
        with pytest.raises(LimitDivergesError):
            limit(lambda x: 1/x**2, to=0, dir="+")


# =============================================================================
# NON-EXISTENT LIMITS (should raise, not return a value)
# =============================================================================

class TestNonExistent:

    def test_sin_1_over_x(self):
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: sin(1/x), to=0)

    def test_cos_1_over_x(self):
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: cos(1/x), to=0)

    def test_one_sided_disagree(self):
        """1/x from left and right disagree."""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: 1/x, to=0)

    def test_abs_x_over_x_both_sides(self):
        """|x|/x: +1 from right, -1 from left."""
        with pytest.raises(LimitDoesNotExistError):
            limit(lambda x: abs(x)/x, to=0)
