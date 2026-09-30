"""Refractive indices of the magnetised QED vacuum at any field strength.

The one-loop (Heisenberg-Euler) effective Lagrangian in the form of
Heyl & Hernquist (1997, J. Phys. A 30, 6485) gives the vacuum dielectric and
inverse permeability tensors

    eps = a I + q b b,        mu^-1 = a I - h b b,

with b the field direction.  a, h, q depend on the field strength only,
through xi = B / B_Q (this module takes xi^2).  Their weak-field limits are
a = 1 - 2 delta, h = 4 delta, q = 7 delta, with delta = (alpha_F/45 pi) xi^2.
In strong fields h saturates at alpha_F / 3 pi and q grows as
(alpha_F / 3 pi) xi.

For a photon at angle theta to B the vacuum is not gyrotropic, so the normal
modes stay linear with

    n_O^2 = (a + q) / (a + q cos^2 theta),    n_X^2 = a / (a - h sin^2 theta).

Both are independent of frequency for hbar omega << m_e c^2, so the
birefringence |Omega| = k Delta_n stays proportional to the photon energy.

This is a port of ``magopac.vacuum_corrections`` that JAX can trace under
jit, vmap and jvp: the lnGamma integral in X0 is a fixed Gauss-Legendre rule
rather than adaptive quadrature, and X0', X0'' are closed forms.
"""
import math

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import digamma, gammaln

from ..entities.poloidal_fields import ALPHA_F

# Glaisher-Kinkelin constant, ln A = 1/12 - zeta'(-1).  
# Python floats, so they are not frozen to float32 when imported before jax_enable_x64 is set.
LN_A = 0.2487544770337842625
LN2 = math.log(2.0)
LN4PI = math.log(4.0 * math.pi)

#: Coupling alpha_F / 2 pi carried by each of a - 1, h, q.
VACUUM_COUPLING = 0.5 * ALPHA_F / math.pi

# Exact Bernoulli numbers B[2k] for k = 0..15.
BERNOULLI_EVEN = np.array([
    1.0, 1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730, 7 / 6,
    -3617 / 510, 43867 / 798, -174611 / 330, 854513 / 138,
    -236364091 / 2730, 8553103 / 6, -23749461029 / 870,
    8615841276005 / 14322,
])
# Terms j = 1..N_WEAK_TERMS of the weak-field series (they need B[2j+2]).
N_WEAK_TERMS = 12
# The series is asymptotic, and the closed form loses digits to cancellation
# of terms ~ x^2 ln x as x = 1/xi grows.  At xi = 0.1 both hold to ~1e-10.
XI2_WEAK_FIELD_SWITCH = 1.0e-2

# lnGamma(v + 2) is analytic on the whole integration range (v + 2 in (1, 6]), 
# so 24 Gauss-Legendre nodes reach double precision.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(24)

################################
# Auxiliary function X0(x = 1/xi)
################################
def _x0_derivatives(x: Array) -> tuple:
    """X0, X0' and X0'' from the closed form of Heyl & Hernquist, x > 0."""
    # The lnGamma(v + 1) in 4 int_0^{x/2-1} lnGamma(v + 1) dv has a log
    # singularity at v = -1 (x -> 0).  Splitting off ln(v + 1) as
    # lnGamma(v + 2) - ln(v + 1) leaves a smooth integrand and an exact
    # remainder int_0^u ln(1 + v) dv = (1 + u) ln(1 + u) - u.
    u = 0.5 * x - 1.0
    v = 0.5 * u[..., None] * (_GL_NODES + 1.0)
    smooth = 0.5 * u * jnp.sum(_GL_WEIGHTS * gammaln(v + 2.0), axis=-1)
    lnx = jnp.log(x)
    log_part = 0.5 * x * (lnx - LN2) - u
    x0 = (4.0 * (smooth - log_part) - lnx / 3.0 + 2.0 * LN4PI - 4.0 * LN_A
          - 5.0 / 3.0 * LN2 - x * (LN4PI + 1.0 - lnx)
          + x * x * (0.75 - 0.5 * lnx + 0.5 * LN2))
    # The derivative of the integral follows from the fundamental theorem
    # of calculus.
    dx0 = (2.0 * gammaln(0.5 * x) - 1.0 / (3.0 * x) - LN4PI + lnx
           + 2.0 * x * (0.75 - 0.5 * lnx + 0.5 * LN2) - 0.5 * x)
    ddx0 = digamma(0.5 * x) + 1.0 / (3.0 * x * x) + 1.0 / x - lnx + LN2
    return x0, dx0, ddx0

def _strong_field_coefficients(xi2: Array) -> tuple:
    """(a, h, q) from the closed form of X0."""
    x = 1.0 / jnp.sqrt(xi2)
    x0, dx0, ddx0 = _x0_derivatives(x)
    x1 = -2.0 * x0 + x * dx0 + 2.0 / 3.0 * ddx0 - 2.0 / (9.0 * x * x)
    return (1.0 + VACUUM_COUPLING * (-2.0 * x0 + x * dx0),
            VACUUM_COUPLING * (x * x * ddx0 - x * dx0),
            -VACUUM_COUPLING * x1)

# X0(x) = -sum_j 4^j B[2j+2] / (j (j+1) (2j+1)) xi^2j, and substituting it
# into a, h and q = -c X1 with X1 = -2 X0 + x X0' + 2/3 X0'' - 2/(9 x^2)
# gives each as a power series in xi^2.
_J = np.arange(1, N_WEAK_TERMS + 1)
_A_SERIES = 2.0 * 4.0**_J * BERNOULLI_EVEN[_J + 1] / (_J * (2 * _J + 1))
_H_SERIES = -4.0 * 4.0**_J * BERNOULLI_EVEN[_J + 1] / (2 * _J + 1)
_Q_SERIES = -(4.0**_J * (6.0 * BERNOULLI_EVEN[_J + 1]
                         - (2 * _J + 1) * BERNOULLI_EVEN[_J])
              / (3.0 * _J * (2 * _J + 1)))

def _weak_field_coefficients(xi2: Array) -> tuple:
    """(a, h, q) from their power series in xi^2."""
    powers = xi2[..., None] ** _J
    return (1.0 + VACUUM_COUPLING * jnp.sum(_A_SERIES * powers, axis=-1),
            VACUUM_COUPLING * jnp.sum(_H_SERIES * powers, axis=-1),
            VACUUM_COUPLING * jnp.sum(_Q_SERIES * powers, axis=-1))

##############
# Coefficients
##############
def vacuum_coefficients(xi2: Array) -> tuple:
    """Vacuum corrections (a, h, q) at field strength xi^2 = (B / B_Q)^2.

    Parameters
    ----------
    xi2 : Array
        (B / B_Q)^2, zero included.

    Returns
    -------
    a, h, q : Array
        eps = a I + q b b and mu^-1 = a I - h b b.
    """
    xi2 = jnp.asarray(xi2, float)
    use_weak = xi2 < XI2_WEAK_FIELD_SWITCH
    # Clamp each branch to its own domain, so the unused one (and its
    # derivative under jvp) stays finite.  jnp.where rather than
    # jnp.minimum, which would split the derivative at the switch.
    weak = _weak_field_coefficients(
        jnp.where(use_weak, xi2, XI2_WEAK_FIELD_SWITCH))
    strong = _strong_field_coefficients(
        jnp.where(use_weak, XI2_WEAK_FIELD_SWITCH, xi2))
    return tuple(jnp.where(use_weak, w, s) for w, s in zip(weak, strong))

def mode_index_offsets(xi2: Array, sin2: Array) -> tuple:
    """(n_O - 1, n_X - 1) at xi^2 = (B / B_Q)^2 and sin^2 of the angle to B.

    They are formed from n^2 - 1 without subtracting 1, so they keep full
    relative precision down to the weakest fields.
    """
    a, h, q = vacuum_coefficients(xi2)
    n2o = q * sin2 / (a + q * (1.0 - sin2))
    n2x = h * sin2 / (a - h * sin2)
    return (n2o / (1.0 + jnp.sqrt(1.0 + n2o)),
            n2x / (1.0 + jnp.sqrt(1.0 + n2x)))

###############
# Birefringence
###############
def vacuum_birefringence(xi2: Array, sin2: Array) -> Array:
    """Delta_n = n_O - n_X of the one-loop QED vacuum at any field strength.

    Parameters
    ----------
    xi2 : Array
        (B / B_Q)^2.
    sin2 : Array
        sin^2 of the angle between the photon and B, (B_perp / B)^2.
    """
    n_o, n_x = mode_index_offsets(xi2, sin2)
    return n_o - n_x

def weak_field_birefringence(xi2: Array, sin2: Array) -> Array:
    """Delta_n = (alpha_F / 30 pi) xi^2 sin^2, valid for B << B_Q."""
    return ALPHA_F / (30.0 * math.pi) * xi2 * sin2
