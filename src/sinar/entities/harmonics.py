"""Poloidal harmonics for magnetic-field with GR (Schwarzschild) corrections.

Flat-space vacuum fields are generated from the scalar potentials of the real spherical harmonics, 
B = -grad(Phi) with Phi_l ~ Y_l / r^(l+1), using automatic differentiation. 
This keeps the construction coordinate-free and 
regular everywhere off the origin: no polar-axis special cases.

GR correction factors (for Schwarzschild spacetime) Page & Sarmiento (1996, Appendix; Muslimov & Tsygan 1986):
for a degree-l harmonic the radial field component is scaled by f_l 
and the transverse components by sqrt(g00) g_l, with x = R_s / r = 2 / r 
in geometric units G = c = M = 1 (lengths in GM/c^2).

Amplitude convention: dipole amplitude B_P * R^3 gives polar field B_P, 
and a quadrupole generator amplitude q * B_P * R^4 gives maximum surface field q * B_P.

The magnetic frame is set by a row-vector orientation matrix called `orient`:
the dipole axis is `[0, 0, 1] @ orient` and the quadrupole azimuth is
measured from `[1, 0, 0] @ orient`, `[0, 1, 0] @ orient`.

The l = 2 correction factors suffer catastrophic cancellation for x -> 0; 
run with JAX_ENABLE_X64, and treat radii beyond r ~ 10^3 GM/c^2 with care.
"""
import jax
import jax.numpy as jnp
from jax import Array


def dipole_potential(position: Array) -> Array:
    """Flat-space scalar potential of a unit-amplitude dipole.

    Phi = z / (2 r^3) in magnetic-frame coordinates, normalized so that
    B = -grad(Phi) has polar strength 1 / r^3 (the Page & Sarmiento
    "maximum surface field" scale: twice the commonly quoted moment).
    """
    r = jnp.linalg.norm(position)
    return 0.5 * position[2] / r**3


def quadrupole_potential(generators: Array, position: Array) -> Array:
    """Flat-space scalar potential of the general quadrupole.

    ``generators`` holds the five amplitudes (Q_0, ..., Q_4) of the
    generating fields b_0, ..., b_4 of Page & Sarmiento (1996, Table 1):
    b_0 is the axisymmetric component, (b_1, b_2) the m = 1 pair with
    poles at colatitude pi/4, and (b_3, b_4) the m = 2 pair with four
    equatorial poles. -grad of this potential reproduces the tabulated
    spherical components; each b_i has unit maximum surface strength.
    """
    x, y, z = position[0], position[1], position[2]
    r2 = x * x + y * y + z * z
    poly = (
        generators[0] * (3.0 * z * z - r2) / 6.0
        + (2.0 / 3.0) * z * (generators[1] * y - generators[2] * x)
        - (2.0 / 3.0) * generators[3] * x * y
        + (generators[4] / 3.0) * (x * x - y * y)
    )
    return poly / jnp.sqrt(r2) ** 5


def schwarzschild_factors(l: int, r: Array) -> tuple[Array, Array]:
    """GR corrections (f_l, sqrt(g00) g_l) for a degree-l poloidal field.

    The radial field component is multiplied by the first factor and
    the transverse (theta and phi) components by the second. Both tend
    to unity at large r. Page & Sarmiento (1996), eqs (A4)-(A5).
    """
    x = 2.0 / r
    log_term = jnp.log1p(-x)
    alpha = jnp.sqrt(1.0 - x)
    if l == 1:
        f = -3.0 / x**3 * (log_term + 0.5 * x * (x + 2.0))
        g = -2.0 * f + 3.0 / (1.0 - x)
    elif l == 2:
        f = 10.0 / 3.0 / x**4 * (
            6.0 * log_term * (3.0 * x - 4.0) / x + x * x + 6.0 * x - 24.0
        )
        g = 10.0 / x**4 * (
            6.0 * log_term * (2.0 - x) / x
            + (x * x - 12.0 * x + 12.0) / (1.0 - x)
        )
    else:
        raise NotImplementedError(f"no Schwarzschild factors for l = {l}")
    return f, alpha * g


def corrected_field(potential, l: int, position: Array) -> Array:
    """Schwarzschild-corrected field of one degree-l flat potential.

    B_flat = -grad(potential) is split into radial and transverse parts
    which are scaled by the degree-l correction factors.
    """
    B_flat = -jax.grad(potential)(position)
    r = jnp.linalg.norm(position)
    r_hat = position / r
    f, g = schwarzschild_factors(l, r)
    B_r = jnp.vecdot(B_flat, r_hat)
    return f * B_r * r_hat + g * (B_flat - B_r * r_hat)


def magnetic_field(components: Array, orient: Array, position: Array) -> Array:
    """Schwarzschild-corrected poloidal magnetic field.

    Parameters
    ----------
    components : Array
        ``[c_dip]`` for a pure dipole, or ``[c_dip, Q_0, ..., Q_4]`` for
        a dipole plus the five quadrupole generators. Amplitudes follow
        the maximum-surface-field convention (module docstring). The
        shape branch is resolved at JAX trace time, so the one-component
        form compiles to the pure-dipole expression at no extra cost.
    orient : Array (3, 3)
        Row-vector orientation of the magnetic frame.
    position : Array (3,)
        Field point in lab coordinates, units of GM/c^2.

    Returns
    -------
    Array (3,)
        Magnetic field vector in lab coordinates.
    """
    p = orient @ position  # magnetic-frame coordinates
    B = components[0] * corrected_field(dipole_potential, 1, p)
    n = components.shape[0]
    if n == 6:
        quad = lambda q: quadrupole_potential(components[1:], q)
        B = B + corrected_field(quad, 2, p)
    elif n != 1:
        raise ValueError(f"components must have length 1 or 6; got {n}")
    return B @ orient