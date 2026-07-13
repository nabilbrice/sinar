"""Adiabatic polarization transport.

Vacuum birefringence in a strong magnetic field couples the photon polarization
to the local X/O mode basis.
"""

import jax
from jax import Array
import jax.numpy as jnp

from functools import partial

from ..rays import normalize, _approx_rayphase_at_r
from .harmonics import magnetic_field

# Gamma units prefactor 1 / 1.00657e+19 for E in keV, B in Gauss, lengths in GM/c^2,
# stellar mass in solar masses.
_GAMMA_CONST = 9.9347e-20


def adiabatic_parameter(
    energy_keV,
    components: Array,
    orient: Array,
    position: Array,
    direction: Array,
    M_solar: float = 1.4,
) -> float:
    """Computes the adiabatic parameter: the ratio Gamma = l_B / l_A 
    (Heyl & Shaviv 2000).

    Gamma > 1/2: polarization adiabatically follows the mode basis.
    Gamma < 1/2: the modes are decoupled and the polarization is frozen.
    """
    k = normalize(direction)
    energy_local = energy_keV / jnp.sqrt(
        1.0 - 2.0 / jnp.linalg.norm(position)
    )

    def b_perp_sq(pos):
        B = magnetic_field(components, orient, pos)
        return jnp.vecdot(B, B) - jnp.vecdot(B, k)**2

    val, slope = jax.jvp(b_perp_sq, (position,), (k,))

    return _GAMMA_CONST * energy_local * M_solar * val * (
        val / (jnp.abs(slope) + 1e-30)
    )


@partial(jax.jit, static_argnames=("n_scan", "n_iter"))
def adiabatic_radius(
    energy_keV,
    components: Array,
    orient: Array,
    b2,
    los_perp: Array,
    los: Array,
    r_min,
    r_max,
    n_scan: int = 30,
    n_iter: int = 20,
    M_solar: float = 1.4,
):
    """Outermost Gamma = 1/2 crossing along one trajectory.

    The trajectory is scanned outside-in with n_scan parallel Gamma evaluations
    to bracket the outermost crossing, which is then refined with n_iter bisection steps.

    Returns NaN when no crossing lies in [r_min, r_max]. Crossings closer than
    (r_max - r_min) / n_scan may be misssed.
    """

    def gamma_at_r(r):
        phase = _approx_rayphase_at_r(r, b2, los, los_perp)
        return adiabatic_parameter(
            energy_keV, components, orient, phase[:3], phase[3:6], M_solar
        )

    r_samples = jnp.linspace(r_max, r_min, n_scan)
    adiabatic = jax.vmap(gamma_at_r)(r_samples) > 0.5
    crossings = adiabatic[:-1] != adiabatic[1:]

    # argmax returns the first True, i.e. the outermost bracket
    idx = jnp.argmax(crossings)
    found = jnp.any(crossings)
    lo = jnp.where(found, r_samples[idx + 1], jnp.nan)
    hi = jnp.where(found, r_samples[idx], jnp.nan)

    def bisect(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        inside = gamma_at_r(mid) > 0.5
        # if still adiabatic at mid (inside True): the crossing is farther out
        return jnp.where(inside, mid, lo), jnp.where(inside, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, bisect, (lo, hi))
    return 0.5 * (lo + hi)


def frozen_polarization(
    components: Array,
    orient: Array,
    r_freeze,
    b2,
    los_perp: Array,
    los: Array,
    fallback_position: Array,
):
    """X-mode direction at the observer for one trajectory.

    The mode basis is evaluated at the freeze-out point on the bent ray: e_X ~ k x B,
    the direction is then parallely-transported to infinity:
    the angle between the polarization and the ray's geodesic plane is conserved,
    so the components on the plane's (tangent, normal) pair are re-assembled
    on the observer sky.
    """
    phase = _approx_rayphase_at_r(r_freeze, b2, los, los_perp)
    valid = ~jnp.isnan(phase[0])
    pos = jnp.where(valid, phase[:3], fallback_position)
    k_march = jnp.where(valid, phase[3:6], -los)
    k_phot = normalize(-k_march) # photons propagate towards the observer

    B = magnetic_field(components, orient, pos)
    e_X = normalize(jnp.cross(k_phot, B))

    n_hat = normalize(jnp.cross(los, los_perp)) # geodesic-plane normal
    t_hat = normalize(jnp.cross(n_hat, k_phot)) # in-plane transverse
    cos_beta = jnp.dot(e_X, t_hat)
    sin_beta = jnp.dot(e_X, n_hat)

    t_inf = jnp.cross(n_hat, los)
    return cos_beta * t_inf + sin_beta * n_hat


def stokes_qu(e_pol: Array, u_hat: Array, v_hat: Array):
    """Normalized (q, u) of fully polarized light along e_pol.

    The polarimeter frame is (u_hat, v_hat) on the observer's sky.
    Degenerate polarization vectors (e.g. from k parallel to B) yield zeros
    rather than NaNs so that the weighted sums stay finite.
    """
    theta = jnp.arctan2(jnp.dot(e_pol, v_hat), jnp.dot(e_pol, u_hat))
    q = jnp.cos(2.0 * theta)
    u = jnp.sin(2.0 * theta)
    q = jnp.where(jnp.isfinite(q), q, 0.0)
    u = jnp.where(jnp.isfinite(u), u, 0.0)
    return q, u


def observed_stokes(
    energy_keV,
    components: Array,
    orient: Array,
    b2s: Array,
    los_perps: Array,
    los: Array,
    u_hat: Array,
    v_hat: Array,
    r_surface,
    *,
    surface_intensity,
    per_ray_radius: bool = True,
    r_freeze = 0.0,
    r_max: float = 15000.0,
    n_scan: int = 30,
    n_iter: int = 20,
    M_solar: float = 1.4,
):
    """Sky-integrated Stokes (I, Q, U) at one energy and orientation.

    Each trajectory in the grid (b2s, los_perps) is followed to the stellar surface
    at r_surface. Rays that miss carry zero weight. Rays that hit are weighted according
    to the surface_intensity(components, orient, position, energy_keV).

    With per_ray_radius True, each trajectory freezes at its own adiabatic_radius,
    (where NaN results in r_max); otherwise all rays freeze at a common r_freeze,
    e.g. the central-ray adiabatic radius.
    """

    def one_ray(b2, los_perp):
        phase_s = _approx_rayphase_at_r(r_surface, b2, los, los_perp)
        is_hit = ~jnp.isnan(phase_s[0])
        pos_s = jnp.where(is_hit, phase_s[:3], r_surface * los)
        weight = jnp.where(
            is_hit,
            surface_intensity(components, orient, pos_s, energy_keV),
            0.0,
        )

        if per_ray_radius:
            r_a = adiabatic_radius(
                energy_keV,
                components,
                orient,
                b2,
                los_perp,
                los,
                r_surface,
                r_max,
                n_scan=n_scan,
                n_iter=n_iter,
                M_solar=M_solar,
            )
            r_a = jnp.where(jnp.isnan(r_a), r_max, r_a)
        else:
            r_a = r_freeze
        r_a = jnp.maximum(r_a, r_surface)

        e_pol = frozen_polarization(
            components, orient, r_a, b2, los_perp, los, pos_s
        )
        q, u = stokes_qu(e_pol, u_hat, v_hat)
        return weight, weight * q, weight * u

    w, wq, wu = jax.vmap(one_ray)(b2s, los_perps)
    return jnp.sum(w), jnp.sum(wq), jnp.sum(wu)


def polarization_fraction_angle(I, Q, U):
    """Polarization degree and angle from Stokes parameters.
    """
    pf = jnp.sqrt(Q**2 + U**2) / I
    pa = jnp.rad2deg(0.5 * jnp.arctan2(U, Q)) % 180.0
    return pf, pa