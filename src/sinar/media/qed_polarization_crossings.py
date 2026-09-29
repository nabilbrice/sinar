"""Threshold-crossing approximations to QED vacuum birefringence.

Replaces the Stokes ODE of ``qed_polarization`` by locating where an
adiabaticity parameter Gamma crosses a threshold along a ray, following the
local mode while Gamma is above it and freezing the polarization while it is
below.  ``freeze_angle`` and ``multi_crossing_freeze`` work on profiles
sampled along a ``QEDVacuum.reference`` ray, ``put_qed_switch`` builds the
Mushtukov et al. (2026) Gamma_Omega and step switch on a ``QEDVacuum``, and
``observed_stokes_multi`` sums the Stokes parameters of an observer's image
with sudden transport through every crossing.
"""
from functools import partial
import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ..rays import normalize, approx_coordphase_at_r
from ..entities.poloidal_fields import magnetic_field
from .qed_polarization import QEDVacuum, _mode_turns, _u_grid

class QEDSwitch(NamedTuple):
    """The Mushtukov et al. (2026) criterion and switch on a QEDVacuum.

    Rays are (b2, r_hat, m_hat, n_hat) tuples, as for ``QEDVacuum``.

    Attributes
    ----------
    gamma_omega : Callable
        gamma_omega(ray, r, modulus=False) -> Gamma_Omega at radii r.
    gamma_grad : Callable
        gamma_grad(ray, r=None, n_c=400, modulus=False) -> (r, Gamma_Omega),
        on ``r`` or on n_c log-spaced radii from R to r_max.
    switch : Callable
        switch(ray, n_steps, modulus=False, threshold=0.5) -> position
        angle [deg] of the step-by-step switch.
    """
    gamma_omega: Callable
    gamma_grad: Callable
    switch: Callable

########################################
# Gamma = |Omega| l_B along a trajectory
########################################
# Should equal K_OMEGA; it is 0.2% above it.  The original value K_OMEGA/3
# under-estimated Gamma and froze rays ~25% too early.
_GAMMA_CONST = 9.9347e-20 * 3.0

def _adiabatic_parameter(energy_keV, components, orient, position, direction,
                         M_solar=1.4):
    """Gamma = |Omega| l_B at a position, with l_B the B_perp^2 scale length."""
    k = direction / jnp.linalg.norm(direction)
    energy_local = energy_keV / jnp.sqrt(
        1.0 - 2.0 / jnp.linalg.norm(position)
    )

    def b_perp_sq(pos):
        B = magnetic_field(components, orient, pos)
        return jnp.vecdot(B, B) - jnp.vecdot(B, k)**2

    # The derivative along k is a single jvp, exact at the point.
    val, slope = jax.jvp(b_perp_sq, (position,), (k,))

    return _GAMMA_CONST * energy_local * M_solar * val * (
        val / (jnp.abs(slope) + 1e-30)
    )

def _adiabatic_radius(energy_keV, components, orient, phase_at_r, r_min, r_max,
                      n_scan=30, n_iter=20, M_solar=1.4, log_scan=True,
                      min_width=0.0):
    """Outermost radius where Gamma rises through 1/2 and stays above it.

    ``phase_at_r`` maps a radius to the trajectory's phase there.  A crossing
    counts only if Gamma stays above 1/2 over a fractional width
    ``min_width`` inward of it.  NaN if there is none.
    """
    def gamma_at_r(r):
        phase = phase_at_r(r)
        return _adiabatic_parameter(
            energy_keV, components, orient, phase[:3], phase[3:6], M_solar
        )

    spacing = jnp.geomspace if log_scan else jnp.linspace
    r_samples = spacing(r_max, r_min, n_scan)
    adiabatic = jax.vmap(gamma_at_r)(r_samples) > 0.5

    # The window length m sets an array shape, so it is computed in Python
    # from the static scan settings.
    lo_r, hi_r = float(r_min), float(r_max)
    if log_scan:
        step = math.log(hi_r / lo_r) / (n_scan - 1)
    else:
        step = ((hi_r - lo_r) / (n_scan - 1)) / hi_r
    m = min(n_scan, max(1, math.ceil(math.log1p(min_width) / step)))

    # A running count over m samples finds where Gamma stays above 1/2 for
    # the whole window, so narrow adiabatic pockets are skipped.
    padded = jnp.concatenate([adiabatic, jnp.full((m,), adiabatic[-1])])
    counts = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(padded.astype(jnp.int32))]
    )
    sustained = (counts[m:] - counts[:-m]) == m

    crossings = (~adiabatic[:-1]) & sustained[1:n_scan]

    idx = jnp.argmax(crossings)
    found = jnp.any(crossings)
    lo = jnp.where(found, r_samples[idx + 1], jnp.nan)
    hi = jnp.where(found, r_samples[idx], jnp.nan)

    def bisect(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        inside = gamma_at_r(mid) > 0.5
        return jnp.where(inside, mid, lo), jnp.where(inside, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_iter, bisect, (lo, hi))
    return 0.5 * (lo + hi)

def _adiabatic_crossings(energy_keV, components, orient, phase_at_r,
                         r_min, r_max, n_scan=400, n_iter=20, max_roots=8,
                         M_solar=1.4):
    """All Gamma = 1/2 crossings along a trajectory, inner to outer.

    Returns (r_cross [max_roots], NaN-padded; whether Gamma(r_min) > 1/2).
    With the number of roots, the second fixes every segment's regime by
    parity.
    """
    def gamma_at_r(r):
        phase = phase_at_r(r)
        return _adiabatic_parameter(
            energy_keV, components, orient, phase[:3], phase[3:6], M_solar
        )

    # Sign changes miss an even number of crossings within one scan cell.
    # Gamma dips through 1/2 in narrow intervals near quasi-tangential points
    # (k nearly along B), so the scan must be fine enough to resolve them.
    r_samples = jnp.linspace(r_min, r_max, n_scan)
    adiabatic = jax.vmap(gamma_at_r)(r_samples) > 0.5
    flips = adiabatic[:-1] != adiabatic[1:]

    # A fixed number of roots keeps the shapes static for jit and vmap;
    # roots beyond max_roots (innermost kept) are dropped.
    idx = jnp.flatnonzero(flips, size=max_roots, fill_value=-1)
    valid = idx >= 0
    idx_safe = jnp.maximum(idx, 0)

    def refine(i):
        lo, hi, state_lo = r_samples[i], r_samples[i + 1], adiabatic[i]

        def bisect(_, bracket):
            lo, hi = bracket
            mid = 0.5 * (lo + hi)
            same = (gamma_at_r(mid) > 0.5) == state_lo
            return jnp.where(same, mid, lo), jnp.where(same, hi, mid)

        lo, hi = jax.lax.fori_loop(0, n_iter, bisect, (lo, hi))
        return 0.5 * (lo + hi)

    r_cross = jax.vmap(refine)(idx_safe)
    return jnp.where(valid, r_cross, jnp.nan), adiabatic[0]

def taverna_radius(B_pole_G: float, energy_keV: float, R: float) -> float:
    """Taverna et al. (2015) analytic adiabatic radius, in code units.

    Parameters
    ----------
    B_pole_G : float
        Polar field strength in Gauss.
    energy_keV : float
    R : float
        Stellar radius, GM/c^2.
    """
    return 4.8 * (B_pole_G / 1.0e11) ** 0.4 * energy_keV ** 0.2 * R

################################
# Crossings of a sampled profile
################################
@jax.jit
def threshold_crossings(r: Array, gamma: Array, threshold: float):
    """All crossings of ``gamma = threshold``, ordered inner to outer.

    Parameters
    ----------
    r, gamma : Array [n]
        A Gamma profile sampled at increasing radii.
    threshold : float

    Returns
    -------
    radii : Array [n - 1]
        Crossing radii, NaN-padded past the last crossing.
    entering : Array [n - 1]
        True where Gamma rises through the threshold, i.e. where the ray
        re-enters the adiabatic zone; False in the padding.
    """
    ad = gamma > threshold
    flips = ad[:-1] != ad[1:]
    # Padding to n - 1, the most crossings n samples can hold, keeps the
    # shapes static without dropping any.
    idx = jnp.flatnonzero(flips, size=flips.shape[0], fill_value=0)
    valid = jnp.arange(flips.shape[0]) < jnp.sum(flips)
    # Gamma is close to a power law in r, so the crossing is interpolated
    # in log-log.
    lg, lr = jnp.log(jnp.maximum(gamma, 1e-300)), jnp.log(r)
    t = (jnp.log(threshold) - lg[idx]) / (lg[idx + 1] - lg[idx])
    radii = jnp.exp(lr[idx] + t * (lr[idx + 1] - lr[idx]))
    return jnp.where(valid, radii, jnp.nan), valid & ad[idx + 1]

@jax.jit
def outermost_crossing(r: Array, gamma: Array, threshold: float):
    """Outermost crossing of ``gamma = threshold``; NaN if there is none.

    See ``threshold_crossings`` for the arguments.
    """
    radii, _ = threshold_crossings(r, gamma, threshold)
    n = jnp.sum(jnp.isfinite(radii))
    return jnp.where(n > 0, radii[jnp.maximum(n - 1, 0)], jnp.nan)

##################################
# Freeze prescriptions along a ray
##################################
@jax.jit
def freeze_angle(r: Array, zeta: Array, r_a):
    """Position angle [deg] of the X mode frozen at one radius.

    Parameters
    ----------
    r, zeta : Array [n]
        The mode angle profile, as ``QEDVacuum.reference`` returns it.
    r_a : float | None
        The freeze radius, clipped to the profile.  None or NaN freezes at
        the outer edge.

    Returns
    -------
    pa : float
        On the screen basis, as ``qed_polarization.stokes_observables``.
    """
    r_a = r[-1] if r_a is None else jnp.where(jnp.isfinite(r_a), r_a, r[-1])
    pa_mode = jnp.rad2deg(0.5 * zeta) + 90.0
    return jnp.interp(jnp.log(jnp.clip(r_a, r[0], r[-1])),
                      jnp.log(r), pa_mode) % 180.0

@partial(jax.jit, static_argnames="project")
def multi_crossing_freeze(r: Array, zeta: Array, r_gamma: Array, gamma: Array,
                          threshold: float, project: bool = False):
    """Position angle and degree across repeated threshold crossings.

    The polarization follows the local mode while Gamma > threshold and is
    frozen while it is below; a ray may leave and re-enter the adiabatic
    zone any number of times.  A single crossing reproduces
    ``freeze_angle``.

    Parameters
    ----------
    r, zeta : Array [n]
        The mode angle profile, as ``QEDVacuum.reference`` returns it.
    r_gamma, gamma : Array [m]
        The Gamma profile, e.g. from ``QEDSwitch.gamma_grad`` or
        ``QEDVacuum.probe``.
    threshold : float
    project : bool
        False follows Mushtukov et al. (2026, App. B1): only the rotation is
        frozen, so an offset picked up in a frozen pocket is carried rigidly
        to the observer and p_L = 1.  True keeps only the projection on the
        local mode at each re-entry, the limit of
        ``qed_polarization.projected_observables``: it resets the offset and
        costs a factor cos(zeta_frozen - zeta_entry) in p_L.

    Returns
    -------
    pa : float
        Position angle [deg] on the screen basis.
    p_L : float
        Linear degree.
    """
    lr = jnp.log(r)
    rx, entering = threshold_crossings(r_gamma, gamma, threshold)
    valid = jnp.isfinite(rx)
    z_c = jnp.interp(jnp.log(jnp.clip(jnp.where(valid, rx, r[0]), r[0], r[-1])),
                     lr, zeta)

    # zp is the doubled polarization angle, equal to the mode angle zeta
    # while locked: surface emission starts in the mode.  z_in is the mode
    # angle where the current adiabatic segment began.
    def step(carry, x):
        amp, zp, z_in, adiabatic = carry
        z, ent, ok = x
        enter = ok & ent & ~adiabatic
        leave = ok & ~ent & adiabatic
        if project:
            amp = jnp.where(enter, amp * jnp.cos(zp - z), amp)
            zp = jnp.where(enter, z, zp)
        # While locked the polarization turns with the mode, so leaving adds
        # the mode rotation over the segment.
        zp = jnp.where(leave, zp + z - z_in, zp)
        z_in = jnp.where(enter, z, z_in)
        adiabatic = (adiabatic | enter) & ~leave
        return (amp, zp, z_in, adiabatic), None

    z0 = zeta[0]
    (amp, zp, z_in, adiabatic), _ = jax.lax.scan(
        step, (jnp.ones_like(z0), z0, z0, gamma[0] > threshold),
        (z_c, entering, valid))
    # Still locked at the outer edge.
    zp = jnp.where(adiabatic, zp + zeta[-1] - z_in, zp)
    # A negative product leaves the vector antiparallel to the local mode,
    # i.e. in the other mode: a 90 deg rotation of the sky angle.
    pa = jnp.rad2deg(0.5 * zp) + 90.0 + jnp.where(amp >= 0.0, 0.0, 90.0)
    return pa % 180.0, jnp.abs(amp)

#####################################
# Mushtukov et al. (2026) step switch
#####################################
def _gamma_omega(coeffs, ray, r, modulus=False):
    """Mushtukov et al. (2026) eq. (7), |Omega| l_Omega, at radii r."""
    # Eq. (7) is typeset with unit-vector hats, which makes it degenerate.
    # It is read with the vector Omega = |Omega| (cos zeta, sin zeta), so
    #     1/Gamma^2 = (d ln|Omega|/dl)^2 / |Omega|^2 + (dzeta/dl)^2 / |Omega|^2.
    # ``modulus`` differentiates |Omega| alone, dropping the second term, the
    # only one that couples the modes: it calls a B_perp^2 minimum adiabatic
    # exactly where zeta turns fastest, but reproduces the paper's figures
    # better.  coeffs is pointwise in r, so a jvp with unit tangent gives
    # dOmega/dr exactly at every sample.
    def omega(x):
        om, w = coeffs(ray, x)[:2]
        return om[:, None] * w
    Om, dOm = jax.jvp(omega, (r,), (jnp.ones_like(r),))
    dsdr = coeffs(ray, r)[2]
    om2 = jnp.sum(Om * Om, axis=1)
    if modulus:
        dom = jnp.abs(jnp.sum(Om * dOm, axis=1)) / jnp.sqrt(om2)
    else:
        dom = jnp.linalg.norm(dOm, axis=1)
    return om2 * dsdr / jnp.maximum(dom, 1e-300)

def _gamma_grad(gamma_omega, R, r_max, ray, r=None, n_c=400, modulus=False):
    # On the reference grid the switch sees the steps the Stokes vector is
    # integrated on: frozen pockets can be a few percent of r wide, and their
    # edges set where the rotation stops and resumes.
    r = jnp.geomspace(R, r_max, n_c) if r is None else jnp.asarray(r)
    return r, gamma_omega(ray, r, modulus)

def _switch(coeffs, R, r_max, ray, n_steps, modulus=False, threshold=0.5):
    # Emulates an integrator that decides per step, as in their App. B2.2:
    # each step's Gamma_Omega is the finite difference of Omega across it,
    # and it turns the polarization with the mode only if adiabatic.  A
    # pocket narrower than a step whose ends are both adiabatic is stepped
    # over.  With many steps this converges to multi_crossing_freeze.
    r = _u_grid(R, r_max, n_steps)
    om, w, dsdr, _, _ = coeffs(ray, r)
    dz = _mode_turns(w)
    ds = 0.5 * (dsdr[1:] + dsdr[:-1]) * jnp.diff(r)
    Om = om[:, None] * w
    dOm = (jnp.abs(jnp.diff(om)) if modulus
           else jnp.linalg.norm(jnp.diff(Om, axis=0), axis=1))
    mid = 0.5 * (om[1:] + om[:-1])
    gamma = mid * mid * ds / jnp.maximum(dOm, 1e-300)
    z = jnp.arctan2(w[0, 1], w[0, 0]) + jnp.sum(jnp.where(gamma > threshold, dz, 0.0))
    return (jnp.rad2deg(0.5 * z) + 90.0) % 180.0

def put_qed_switch(medium: QEDVacuum) -> QEDSwitch:
    """Puts the Mushtukov et al. (2026) switch on a QEDVacuum.

    The switch reads the medium's coefficients, so it sees exactly the Omega
    the Stokes ODE integrates.

    Parameters
    ----------
    medium : QEDVacuum

    Returns
    -------
    switch : QEDSwitch
    """
    gamma_omega = jax.jit(partial(_gamma_omega, medium.coeffs),
                          static_argnums=(2,), static_argnames=("modulus",))
    return QEDSwitch(
        gamma_omega=gamma_omega,
        gamma_grad=partial(_gamma_grad, gamma_omega, medium.R, medium.r_max),
        switch=jax.jit(partial(_switch, medium.coeffs, medium.R, medium.r_max),
                       static_argnums=(1, 2),
                       static_argnames=("n_steps", "modulus")),
    )

###########################################
# Sudden transport over an observer's image
###########################################
def _mode_frame_angle(components, orient, r, b2, los_perp, los):
    """(cos 2b, sin 2b) of the X-mode axis in the ray transport frame.

    b is the angle between e_X ~ k x B and the in-plane transverse direction
    t_hat; (t_hat, n_hat) is parallel-transported and maps to (t_inf, n_hat)
    at infinity.
    """
    phase = approx_coordphase_at_r(r, b2, los, los_perp)
    valid = ~jnp.isnan(phase[0])
    pos = jnp.where(valid, phase[:3], r * los)
    k_march = jnp.where(valid, phase[3:6], -los)
    k_phot = normalize(-k_march)

    B = magnetic_field(components, orient, pos)
    e_X = jnp.cross(k_phot, B)

    n_hat = normalize(jnp.cross(los, los_perp))
    t_hat = normalize(jnp.cross(n_hat, k_phot))
    c = jnp.dot(e_X, t_hat)
    s = jnp.dot(e_X, n_hat)
    # k parallel to B has no mode axis.  It falls back to (1, 0), which is
    # harmless: such points lie deep inside a frozen segment, where the
    # basis is never sampled.
    norm = c * c + s * s
    ok = norm > 0.0
    cos2b = jnp.where(ok, (c * c - s * s) / jnp.where(ok, norm, 1.0), 1.0)
    sin2b = jnp.where(ok, 2.0 * c * s / jnp.where(ok, norm, 1.0), 0.0)
    return cos2b, sin2b

def _multi_crossing_stokes(components, orient, r_cross, adiab_start, b2,
                           los_perp, los, u_hat, v_hat, r_surface, r_max,
                           q_mode0=1.0, project=False):
    """Reduced Stokes (q, u) at the observer for one trajectory."""
    # Sharp-edge (sudden) transport through alternating segments, following
    # the switch of Mushtukov et al. (2026, App. B1), which freezes only the
    # rotation:
    # * adiabatic: the mode amplitudes are conserved, so the mode-frame pair
    #   (q_m, u_m) is constant while the modes turn with B_perp;
    # * frozen: the polarization direction is parallel-transported, so the
    #   transport-frame pair (q_t, u_t) is constant;
    # * adiabatic -> frozen at r_i: (q_t, u_t) = R(2 b_i) (q_m, u_m);
    # * frozen -> adiabatic at r_j: (q_m, u_m) = R(-2 b_j) (q_t, u_t).
    # With ``project`` the re-entry drops u_m instead: the relative phase
    # after re-entry is >> 1, so U and V dephase.
    def angle_at(r):
        return _mode_frame_angle(components, orient, r, b2, los_perp, los)

    # Born frozen: imprint the surface basis immediately.
    c0, s0 = angle_at(r_surface)
    q_t0 = jnp.where(adiab_start, 0.0, q_mode0 * c0)
    u_t0 = jnp.where(adiab_start, 0.0, q_mode0 * s0)

    def step(carry, r_i):
        q_t, u_t, q_m, u_m, adiab = carry
        valid = ~jnp.isnan(r_i)
        c2, s2 = angle_at(jnp.where(valid, r_i, r_surface))

        q_f, u_f = q_m * c2 - u_m * s2, q_m * s2 + u_m * c2
        q_a = q_t * c2 + u_t * s2
        u_a = 0.0 if project else u_t * c2 - q_t * s2

        carry = (
            jnp.where(valid & adiab, q_f, q_t),
            jnp.where(valid & adiab, u_f, u_t),
            jnp.where(valid & ~adiab, q_a, q_m),
            jnp.where(valid & ~adiab, u_a, u_m),
            jnp.where(valid, ~adiab, adiab),
        )
        return carry, None

    (q_t, u_t, q_m, u_m, adiab), _ = jax.lax.scan(
        step, (q_t0, u_t0, q_mode0, jnp.zeros_like(q_t0), adiab_start),
        r_cross,
    )

    # Still adiabatic past the last root (the outermost freeze lies beyond
    # the scan, or max_roots ran out): imprint at r_max.
    c_g, s_g = angle_at(r_max)
    q_t = jnp.where(adiab, q_m * c_g - u_m * s_g, q_t)
    u_t = jnp.where(adiab, q_m * s_g + u_m * c_g, u_t)

    # Transport frame -> polarimeter frame at infinity.
    n_hat = normalize(jnp.cross(los, los_perp))
    t_inf = jnp.cross(n_hat, los)
    cp = jnp.dot(t_inf, u_hat)
    sp = jnp.dot(t_inf, v_hat)
    c2p, s2p = cp * cp - sp * sp, 2.0 * cp * sp
    q_obs = q_t * c2p - u_t * s2p
    u_obs = u_t * c2p + q_t * s2p
    q_obs = jnp.where(jnp.isfinite(q_obs), q_obs, 0.0)
    u_obs = jnp.where(jnp.isfinite(u_obs), u_obs, 0.0)
    return q_obs, u_obs

def observed_stokes_multi(energy_keV: float, components: Array, orient: Array,
                          b2s: Array, los_perps: Array, los: Array,
                          u_hat: Array, v_hat: Array, r_surface: float, *,
                          surface_intensity: Callable,
                          q_mode0: float = 1.0,
                          project: bool = False,
                          r_max: float = 15000.0,
                          n_scan: int = 400,
                          n_iter: int = 20,
                          max_roots: int = 8,
                          M_solar: float = 1.4):
    """Image-summed Stokes (I, Q, U) with sudden transport through crossings.

    Every Gamma = |Omega| l_B = 1/2 crossing along each observer ray is
    found, and the polarization is frozen or locked between them.  Each ray
    stays fully polarized unless ``project`` is set; V is identically zero,
    since it is generated only in the transition layers the sharp edge
    cannot resolve.  Run with JAX_ENABLE_X64.

    Parameters
    ----------
    energy_keV : float
    components, orient : Array
        The field, as ``poloidal_fields.magnetic_field`` takes it.
    b2s, los_perps : Array [N], [N, 3]
        Per-pixel ray invariants, from ``renderers.construct_obs_rays``.
    los : Array [3]
        Unit direction toward the observer.
    u_hat, v_hat : Array [3]
        Sky-plane polarimeter axes; Q is measured along u_hat.
    r_surface : float
        Stellar radius, GM/c^2.
    surface_intensity : Callable
        surface_intensity(components, orient, position, energy_keV) -> the
        intensity weight of a ray leaving the surface at ``position``.
    q_mode0 : float
        Mode-frame Stokes Q at emission: +1 for pure X, -1 for pure O,
        2 p0 - 1 for a partially polarized seed.
    project : bool
        Drop the mode-frame u at each re-entry, as in
        ``multi_crossing_freeze``, so p_L < 1 per ray.
    r_max : float
        Outer edge of the crossing search.
    n_scan, n_iter : int
        Samples of the crossing scan and bisection steps per root.  The scan
        must resolve the narrowest Gamma dip of interest.
    max_roots : int
        Crossings kept per ray, innermost first.
    M_solar : float

    Returns
    -------
    I, Q, U : float
        Intensity-weighted sums over the rays that hit the surface.
    mean_crossings : float
        Mean number of crossings over those rays.
    """
    def one_ray(b2, los_perp):
        phase_s = approx_coordphase_at_r(r_surface, b2, los, los_perp)
        is_hit = ~jnp.isnan(phase_s[0])
        pos_s = jnp.where(is_hit, phase_s[:3], r_surface * los)
        weight = jnp.where(
            is_hit,
            surface_intensity(components, orient, pos_s, energy_keV),
            0.0,
        )

        r_cross, adiab0 = _adiabatic_crossings(
            energy_keV, components, orient,
            lambda r: approx_coordphase_at_r(r, b2, los, los_perp),
            r_surface, r_max,
            n_scan=n_scan, n_iter=n_iter, max_roots=max_roots,
            M_solar=M_solar,
        )
        q, u = _multi_crossing_stokes(
            components, orient, r_cross, adiab0, b2, los_perp, los,
            u_hat, v_hat, r_surface, r_max, q_mode0=q_mode0,
            project=project,
        )
        n_cross = jnp.sum(~jnp.isnan(r_cross)).astype(jnp.float64)
        return weight, weight * q, weight * u, is_hit, n_cross

    w, wq, wu, hit, nc = jax.vmap(one_ray)(b2s, los_perps)
    mean_crossings = jnp.sum(jnp.where(hit, nc, 0.0)) / jnp.maximum(
        jnp.sum(hit), 1
    )
    return jnp.sum(w), jnp.sum(wq), jnp.sum(wu), mean_crossings
