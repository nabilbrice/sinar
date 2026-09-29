"""QED vacuum birefringence around a magnetised star.

Transports the polarization of rays leaving a stellar surface through the
field of ``entities.poloidal_fields``, averaged over an energy band.
``put_qed_vacuum`` builds the transport kernels for one star, field and band,
``solve_image`` runs them over a set of rays, and the ``*_observables``
functions and ``band_error`` read off the results.
"""
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ..rays import approx_statictraj_at_r, ds_dr, screen_basis
from ..entities.poloidal_fields import magnetic_field, K_OMEGA

class QEDVacuum(NamedTuple):
    """The transport kernels of one star, magnetic field, and energy band.

    A ray is a surface-anchored tuple (b2, r_hat, m_hat, n_hat),
    as ``rays.ray_from_surface`` and ``rays.ray_from_image`` return it. 
    Stokes vectors S = (Q, U, V)/I are on the screen basis
        (e1, e2) = (n_hat x k_hat, n_hat).

    Attributes
    ----------
    coeffs : Callable
        coeffs(ray, r) -> (|Omega|, (cos zeta, sin zeta), ds/dr, B_perp^2, B^2) at radii r, 
        with zeta twice the field angle.
    probe : Callable
        probe(ray) -> (Gamma_ex, B_perp^2/B^2) on ``r_probe``.
    inner_cut : Callable
        inner_cut(gamma, frac) -> (r_start, crossing flag) from ``probe``.
    band : Callable
        band(ray, r_start, n_steps) -> (band-averaged S, zeta_inf).
    spectral : Callable
        spectral(ray, r_start, n_steps) -> (S per node, weights, zeta_inf).
    reference : Callable
        reference(ray, n_steps) -> (r, zeta(r), band-averaged S, zeta_inf),
        integrated from the surface with no inner cut.
    r_probe : Array
        Radii of the probe.
    band_lambda, band_weights : Array
        Band nodes E / E_c and their weights.
    R, r_max : float
        Stellar radius and outer edge of the integration.
    """
    coeffs: Callable
    probe: Callable
    inner_cut: Callable
    band: Callable
    spectral: Callable
    reference: Callable
    r_probe: Array
    band_lambda: Array
    band_weights: Array
    R: float
    r_max: float

#############
# Energy band
#############
def _band_nodes(band_halfwidth, band_samples, band_seed,
                band_lambda, band_weights):
    """Nodes lambda = E / E_c and normalised weights of the band average."""
    if band_lambda is not None:
        lam = jnp.asarray(band_lambda, float)
        w = (jnp.ones_like(lam) if band_weights is None
             else jnp.asarray(band_weights, float))
        if lam.shape != w.shape:
            raise ValueError("band_lambda and band_weights differ in length")
        return lam, w / w.sum()
    n, h = int(band_samples), float(band_halfwidth)
    if h == 0.0 or n == 1:
        return jnp.ones(1), jnp.ones(1)
    # One jittered node per equal sub-interval.  A regular grid aliases: a ray
    # whose post-crossing phase winds k * 2 pi between nodes would look fully
    # coherent.  Jitter leaves an unbiased O(1/sqrt(n)) residual instead,
    # which band_error estimates.
    jit = jax.random.uniform(jax.random.key(band_seed), (n,))
    lam = 1.0 - h + 2.0 * h * (jnp.arange(n) + jit) / n
    return lam, jnp.full(n, 1.0 / n)

##############
# Coefficients
##############
def _transport_coeffs(field, components, orient, M_solar, energy_keV, R,
                      ray, r):
    """Computes the coefficients of the transport equation from magnetic field and ray trajectory.

    Returns
    -------
    omega : float
        The precession strength of the polarization vector S
    mode : Array(N+1,2)
        The direction in the Q-U plane (polarization mode)
    dsdr : Array(N+1,)
        The proper length scaling at each radius r
    """
    b2, r_hat, m_hat, n_hat = ray
    pos, k, q = approx_statictraj_at_r(r, b2, r_hat, m_hat, R)
    B = jax.vmap(field, in_axes=(None, None, 0))(components, orient, pos)
    e1, e2 = screen_basis(n_hat, k)
    c = jnp.sum(B * e1, axis=-1)
    s = jnp.sum(B * e2, axis=-1)
    bp2 = c * c + s * s
    # The (polarization) modes are axes, defined only up to 180 deg, 
    # so they are expressed as zeta = 2 phi_B. 
    # The double-angle identities give it without an arctan branch cut.
    nrm = jnp.maximum(bp2, 1e-300) # setting a floor for when B is parallel to propagation direction.
    mode = jnp.stack([(c * c - s * s) / nrm, 2.0 * c * s / nrm], axis=1)
    # |Omega| = k_loc * Delta_n with Delta_n = (alpha_F / 30 pi)(B_perp/B_Q)^2,
    # at the locally blueshifted energy.
    omega = K_OMEGA * M_solar * energy_keV / jnp.sqrt(1.0 - 2.0 / r) * bp2
    return omega, mode, ds_dr(r, q), bp2, jnp.sum(B * B, axis=-1)

def _mode_turns(modes):
    """Measures the local mode rotation between consecutive samples."""
    # arctan2(cross, dot) keeps each increment in (-pi, pi], so summing them
    # unwraps zeta without tracking branches.
    cr = modes[:-1, 0] * modes[1:, 1] - modes[:-1, 1] * modes[1:, 0]
    return jnp.arctan2(cr, jnp.sum(modes[:-1] * modes[1:], axis=1))

def _u_grid(r_start, r_max, n_steps):
    """Constructs a uniformly spaced grid in u = -1/r.
    """
    # |Omega| ~ r^-6, so the steps crowd towards the star where it changes.
    return -1.0 / jnp.linspace(-1.0 / r_start, -1.0 / r_max, n_steps + 1)

###########
# Transport
###########
def _rotate(S, v):
    """Applies one norm-preserving rotation of S by the vector v.
    
    Rodrigues' formula for rotation. The magnitude of v specifies the rotation angle.
    """
    th = jnp.linalg.norm(v)
    ax = v / jnp.where(th > 0, th, 1.0)
    ct, st = jnp.cos(th), jnp.sin(th)
    return S * ct + jnp.cross(ax, S) * st + ax * jnp.dot(ax, S) * (1.0 - ct)

def _magnus_scan(lam, r, omega, mode, dsdr):
    """Propagates the Stokes vector using the Magnus scheme.

    For Stokes vector S, transport is dS/ds = Omega x S, Omega = |Omega|(cos zeta, sin zeta, 0),
    here solved in the eigen-frame S' = R_3(-zeta) S.
    """
    # The generator (|Omega|, 0, -dzeta/ds) varies slowly and a locked photon sits still, 
    # so the ~1e8 rad of birefringent phase costs nothing.
    # Each step is one exact rotation by its trapezoid-integrated generator.

    # Convert the radial grid ``r`` to a grid of path lengths in each interval ``ds``
    ds = 0.5 * (dsdr[1:] + dsdr[:-1]) * jnp.diff(r)
    # The accumulated birefringence phase across each interval (at the central energy, Ec) is
    dphi1 = 0.5 * (omega[1:] + omega[:-1]) * ds
    # The apparent precession caused by turning coordinate frame
    dz = _mode_turns(mode)

    # At fixed ray the geometry and zeta do not depend on energy;
    # |Omega| is proportional to it: 
    def one(l):
        # coefficients are shared by all nodes and only the phase increments are rescaled
        dphi = l * dphi1
        # The photon starts in the X mode, anti-parallel to the local
        # adiabatic axis (which includes the dzeta tilt, not just Omega_hat).
        a0 = jnp.array([dphi[0], 0.0, -dz[0]])
        # Apply rotations [dphi, 0, -dz] sequentially by scanning through
        S, _ = jax.lax.scan(
            lambda s_, v: (_rotate(s_, v), None),
            -a0 / jnp.linalg.norm(a0),
            # Stack: [dphi, 0, -dz] for every grid point
            jnp.stack([dphi, jnp.zeros_like(dphi), -dz], axis=1))
        return S

    S = jax.vmap(one)(lam)
    # Transform from the eigenframe back to the screen basis:
    z = jnp.arctan2(mode[0, 1], mode[0, 0]) + jnp.sum(dz)
    cz, sz = jnp.cos(z), jnp.sin(z)
    return jnp.stack([cz * S[:, 0] - sz * S[:, 1],
                      sz * S[:, 0] + cz * S[:, 1], S[:, 2]], axis=1), z

def _spectral_stokes(coeffs, lam, weights, R, r_max, ray, r_start, n_steps):
    """Computes the spectrally resolved Stokes vector for a ray.
    """
    r = _u_grid(jnp.maximum(r_start, R), r_max, n_steps)
    omega, mode, dsdr, _, _ = coeffs(ray, r)
    S, z = _magnus_scan(lam, r, omega, mode, dsdr)
    return S, weights, z

def _band_averaged_stokes(coeffs, lam, weights, R, r_max, ray, r_start, n_steps):
    """Computes the energy-band averaged Stokes vector for a ray bundle.
    """
    # The band average is taken on the Stokes vector itself rather than by
    # projecting onto the asymptotic mode: after a mode crossing the part
    # perpendicular to the mode dephases across the band only if the phase
    # accumulated before freeze-out is large, which near-aligned rays from a
    # polar cap violate.
    S, weights, z = _spectral_stokes(coeffs, lam, weights, R, r_max,
                                 ray, r_start, n_steps)
    return weights @ S, z

def _reference_stokes(coeffs, lam, weights, R, r_max, ray, n_steps):
    """Computes the energy-band averaged Stokes vector for a ray bundle,
    starting at the surface.

    It is useful to measure the effect of skipping the initial propagation.
    """
    r = _u_grid(R, r_max, n_steps)
    omega, mode, dsdr, _, _ = coeffs(ray, r)
    zeta = jnp.concatenate([jnp.zeros(1), jnp.cumsum(_mode_turns(mode))]) \
        + jnp.arctan2(mode[0, 1], mode[0, 0])
    S, z = _magnus_scan(lam, r, omega, mode, dsdr)
    return r, zeta, weights @ S, z

###########
# Inner cut
###########
def _probe_trajectory(coeffs, r_probe, ray):
    """Probes where adiabatic tracking may fail along a ray trajectory.
    
    (Gamma_ex, B_perp^2/B^2) at each radius of r_probe.
    """
    # Gamma_ex = |Omega| / |dzeta/ds| is the ratio that appears in the eigenframe generator.  
    # dzeta/ds is taken pointwise with a jvp: 
    # a segment-averaged Gamma smears narrow mode crossings and can
    # over-estimate it by orders of magnitude.
    def one(r):
        def mode_at(x):
            return coeffs(ray, jnp.atleast_1d(x))[1][0]
        w, dw = jax.jvp(mode_at, (r,), (1.0,))
        omega, _, dsdr, bp2, btot2 = coeffs(ray, jnp.atleast_1d(r))
        zp = jnp.abs(w[0] * dw[1] - w[1] * dw[0]) / dsdr[0]
        return omega[0] / jnp.maximum(zp, 1e-300), bp2[0] / btot2[0]

    return jax.vmap(one)(r_probe)

def _inner_cut(r_probe, gamma_start, bperp_floor, flag_margin, gamma, frac):
    """Chooses how much of the adiabatically locked inner region to skip.
    """
    # Inside the cut the photon is locked, so the integration can start there
    # with a truncation error ~ 1 / gamma_start.  The cut is never taken past
    # a local minimum of B_perp^2/B^2 below the floor: that minimum *is* a
    # crossing, and a Gamma-only rule can step over a narrow one and silently
    # discard the mode conversion.
    n = r_probe.shape[0]
    below = gamma <= gamma_start
    i_g = jnp.where(jnp.any(below), jnp.argmax(below), n - 1)
    is_min = jnp.zeros(n, bool).at[1:-1].set(
        (frac[1:-1] < frac[:-2]) & (frac[1:-1] < frac[2:])
        & (frac[1:-1] < bperp_floor))
    i_b = jnp.where(jnp.any(is_min), jnp.argmax(is_min), n - 1)
    # One probe sample further in, as margin for the coarse probe grid.
    r_start = r_probe[jnp.maximum(jnp.minimum(i_g, i_b) - 1, 0)]
    flag = jnp.min(frac) < bperp_floor * flag_margin
    return r_start, flag

def put_qed_vacuum(R: float, M_solar: float, energy_keV: float,
                   components: Array, orient: Array, *,
                   r_max: float = 800.0,
                   n_probe: int = 96,
                   gamma_start: float = 1000.0,
                   bperp_floor: float = 1.0e-2,
                   flag_margin: float = 3.0,
                   field: Callable = magnetic_field,
                   band_halfwidth: float = 0.05,
                   band_samples: int = 64,
                   band_seed: int = 0,
                   band_lambda=None,
                   band_weights=None) -> QEDVacuum:
    """Constructs the transport callables and diagnostic callables
    for a given star, magnetic field, and energy band.

    Run with JAX_ENABLE_X64: the field's l = 2 GR factors need it.

    Parameters
    ----------
    R : float
        Stellar radius, GM/c^2.
    M_solar : float
    energy_keV : float
        Band centre E_c at infinity.
    components, orient : Array
        Harmonic amplitudes (1 or 6) and magnetic-frame orientation, as
        ``poloidal_fields.magnetic_field`` takes them.
    r_max : float
        Outer edge of the integration; the l = 2 GR factors lose precision
        past ~1e3.
    n_probe : int
        Samples of the inner-cut probe.
    gamma_start : float
        Gamma_ex at the inner cut; the truncation error is ~ 1 / gamma_start.
    bperp_floor : float
        Rays whose min B_perp^2/B^2 falls below it are flagged as crossing a
        mode.
    flag_margin : float
        Safety factor on ``bperp_floor`` for flagging, for probes that are
        extrapolated from a neighbouring ray.
    field : Callable
        ``magnetic_field`` or the drop-in ``magnetic_field_fast``.
    band_halfwidth, band_samples : float, int
        A top-hat band of fractional half-width ``band_halfwidth``, sampled
        at ``band_samples`` jittered nodes.  A zero half-width gives the
        monochromatic solution.
    band_seed : int
        Seed of the jitter.
    band_lambda, band_weights : Array | None
        Explicit nodes E / E_c and weights, e.g. a spectrum times a
        response, in place of the top hat.

    Returns
    -------
    medium : QEDVacuum
    """
    lam, wts = _band_nodes(band_halfwidth, band_samples, band_seed,
                           band_lambda, band_weights)
    r_probe = jnp.geomspace(R, r_max, n_probe)
    coeffs = partial(_transport_coeffs, field, components, orient,
                     M_solar, energy_keV, R)
    return QEDVacuum(
        coeffs=coeffs,
        probe=jax.jit(partial(_probe_trajectory, coeffs, r_probe)),
        inner_cut=partial(_inner_cut, r_probe, gamma_start, bperp_floor,
                          flag_margin),
        band=jax.jit(partial(_band_averaged_stokes, coeffs, lam, wts, R, r_max),
                     static_argnums=(2,)),
        spectral=jax.jit(partial(_spectral_stokes, coeffs, lam, wts, R, r_max),
                         static_argnums=(2,)),
        reference=jax.jit(partial(_reference_stokes, coeffs, lam, wts, R, r_max),
                          static_argnums=(1,)),
        r_probe=r_probe,
        band_lambda=lam,
        band_weights=wts,
        R=R,
        r_max=r_max,
    )

#############
# Observables
#############
def stokes_observables(S: Array):
    """Converts the Stokes vector to degree and angle.

    Parameters
    ----------
    S : Array [..., 3]
        (Q, U, V)/I on the screen basis, e.g. from ``QEDVacuum.band``.

    Returns
    -------
    pa : Array [...]
        Position angle in degrees, from e1 toward e2, in [0, 180).  An
        X-mode photon reads zeta_inf/2 + 90 deg, a converted one zeta_inf/2.
    p_L : Array [...]
        Linear polarization degree.
    v : Array [...]
        Circular polarization V/I.
    """
    q, u, v = S[..., 0], S[..., 1], S[..., 2]
    return jnp.rad2deg(0.5 * jnp.arctan2(u, q)) % 180.0, jnp.hypot(q, u), v

def projected_observables(S: Array, zeta_inf: float):
    """(position angle [deg], degree) of S projected onto the asymptotic mode.

    This is the limit where the phase accumulated after the last mode
    crossing winds many times across the band, so everything perpendicular
    to the mode averages away.  It is wrong when a crossing sits close to
    the final freeze-out, where it removes a real angle offset and all of V.
    With ``band_halfwidth=0`` it is the cheapest approximation to a wide
    band.

    Parameters
    ----------
    S : Array [..., 3]
    zeta_inf : Array [...]
        The asymptotic mode angle returned alongside S.
    """
    d = S[..., 0] * jnp.cos(zeta_inf) + S[..., 1] * jnp.sin(zeta_inf)
    # A negative projection is the X mode, 90 deg from the mode axis.
    pa = jnp.rad2deg(0.5 * zeta_inf) + jnp.where(d > 0.0, 0.0, 90.0)
    return pa % 180.0, jnp.abs(d)

def band_error(S_nodes: Array, weights: Array) -> Array:
    """Estimates the uncertainty in the band-averaged Stokes vector.

    Excludes uncertainties in radial integration, inner-region cut, outer-boundary,
    and model errors.

    Parameters
    ----------
    S_nodes, weights : Array
        The per-node Stokes vectors and weights from ``QEDVacuum.spectral``.

    Returns
    -------
    error : Array [3]
    """
    # The jittered nodes are treated as independent samples of the band,
    # which is conservative for smooth integrands and correct for dephased
    # ones.
    mean = weights @ S_nodes
    var = weights @ (S_nodes - mean) ** 2
    n_eff = 1.0 / jnp.sum(weights ** 2)
    return jnp.sqrt(var / jnp.maximum(n_eff - 1.0, 1.0))

#######
# Image
#######
def solve_image(medium: QEDVacuum, rays, anchor_every: int = 4,
                n_clean: int = 200, n_flagged: int = 1600) -> dict:
    """Band-averaged polarization of a sequence of rays.

    Consecutive rays share a probe, so order them so that neighbours are
    close on the image (e.g. pixel order).

    Parameters
    ----------
    medium : QEDVacuum
    rays : Iterable
        (b2, r_hat, m_hat, n_hat) tuples, as ``rays.ray_from_image`` or
        ``rays.ray_from_surface`` return them.
    anchor_every : int
        Rays per shared probe.
    n_clean, n_flagged : int
        Integration steps for rays without and with a mode crossing.

    Returns
    -------
    solution : dict
        Per ray: pa, p_L, v (see ``stokes_observables``), stokes, r_start and
        flag; plus n_probes, the number of probes taken.
    """
    out = {k: [] for k in ("pa", "p_L", "v", "stokes", "r_start", "flag")}
    n_probes = 0
    anchor = None

    for i, (b2, *basis) in enumerate(rays):
        basis = tuple(jnp.asarray(v) for v in basis)
        # The probe dominates the cost of a short sweep, and its profiles
        # are smooth in the ray, so it is taken on one anchor ray per block,
        # with its derivative in b2, and Taylor-expanded to the others.
        if i % anchor_every == 0:
            anchor = (b2, jax.jvp(lambda x: medium.probe((x, *basis)),
                                  (jnp.asarray(b2),), (1.0,)))
            n_probes += 1
        b2_a, ((g_a, f_a), (dg, df)) = anchor
        db = b2 - b2_a
        r_start, flag = medium.inner_cut(g_a + db * dg, f_a + db * df)
        flag = bool(flag)
        # Mode crossings need resolving, so they get the larger budget.
        S, _ = medium.band((b2, *basis), r_start,
                           n_flagged if flag else n_clean)
        pa, p_L, v = stokes_observables(S)
        out["pa"].append(pa)
        out["p_L"].append(p_L)
        out["v"].append(v)
        out["stokes"].append(S)
        out["r_start"].append(r_start)
        out["flag"].append(flag)

    out = {k: jnp.asarray(v) for k, v in out.items()}
    out["n_probes"] = n_probes
    return out
