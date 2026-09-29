"""Contours of the polarization degree of thermal surface emission from a
magnetized neutron star, over photon energy and viewing angle.

A copy of ``fig5.py`` without the rotation: the dipole axis is fixed
at the viewing angle iota from the line of sight, and the map runs over
(iota, energy) rather than (rotational phase, energy).  The star, field,
emission pattern and render chain are those of ``fig5.py``; the Stokes
vector of each ray is integrated with the ODE of
``media.qed_polarization``.

The star keeps the spin axis of ``fig5.py``, inclined by XI_DEG to the
dipole, so that ``--quadrupole`` describes the same star in both scripts.
Each iota is ``fig5.py``'s view at rotational phase 0 with
chi = iota + XI_DEG: the spin axis, dipole axis and line of sight are
coplanar, with the dipole between the other two.  Without a quadrupole
the spin axis plays no role.

1. initialise the observer-at-infinity ray invariants
   (``construct_obs_rays``) and anchor each hitting ray at the surface
   (``rays.reanchor``);
2. build the transport kernels for the star, field and energy band
   (``put_qed_vacuum``), probe each ray and choose its inner cut, the
   radius inside which the photon is adiabatically locked to the X mode;
3. integrate the Stokes vector from the inner cut to ``r_max`` with the
   Magnus sweep, averaged over the band; rays that cross a mode get
   ``--n-flagged`` steps, the rest ``--n-clean``;
4. shade: the blackbody spectral intensity is the ray weight, and the
   band-averaged Stokes vector, rotated from the ray's screen basis onto
   the polarimeter frame, gives the Stokes observables.

Example::

    python poldeg_contours.py
    python poldeg_contours.py --quadrupole 0.5 5 0 --n-iota 31   # aligned
    python poldeg_contours.py --quadrupole 0.5 20 90
"""
from __future__ import annotations

import argparse
import time
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from sinar.rays import normalize, observer_anchor, reanchor
from sinar.renderers import construct_obs_rays
from sinar.entities.colors import blackbody
from sinar.entities.shapes import rotation, rotate_about_axis
from sinar.entities.poloidal_fields import (
    magnetic_field,
    schwarzschild_factors,
)
from sinar.media.qed_polarization import put_qed_vacuum, stokes_observables

# ----------------------------------------------------------------------
# Setup: star, field, viewing geometry, emission pattern
# (as in fig5.py, Taverna et al. 2015)
# ----------------------------------------------------------------------
M_SOLAR = 1.4                    # NS mass [M_sun]
R_NS_KM = 10.0                   # NS radius [km]
B_P_PHYSICAL = 1.0e13            # physical magnetic field on the surface [G] (after GR correction)
XI_DEG = 5.0                     # dipole-axis inclination to spin axis, as in fig5.py
T_POLE = 0.150                   # polar temperature [keV]
T_EQ = 0.100                     # equatorial temperature floor [keV]

R_CODE = R_NS_KM / (1.47662 * M_SOLAR)   # stellar radius in GM/c^2
REDSHIFT = float(np.sqrt(1.0 - 2.0 / R_CODE))
R_MAX = 800.0                    # outer edge of the Stokes integration

# Although it looks like for actually recreating the plot, they use the B_P = B_P_PHYSICAL
B_P = B_P_PHYSICAL / float(schwarzschild_factors(1, R_CODE)[0])
# Flat-space polar field of a unit quadrupole whose GR-corrected polar
# field is B_P_PHYSICAL, so ``--quadrupole`` strengths are physical ratios.
B_Q = B_P_PHYSICAL / float(schwarzschild_factors(2, R_CODE)[0])

# Keeps b2 clear of the singular points of a surface-anchored ray, as
# ``rays._safe_b2`` does: sin psi ~ sqrt(b2) has an infinite derivative at
# b2 = 0, and cos(alpha_R) rounds to NaN at the tangential limit.
B2_LIMIT = R_CODE**2 / (1.0 - 2.0 / R_CODE)
B2_FLOOR = (1.0e-6 * R_CODE) ** 2

LOS = jnp.array([0.0, 0.0, 1.0])
U_HAT = jnp.array([1.0, 0.0, 0.0])   # psi = 0 polarimeter frame
V_HAT = jnp.array([0.0, 1.0, 0.0])

# Rays per compiled sweep; groups are padded up to a multiple of it so the
# sweeps compile once per step count.
CHUNK = 512


def field_components(quad_strength=None):
    """Multipole amplitudes for ``poloidal_fields.magnetic_field``.

    ``c_dip = B_P R^3`` makes the GR-corrected polar dipole field
    B_P_PHYSICAL.  ``quad_strength`` q adds the axisymmetric quadrupole
    generator Q_0 with GR-corrected polar field q B_P_PHYSICAL, so q is the
    physical ratio of the two polar surface fields; its axis is set by
    ``orientation``.
    """
    c_dip = B_P * R_CODE**3
    if not quad_strength:
        return jnp.array([c_dip])
    return jnp.array([c_dip, quad_strength * B_Q * R_CODE**4,
                      0.0, 0.0, 0.0, 0.0])


@jax.jit
def axis_frame(chi: float, tilt: float, gamma: float):
    """Frame of an axis tilted from the spin axis, at rotational phase gamma.

    Row-vector convention: the axis is ``[0,0,1] @ frame``.  The LOS is
    inclined by chi and the axis by tilt to the spin axis, as in
    ``fig5.py``.
    """
    spin_axis = jnp.array([0.0, 0.0, 1.0]) @ rotation(phi=chi, theta=0.0)
    phase_zero = rotation(phi=chi - tilt, theta=0.0)
    return phase_zero @ rotate_about_axis(-gamma, axis=spin_axis)


def orientation(iota: float, xi: float, quad=None):
    """Magnetic-frame orientation at viewing angle iota.

    The dipole axis is (sin iota, 0, cos iota), inclined by iota to the LOS
    (+z) and tilted toward U_HAT, with the spin axis a further xi beyond
    it.  ``quad`` = (q, xi_q, phi_q) adds a quadrupole axis inclined by
    xi_q to the spin axis and phi_q ahead of the dipole about it (radians),
    as in ``fig5.orientation``: (xi, 0) aligns the two axes.  The result is
    then the stacked ``[orient_dipole, orient_quadrupole]`` that
    ``magnetic_field`` takes.
    """
    chi = iota + xi
    dip = axis_frame(chi, xi, 0.0)
    if quad is None:
        return dip
    _, xi_q, phi_q = quad
    return jnp.stack([dip, axis_frame(chi, xi_q, phi_q)])


def surface_intensity(components, orient, position, energy_keV):
    """Emission pattern: redshifted blackbody with a magnetized map.

    The magnetized-envelope conduction map T = Tp |cos theta_B|^(1/2)
    (floored at Te) sets the local temperature from the angle between
    the total surface field and the radial direction; the observed
    weight is the blackbody intensity at the redshifted temperature,
    using g^3 I(E/g; T) = I(E; g T).
    """
    B = magnetic_field(components, orient, position)
    cos_thB = jnp.vecdot(normalize(B), normalize(position))
    T_loc = jnp.maximum(T_POLE * jnp.sqrt(jnp.abs(cos_thB)), T_EQ)
    return blackbody(REDSHIFT * T_loc, energy_keV)


# ----------------------------------------------------------------------
# Render chain (per ray): anchor -> inner cut -> Stokes sweep -> shade
# ----------------------------------------------------------------------
def hitting_rays(b2s, los_perps):
    """The sky rays that reach the surface, with b2 kept off its singular points."""
    b2s, los_perps = np.asarray(b2s), np.asarray(los_perps)
    hit = b2s <= B2_LIMIT * (1.0 + 1e-12)
    b2s = np.maximum(b2s[hit], B2_FLOOR)
    b2s = np.where(np.abs(b2s - B2_LIMIT) <= 1e-12 * B2_LIMIT,
                   B2_LIMIT * (1.0 - 1e-14), b2s)
    return jnp.asarray(b2s), jnp.asarray(los_perps[hit])


def image_ray(b2, los_perp):
    """Surface-anchored ray (b2, r_hat, m_hat, n_hat) of a sky pixel."""
    r_hat, m_hat = reanchor(b2, *observer_anchor(LOS, los_perp), R_CODE)
    return b2, r_hat, m_hat, normalize(jnp.cross(r_hat, m_hat))


def screen_to_sky(S, n_hat):
    """(Q, U, V)/I from the ray's screen basis onto (U_HAT, V_HAT).

    At the observer the screen basis is (e1, e2) = (n_hat x LOS, n_hat),
    which has the handedness of (U_HAT, V_HAT) about LOS, so the Stokes
    vector turns by twice the angle of e1 from U_HAT.
    """
    e1 = jnp.cross(n_hat, LOS)
    two_th = 2.0 * jnp.arctan2(jnp.dot(e1, V_HAT), jnp.dot(e1, U_HAT))
    c, s = jnp.cos(two_th), jnp.sin(two_th)
    return jnp.stack([c * S[0] - s * S[1], s * S[0] + c * S[1], S[2]])


def make_render(band_halfwidth, band_samples):
    """Jitted (cut, sweep) stages for one choice of energy band.

    Energy, field and orientation are traced arguments, so each stage
    compiles once per run rather than once per ``put_qed_vacuum``.
    """
    def medium(energy_keV, components, orient):
        return put_qed_vacuum(
            R_CODE, M_SOLAR, energy_keV, components, orient, r_max=R_MAX,
            band_halfwidth=band_halfwidth, band_samples=band_samples,
        )

    @jax.jit
    def cut(energy_keV, components, orient, b2s, los_perps):
        """Per ray: inner-cut radius, mode-crossing flag, intensity weight."""
        m = medium(energy_keV, components, orient)

        def one(b2, los_perp):
            ray = image_ray(b2, los_perp)
            r_start, flag = m.inner_cut(*m.probe(ray))
            weight = surface_intensity(components, orient, R_CODE * ray[1],
                                       energy_keV)
            return r_start, flag, weight

        return jax.vmap(one)(b2s, los_perps)

    @partial(jax.jit, static_argnames=("n_steps",))
    def sweep(energy_keV, components, orient, b2s, los_perps, r_starts,
              n_steps):
        """Per ray: band-averaged (Q, U, V)/I on the polarimeter frame."""
        m = medium(energy_keV, components, orient)

        def one(b2, los_perp, r_start):
            ray = image_ray(b2, los_perp)
            S, _ = m.band(ray, r_start, n_steps)
            return screen_to_sky(S, ray[3])

        return jax.vmap(one)(b2s, los_perps, r_starts)

    return cut, sweep


def sky_stokes(render, energy_keV, components, orient, b2s, los_perps,
               n_clean, n_flagged):
    """Sky-integrated Stokes (I, Q, U, V) and the flagged-ray fraction."""
    cut, sweep = render
    r_starts, flags, weights = cut(energy_keV, components, orient,
                                   b2s, los_perps)
    flags = np.asarray(flags)

    # Mode crossings need resolving, so they get the larger budget.
    S = np.zeros((b2s.shape[0], 3))
    for mask, n_steps in ((~flags, n_clean), (flags, n_flagged)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        # Pad with the group's first ray so every call has shape (CHUNK,).
        padded = np.resize(idx, -(-idx.size // CHUNK) * CHUNK)
        padded[idx.size:] = idx[0]
        for lo in range(0, padded.size, CHUNK):
            sel = padded[lo:lo + CHUNK]
            n_valid = min(CHUNK, idx.size - lo)
            S_c = sweep(energy_keV, components, orient, b2s[sel],
                        los_perps[sel], r_starts[sel], n_steps=n_steps)
            S[sel[:n_valid]] = np.asarray(S_c)[:n_valid]

    w = np.asarray(weights)
    I = w.sum()
    Q, U, V = w @ S
    return I, Q, U, V, flags.mean()


# ----------------------------------------------------------------------
# Energy-inclination maps
# ----------------------------------------------------------------------
def compute_maps(render, components, quad, iotas_deg, energies, n_pix,
                 n_clean, n_flagged, oversize=1.02):
    """PF and PA maps over (energy, iota).

    ``quad`` is None or (q, xi_q, phi_q) with angles in radians, as
    ``orientation`` takes it.
    """
    b_max = R_CODE / np.sqrt(1.0 - 2.0 / R_CODE)
    b2s, los_perps, _ = construct_obs_rays(
        n_pix, n_pix, oversize * b_max, los=LOS
    )
    b2s, los_perps = hitting_rays(b2s, los_perps)

    pf = np.zeros((len(energies), len(iotas_deg)))
    pa = np.zeros_like(pf)
    q_over_i = np.zeros_like(pf)
    u_over_i = np.zeros_like(pf)
    v_over_i = np.zeros_like(pf)
    flagged = np.zeros_like(pf)

    for i_i, iota_deg in enumerate(iotas_deg):
        t0 = time.time()
        orient = orientation(np.deg2rad(iota_deg), np.deg2rad(XI_DEG), quad)
        for i_e, energy in enumerate(energies):
            I, Q, U, V, f = sky_stokes(
                render, energy, components, orient, b2s, los_perps,
                n_clean, n_flagged,
            )
            S = np.array([Q, U, V]) / I
            pa_i, pf_i, _ = stokes_observables(S)
            pf[i_e, i_i], pa[i_e, i_i] = float(pf_i), float(pa_i)
            q_over_i[i_e, i_i], u_over_i[i_e, i_i], v_over_i[i_e, i_i] = S
            flagged[i_e, i_i] = f
        print(f"iota = {iota_deg:5.1f} deg done in {time.time() - t0:6.1f} s "
              f"| PF range [{pf[:, i_i].min():.3f}, {pf[:, i_i].max():.3f}] "
              f"| flagged rays {100.0 * flagged[:, i_i].mean():.1f}%")

    return pf, pa, q_over_i, u_over_i, v_over_i, flagged


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def plot_contours(energies, iotas_deg, pf, outfile, subtitle=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    levels = np.linspace(0.0, 1.0, 11)
    filled = ax.contourf(iotas_deg, energies, pf, levels=levels,
                         cmap="viridis", extend="neither")
    lines = ax.contour(iotas_deg, energies, pf, levels=levels[1:-1],
                       colors="k", linewidths=0.6)
    ax.clabel(lines, fmt="%.1f", fontsize=7)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\iota$ [deg]")
    ax.set_ylabel(r"$E$ [keV]")
    ax.set_xlim(iotas_deg[0], iotas_deg[-1])
    ax.set_xticks(np.arange(0.0, 91.0, 15.0))
    fig.colorbar(filled, ax=ax, label=r"$\Pi_L$")
    ax.set_title(
        rf"$B_P = {B_P:.0e}$ G, $R = {R_NS_KM:.0f}$ km, "
        rf"$M = {M_SOLAR}\,M_\odot${subtitle}",
        fontsize=10,
    )
    fig.savefig(outfile, dpi=160)
    print(f"saved {outfile}")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-pix", type=int, default=64,
                   help="sky-plane rays per side; even avoids the b=0 ray")
    p.add_argument("--n-energy", type=int, default=16)
    p.add_argument("--e-min", type=float, default=0.1, help="keV")
    p.add_argument("--e-max", type=float, default=10.0, help="keV")
    p.add_argument("--n-iota", type=int, default=19,
                   help="viewing angles from 0 to 90 deg inclusive")
    p.add_argument("--quadrupole", type=float, nargs=3, default=None,
                   metavar=("Q", "XI_Q", "PHI_Q"),
                   help="axisymmetric quadrupole: polar surface field as a "
                        "multiple of the dipole's (both GR-corrected), tilt "
                        "of its axis from the spin axis [deg], and azimuth "
                        "about the spin axis ahead of the dipole [deg]; "
                        f"'Q {XI_DEG:g} 0' aligns it with the dipole")
    p.add_argument("--band-halfwidth", type=float, default=0.05,
                   help="fractional half-width of the band averaged at "
                        "each energy; 0 is monochromatic")
    # Sampling error of the band average falls only as 1/sqrt(n), so a few
    # nodes are enough next to the other errors of the sky sum.
    p.add_argument("--band-samples", type=int, default=16,
                   help="jittered band nodes; ignored when --band-halfwidth is 0.0")
    p.add_argument("--n-clean", type=int, default=200,
                   help="Stokes steps for rays without a mode crossing")
    # Checked against 12800 steps on the sky-summed Stokes vector, for the
    # pure dipole and quadrupoles up to Q = 2 at tilts to 60 deg, 0.01-10 keV:
    # 200 steps is within |dS| <= 2e-3 in most cases and 8e-3 at worst
    # (Q = 2), i.e. under 1% in PF.  Doubling the steps roughly halves it.
    p.add_argument("--n-flagged", type=int, default=200,
                   help="Stokes steps for rays with a mode crossing")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-npz", type=str, default=None,
                   help="empty string disables")
    args = p.parse_args()

    quad = None
    if args.quadrupole is not None and args.quadrupole[0] != 0.0:
        q, xi_q, phi_q = args.quadrupole
        quad = (q, np.deg2rad(xi_q), np.deg2rad(phi_q))
    components = field_components(None if quad is None else quad[0])

    tag = "poldeg_contours"
    if quad is not None:
        tag += (f"_q{q:g}_xi{xi_q:g}_phi{phi_q:g}"
                .replace(".", "p").replace("-", "m"))
    if args.out is None:
        args.out = f"{tag}.png"
    if args.save_npz is None:
        args.save_npz = f"{tag}.npz"

    if args.n_pix % 2:
        args.n_pix += 1
        print(f"n_pix increased to even value {args.n_pix} to avoid b = 0")

    energies = np.geomspace(args.e_min, args.e_max, args.n_energy)
    iotas_deg = np.linspace(0.0, 90.0, args.n_iota)
    render = make_render(args.band_halfwidth, args.band_samples)

    t0 = time.time()
    pf, pa, q_i, u_i, v_i, flagged = compute_maps(
        render, components, quad, iotas_deg, energies, args.n_pix,
        args.n_clean, args.n_flagged,
    )
    print(f"done in {time.time() - t0:6.1f} s "
          f"| PF range [{pf.min():.3f}, {pf.max():.3f}] "
          f"| flagged rays {100.0 * flagged.mean():.1f}%")

    if args.save_npz:
        np.savez(
            args.save_npz,
            energies=energies, iota_deg=iotas_deg,
            pf=pf, pa=pa, q_over_i=q_i, u_over_i=u_i, v_over_i=v_i,
            flagged_fraction=flagged,
            # (Q, XI_Q [deg], PHI_Q [deg]); empty without a quadrupole.
            quadrupole=np.array([] if quad is None else args.quadrupole),
            xi_deg=np.array(XI_DEG),
            band_halfwidth=np.array(args.band_halfwidth),
            band_samples=np.array(args.band_samples),
            n_clean=np.array(args.n_clean),
            n_flagged=np.array(args.n_flagged),
        )
        print(f"saved {args.save_npz}")

    subtitle = ""
    if quad is not None:
        subtitle += (rf", $B_Q/B_P = {q:g}$, "
                     rf"$(\xi_Q, \phi_Q) = ({xi_q:g}^\circ, {phi_q:g}^\circ)$")
    plot_contours(energies, iotas_deg, pf, args.out, subtitle=subtitle)


if __name__ == "__main__":
    main()
