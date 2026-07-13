"""Replicate Taverna et al. (2015) Figure 5: energy-phase maps of the
polarization fraction and polarization angle for thermal surface
emission from a rotating magnetized neutron star.

This script only defines the setup: the star (mass, radius), the
magnetic field (polar strength, optional quadrupole), the viewing
geometry (spin-axis inclinations chi and xi, rotational phase), and the
emission pattern on the surface (a magnetized-conduction temperature
map feeding a redshifted blackbody). All of the physics follows from
the library modules: ``harmonics.magnetic_field`` provides the
Schwarzschild-corrected field, and ``birefringence`` provides adiabatic
freeze-out along Beloborodov trajectories, transport of the frozen
polarization to the observer, and the Stokes representation.

By default every trajectory freezes at its own adiabatic radius. With
``--single-radius`` all trajectories share the adiabatic radius of the
central (b = 0) ray, the analogue of the single scalar radius used by
Taverna et al.

Example::

    python fig5.py --quadrupole 0.5
    python fig5.py --quadrupole 0.2 0.4 -0.3 0.25 -0.15 --single-radius
"""
from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from sinar.rays import normalize
from sinar.renderers import construct_obs_rays
from sinar.entities.colors import blackbody
from sinar.entities.shapes import rotation, rotate_about_axis
from sinar.entities.harmonics import magnetic_field
from sinar.entities.birefringence import (
    adiabatic_radius,
    observed_stokes,
    polarization_fraction_angle,
)

# ----------------------------------------------------------------------
# Setup: star, field, viewing geometry, emission pattern
# (Taverna et al. 2015, Fig. 5)
# ----------------------------------------------------------------------
M_SOLAR = 1.4                    # NS mass [M_sun]
R_NS_KM = 10.0                   # NS radius [km]
B_P = 1.0e13                     # polar magnetic field [G]
XI_DEG = 5.0                     # magnetic-axis inclination to spin axis
CHI_DEG = (0.0, 15.0, 30.0)      # LOS inclinations to spin axis
T_POLE = 0.150                   # polar temperature [keV]
T_EQ = 0.100                     # equatorial temperature floor [keV]

R_CODE = R_NS_KM / (1.47662 * M_SOLAR)   # stellar radius in GM/c^2
REDSHIFT = float(np.sqrt(1.0 - 2.0 / R_CODE))
R_FREEZE_MAX = 1500.0            # outer bound of the freeze-radius scan

LOS = jnp.array([0.0, 0.0, 1.0])
U_HAT = jnp.array([1.0, 0.0, 0.0])   # psi = 0 polarimeter frame
V_HAT = jnp.array([0.0, 1.0, 0.0])


def field_components(quadrupole=(0.0,)):
    """Multipole amplitudes for ``harmonics.magnetic_field``.

    ``c_dip = B_P R^3`` makes the flat-space polar dipole field exactly
    B_P; each quadrupole generator amplitude ``q_i B_P R^4`` gives that
    component a maximum surface field of ``q_i B_P``. One quadrupole
    value selects the axisymmetric generator aligned with the dipole;
    five values are the full Page & Sarmiento set Q_0..Q_4.
    """
    quad = tuple(float(q) for q in quadrupole)
    if len(quad) == 1:
        quad = (quad[0], 0.0, 0.0, 0.0, 0.0)
    if len(quad) != 5:
        raise ValueError(f"quadrupole takes 1 or 5 values; got {len(quad)}")
    if all(q == 0.0 for q in quad):
        return jnp.array([B_P * R_CODE**3])
    return jnp.concatenate(
        [jnp.array([B_P * R_CODE**3]), B_P * R_CODE**4 * jnp.asarray(quad)]
    )


@jax.jit
def orientation(chi: float, xi: float, gamma: float):
    """Magnetic-frame orientation at rotational phase gamma.

    Row-vector convention: the magnetic axis is ``[0,0,1] @ orient``.
    The LOS is inclined by chi and the magnetic axis by xi to the spin
    axis, reproducing equation (13) of Taverna et al. (2015).
    """
    spin_axis = jnp.array([0.0, 0.0, 1.0]) @ rotation(phi=chi, theta=0.0)
    phase_zero = rotation(phi=chi - xi, theta=0.0)
    return phase_zero @ rotate_about_axis(-gamma, axis=spin_axis)


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
# Energy-phase maps
# ----------------------------------------------------------------------
def compute_maps(components, chi_deg, energies, rot_phases, n_pix,
                 single_radius=False, oversize=1.02):
    """PF and PA maps over (phase, energy) for one viewing geometry."""
    chi, xi = np.deg2rad(chi_deg), np.deg2rad(XI_DEG)

    b_max = R_CODE / np.sqrt(1.0 - 2.0 / R_CODE)
    b2s, los_perps, _ = construct_obs_rays(
        n_pix, n_pix, oversize * b_max, los=LOS
    )

    pf = np.zeros((len(rot_phases), len(energies)))
    pa = np.zeros_like(pf)
    q_over_i = np.zeros_like(pf)
    u_over_i = np.zeros_like(pf)

    for i_g, rot_phase in enumerate(rot_phases):
        orient = orientation(chi, xi, rot_phase)
        for i_e, energy in enumerate(energies):
            r_freeze = 0.0
            if single_radius:
                r_freeze = adiabatic_radius(
                    energy, components, orient, 0.0, V_HAT, LOS,
                    R_CODE, R_FREEZE_MAX,
                )
                r_freeze = jnp.where(jnp.isnan(r_freeze), R_FREEZE_MAX,
                                     r_freeze)
            I, Q, U = observed_stokes(
                energy, components, orient, b2s, los_perps,
                LOS, U_HAT, V_HAT, R_CODE,
                surface_intensity=surface_intensity,
                per_ray_radius=not single_radius,
                r_freeze=r_freeze,
                r_max=R_FREEZE_MAX,
                M_solar=M_SOLAR,
            )
            pf_i, pa_i = polarization_fraction_angle(I, Q, U)
            pf[i_g, i_e], pa[i_g, i_e] = float(pf_i), float(pa_i)
            q_over_i[i_g, i_e] = float(Q / I)
            u_over_i[i_g, i_e] = float(U / I)

    return pf, pa, q_over_i, u_over_i


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
def plot_fig5(energies, rot_phases, results, outfile, subtitle=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_col = len(results)
    fig, axes = plt.subplots(
        2, n_col, figsize=(4.2 * n_col, 7.2),
        sharex=True, sharey=True, constrained_layout=True,
    )
    axes = np.atleast_2d(axes.reshape(2, n_col))

    E_edges = np.geomspace(energies[0], energies[-1], len(energies) + 1)
    g_edges = np.linspace(rot_phases[0], rot_phases[-1], len(rot_phases) + 1)

    for j, (chi_deg, pf, pa, _, _) in enumerate(results):
        pf_mesh = axes[0, j].pcolormesh(
            E_edges, g_edges, pf, cmap="jet", vmin=0.0, vmax=1.0,
            shading="auto",
        )
        pa_mesh = axes[1, j].pcolormesh(
            E_edges, g_edges, pa, cmap="jet", vmin=0.0, vmax=180.0,
            shading="auto",
        )
        axes[0, j].set_title(rf"$\chi = {chi_deg:.0f}^\circ$")
        axes[1, j].set_xlabel(r"$E$ [keV]")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xscale("log")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\gamma$ [rad]")
        ax.set_yticks([0, np.pi, 2.0 * np.pi])
        ax.set_yticklabels(["0", r"$\pi$", r"$2\pi$"])

    fig.colorbar(pf_mesh, ax=axes[0, :], label=r"$\Pi_L$", shrink=0.9)
    fig.colorbar(pa_mesh, ax=axes[1, :], label=r"$\chi_p$ [deg]", shrink=0.9)
    fig.suptitle(
        rf"$B_P = {B_P:.0e}$ G, $R = {R_NS_KM:.0f}$ km, "
        rf"$M = {M_SOLAR}\,M_\odot$, $\xi = {XI_DEG:.0f}^\circ${subtitle}"
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
    p.add_argument("--n-phase", type=int, default=32)
    p.add_argument("--e-min", type=float, default=1e-3, help="keV")
    p.add_argument("--e-max", type=float, default=10.0, help="keV")
    p.add_argument("--chi", type=float, nargs="+", default=list(CHI_DEG),
                   help="LOS inclinations [deg]; one panel column each")
    p.add_argument("--quadrupole", type=float, nargs="+", default=[0.0],
                   help="1 value (axisymmetric) or 5 values (Q_0..Q_4), "
                        "as multiples of B_P")
    p.add_argument("--single-radius", action="store_true",
                   help="freeze all rays at the central-ray adiabatic "
                        "radius instead of per-trajectory radii")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-npz", type=str, default=None,
                   help="empty string disables")
    args = p.parse_args()

    components = field_components(args.quadrupole)
    quad_on = components.shape[0] > 1

    tag = "single" if args.single_radius else "per_ray"
    if quad_on:
        q_tag = "_".join(f"{q:g}" for q in args.quadrupole)
        tag += f"_q{q_tag}".replace(".", "p").replace("-", "m")
    if args.out is None:
        args.out = f"taverna_fig5_{tag}.png"
    if args.save_npz is None:
        args.save_npz = f"taverna_fig5_{tag}.npz"

    if args.n_pix % 2:
        args.n_pix += 1
        print(f"n_pix increased to even value {args.n_pix} to avoid b = 0")

    energies = np.geomspace(args.e_min, args.e_max, args.n_energy)
    gammas = np.linspace(0.0, 2.0 * np.pi, args.n_phase, endpoint=False)

    results = []
    for chi_deg in args.chi:
        t0 = time.time()
        pf, pa, q_i, u_i = compute_maps(
            components, chi_deg, energies, gammas, args.n_pix,
            single_radius=args.single_radius,
        )
        print(f"chi = {chi_deg:5.1f} deg done in {time.time() - t0:6.1f} s "
              f"| PF range [{pf.min():.3f}, {pf.max():.3f}] "
              f"| PA range [{pa.min():.1f}, {pa.max():.1f}] deg")
        results.append((chi_deg, pf, pa, q_i, u_i))

    if args.save_npz:
        np.savez(
            args.save_npz,
            energies=energies, gammas=gammas,
            chi_deg=np.array([r[0] for r in results]),
            pf=np.stack([r[1] for r in results]),
            pa=np.stack([r[2] for r in results]),
            q_over_i=np.stack([r[3] for r in results]),
            u_over_i=np.stack([r[4] for r in results]),
            quadrupole=np.array(args.quadrupole),
            single_radius=np.array(args.single_radius),
        )
        print(f"saved {args.save_npz}")

    subtitle = ""
    if quad_on:
        q_str = ",".join(f"{q:g}" for q in args.quadrupole)
        subtitle += rf", $Q/B_P = ({q_str})$"
    if args.single_radius:
        subtitle += ", single radius"
    plot_fig5(energies, gammas, results, args.out, subtitle=subtitle)


if __name__ == "__main__":
    main()