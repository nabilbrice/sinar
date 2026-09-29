"""Step convergence of the image-summed Stokes vector of ``examples/fig5.py``.

Checks whether the ``--n-flagged`` default of ``fig5.py`` and
``poldeg_contours.py`` is enough for their maps, which only see the
intensity-weighted sum over the image.  Each case is a field and viewing
geometry, run through the script's own render stages (``make_render``) at
several step counts and compared against a finely stepped reference.

    pytest -m diagnostic tests/diagnostics/diag_image_steps.py

Like the other diagnostics each test draws a figure rather than asserting a
tolerance, and fails only on non-finite values.  Errors are Euclidean
distances between Stokes vectors (Q, U, V)/I, so 1e-2 is roughly a 1% error
in the polarization degree.
"""
import importlib.util
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from sinar.renderers import construct_obs_rays
from sinar.media.qed_polarization import stokes_observables

pytestmark = pytest.mark.diagnostic

_FIG5 = Path(__file__).parents[2] / "examples" / "fig5.py"
_spec = importlib.util.spec_from_file_location("fig5", _FIG5)
fig5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig5)

# The scripts' defaults; ``fig5.py`` sets them in its argument parser.
N_FLAGGED = 200
BAND_HALFWIDTH, BAND_SAMPLES = 0.05, 16

N_PIX = 32
N_STEPS = [100, 200, 400, 800, 1600, 6400]
N_REF = 12800
ENERGIES = [0.01, 0.1, 1.0, 5.0, 10.0]

XI = fig5.XI_DEG
# id -> (title, quadrupole (Q, XI_Q [deg], PHI_Q [deg]) or None,
#        chi [deg], rotational phase [rad]).  The poldeg cases are
#        ``poldeg_contours.py`` at viewing angle iota: chi = iota + XI, phase 0.
CASES = {
    "dipole": ("dipole", None, 30.0, 0.0),
    "aligned": ("quadrupole 0.5, aligned", (0.5, XI, 0.0), 30.0, 1.0),
    "tilted": ("quadrupole 0.5, tilt 20, azimuth 90", (0.5, 20.0, 90.0),
               30.0, 1.0),
    "strong": ("quadrupole 1.0, tilt 45, azimuth 180", (1.0, 45.0, 180.0),
               15.0, 2.5),
    "stronger": ("quadrupole 2.0, tilt 60, azimuth 90", (2.0, 60.0, 90.0),
                 60.0, 0.5),
    "poldeg_iota0": ("poldeg iota = 0, quadrupole 0.5, tilt 20, azimuth 90",
                     (0.5, 20.0, 90.0), 0.0 + XI, 0.0),
    "poldeg_iota60": ("poldeg iota = 60, quadrupole 1.0, tilt 45, azimuth 180",
                      (1.0, 45.0, 180.0), 60.0 + XI, 0.0),
}


@pytest.fixture(scope="module")
def render():
    """The script's (cut, sweep) stages and the rays that hit the star."""
    b2s, los_perps, _ = construct_obs_rays(
        N_PIX, N_PIX, 1.02 * np.sqrt(fig5.B2_LIMIT), los=fig5.LOS)
    b2s, los_perps = fig5.hitting_rays(b2s, los_perps)
    return fig5.make_render(BAND_HALFWIDTH, BAND_SAMPLES), b2s, los_perps


def _field(quad, chi_deg, gamma):
    xi = np.deg2rad(XI)
    if quad is None:
        return fig5.field_components(), fig5.orientation(
            np.deg2rad(chi_deg), xi, gamma)
    q, xi_q, phi_q = quad
    quad = (q, np.deg2rad(xi_q), np.deg2rad(phi_q))
    return (fig5.field_components(q),
            fig5.orientation(np.deg2rad(chi_deg), xi, gamma, quad))


def _floor(x):
    """Keeps exact zeros visible on log axes."""
    return np.maximum(np.asarray(x, float), 1e-17)


@pytest.mark.parametrize("case_id", list(CASES))
def test_image_step_convergence(case_id, render, savefig):
    """Error of the image-summed and single-ray Stokes vectors against n.

    Every ray is swept at the same n, so n plays the role of
    ``--n-flagged``.  Left: the image-summed S that the maps use; right:
    the worst single ray, whose errors largely cancel in the sum.
    """
    (cut, sweep), b2s, los_perps = render
    title, quad, chi_deg, gamma = CASES[case_id]
    components, orient = _field(quad, chi_deg, gamma)

    fig, (ax_sum, ax_ray) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    flagged = []
    for c, energy in zip(colors, ENERGIES):
        r_starts, flags, w = cut(energy, components, orient, b2s, los_perps)
        flagged.append(float(jnp.mean(flags)))

        def image(n):
            S = sweep(energy, components, orient, b2s, los_perps, r_starts,
                      n_steps=n)
            assert jnp.all(jnp.isfinite(S))
            return S, (w @ S) / jnp.sum(w)

        S_ref, sum_ref = image(N_REF)
        e_sum, e_ray = [], []
        for n in N_STEPS:
            S, s = image(n)
            e_sum.append(float(jnp.linalg.norm(s - sum_ref)))
            e_ray.append(float(jnp.max(jnp.linalg.norm(S - S_ref, axis=1))))
        _, pf, _ = stokes_observables(sum_ref)
        ax_sum.loglog(N_STEPS, _floor(e_sum), "o-", color=c, ms=3,
                      label=f"E = {energy:g} keV, PF = {float(pf):.3f}")
        ax_ray.loglog(N_STEPS, _floor(e_ray), "o-", color=c, ms=3)

    for ax in (ax_sum, ax_ray):
        x = np.asarray(N_STEPS, float)
        ax.plot(x, 1e-1 * x[0] / x, color="0.6", lw=1, label="1 / n")
        ax.axvline(N_FLAGGED, color="k", lw=0.8, ls="-.")
        ax.set_xlabel("n_steps (all rays)")
        ax.grid(True, which="major", alpha=0.3)
    ax_sum.axhline(1e-2, color="C3", lw=0.8, ls=":")
    ax_sum.set_ylabel(f"|S(n) - S({N_REF})|")
    ax_sum.set_title("image sum (dotted: ~1% in PF)")
    ax_ray.set_title("worst single ray")
    ax_sum.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{title}; chi = {chi_deg:g} deg, phase = {gamma:g} rad; "
                 f"flagged rays {100 * np.mean(flagged):.0f}% "
                 f"(dash-dot: --n-flagged default)")
    savefig(fig, f"image_steps_{case_id}")
