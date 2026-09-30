import jax
import pytest
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sinar.entities.poloidal_fields import K_OMEGA
from sinar.media.qed_polarization import (
    N_PROBE_TRACED, _magnus_scan, _u_grid, adaptive_r_max, put_qed_vacuum,
    solve_image, truncation_bound)
from sinar.media.vacuum_corrections import weak_field_birefringence
from sinar.rays import ray_from_image, ray_from_surface

R, M_SOLAR, ENERGY_KEV = 6.0, 1.4, 1.0


def _dipole_vacuum(b_pole, **kw):
    """A dipole tilted 30 deg from z, with polar surface field b_pole [G]."""
    t = jnp.deg2rad(30.0)
    orient = jnp.array([[jnp.cos(t), 0.0, -jnp.sin(t)],
                        [0.0, 1.0, 0.0],
                        [jnp.sin(t), 0.0, jnp.cos(t)]])
    return put_qed_vacuum(R, M_SOLAR, ENERGY_KEV, jnp.array([b_pole * R**3]), orient,
                          **kw)


def _surface_ray():
    return ray_from_surface(jnp.array([1.0, 0.0, 0.0]),
                            jnp.array([0.0, 1.0, 0.0]), 0.6, R)


def _synthetic_coeffs(key, r_start, r_max, n_steps, omega_scale):
    """Random but well-formed transport coefficients on a u-grid."""
    k_om, k_z, k_ds = jax.random.split(key, 3)
    r = _u_grid(r_start, r_max, n_steps)
    # |Omega| ~ r^-6 with some jitter, as in the physical problem.
    omega = omega_scale * r ** -6 * jax.random.uniform(
        k_om, r.shape, minval=0.5, maxval=1.5)
    # A mode angle that wanders, including a few fast turns.
    zeta = jnp.cumsum(jax.random.normal(k_z, r.shape))
    mode = jnp.stack([jnp.cos(zeta), jnp.sin(zeta)], axis=1)
    dsdr = jax.random.uniform(k_ds, r.shape, minval=1.0, maxval=2.0)
    return r, omega, mode, dsdr


class TestMagnusScan:
    """Tests of the Magnus transport of the Stokes vector."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("omega_scale", [0.0, 1.0, 1e4, 1e12])
    def test_preserves_norm(self, seed, omega_scale):
        """|S| stays 1 for every band node, from weak to ~1e8 rad of phase."""
        r, omega, mode, dsdr = _synthetic_coeffs(
            jax.random.key(seed), r_start=10.0, r_max=800.0,
            n_steps=400, omega_scale=omega_scale)
        lam = jnp.linspace(0.9, 1.1, 7)

        S, _ = _magnus_scan(lam, r, omega, mode, dsdr)

        assert S.shape == (lam.shape[0], 3)
        assert jnp.all(jnp.isfinite(S))
        norm = jnp.linalg.norm(S, axis=1)
        assert jnp.allclose(norm, 1.0, atol=1e-10), \
            f"max | |S| - 1 | = {jnp.max(jnp.abs(norm - 1.0)):.3e}"


class TestTransportCoefficients:
    """Tests of |Omega| from the magnetised-vacuum birefringence."""

    def test_weak_field_option_reproduces_k_omega(self):
        """The B << B_Q limit is the former |Omega| = K_OMEGA M E_loc B_perp^2."""
        medium = _dipole_vacuum(1e13, birefringence=weak_field_birefringence)
        r = _u_grid(R, 800.0, 64)
        omega, _, _, bp2, _ = medium.coeffs(_surface_ray(), r)
        ref = K_OMEGA * M_SOLAR * ENERGY_KEV / jnp.sqrt(1.0 - 2.0 / r) * bp2
        assert jnp.allclose(omega, ref, rtol=1e-12, atol=0.0)

    def test_strong_field_correction_near_the_surface(self):
        """At B ~ 20 B_Q the full Delta_n is several times below the weak one,
        and the two agree once the dipole has fallen to B << B_Q."""
        ray = _surface_ray()
        r = _u_grid(R, 800.0, 64)
        full = _dipole_vacuum(1e15).coeffs(ray, r)[0]
        weak = _dipole_vacuum(
            1e15, birefringence=weak_field_birefringence).coeffs(ray, r)[0]
        ratio = full / weak
        assert jnp.all(jnp.isfinite(ratio))
        assert ratio[0] < 0.5
        assert jnp.isclose(ratio[-1], 1.0, rtol=1e-6, atol=0.0)

    def test_magnetar_image_is_well_formed(self):
        """Probe, its jvp, inner cut and band transport run at B_pole > B_Q."""
        medium = _dipole_vacuum(1e15, band_samples=8)
        rays = [ray_from_image(b, 0.4, R) for b in (0.5, 3.0)]
        out = solve_image(medium, rays, anchor_every=2, n_clean=100,
                          n_flagged=200)
        assert jnp.all(jnp.isfinite(out["stokes"]))
        assert jnp.all(jnp.linalg.norm(out["stokes"], axis=1) <= 1.0 + 1e-10)
        assert jnp.all(out["r_start"] >= R)


class TestOuterEdge:
    """Tests of the adaptive outer edge r_max of the integration."""

    @pytest.mark.parametrize("b_pole, energy_keV", [
        (1e12, 1.0), (1e14, 1.0), (1e15, 10.0)])
    def test_meets_tolerance(self, b_pole, energy_keV):
        tol = 1e-4
        medium = put_qed_vacuum(R, M_SOLAR, energy_keV,
                                jnp.array([b_pole * R**3]), jnp.eye(3),
                                r_max_tol=tol)
        assert medium.tail_bound <= tol * (1.0 + 1e-12)
        assert medium.r_max >= 10.0 * R
        # The probe keeps its density per decade as r_max grows.
        assert medium.r_probe[-1] == pytest.approx(float(medium.r_max))

    @pytest.mark.parametrize("c_dip", [0.0, 1e13 * R**3])
    def test_quadrupole_meets_tolerance(self, c_dip):
        comps = jnp.array([c_dip, 1e15 * R**4, 0.0, 0.0, 0.0, 0.0])
        r_max = adaptive_r_max(R, M_SOLAR, 1.05, comps, 1e-4)
        assert truncation_bound(M_SOLAR, 1.05, comps, r_max) <= 1e-4 * (
            1.0 + 1e-12)
        # A quadrupole falls off faster than a dipole of the same surface
        # strength, so it needs less range.
        r_dip = adaptive_r_max(R, M_SOLAR, 1.05,
                               jnp.array([1e15 * R**3]), 1e-4)
        assert 10.0 * R < r_max < r_dip

    def test_scales_as_polarization_limiting_radius(self):
        """r_max ~ (E B_dip^2)^(1/5) above the 10 R floor."""
        def r_max(b_pole, energy_keV):
            return adaptive_r_max(R, M_SOLAR, energy_keV,
                                  jnp.array([b_pole * R**3]), 1e-4)

        assert r_max(4e14, 1.0) / r_max(1e14, 1.0) == pytest.approx(
            4.0 ** 0.4, rel=1e-12)
        assert r_max(1e14, 32.0) / r_max(1e14, 1.0) == pytest.approx(
            2.0, rel=1e-12)
        assert r_max(1e6, 1.0) == pytest.approx(10.0 * R)

    def test_explicit_and_capped(self):
        comps = jnp.array([1e14 * R**3])
        explicit = put_qed_vacuum(R, M_SOLAR, ENERGY_KEV, comps, jnp.eye(3),
                                  r_max=800.0)
        assert explicit.r_max == 800.0
        assert explicit.r_probe.shape == (96,)
        capped = put_qed_vacuum(R, M_SOLAR, ENERGY_KEV, comps, jnp.eye(3),
                                r_light_cylinder=2000.0)
        assert capped.r_max == 2000.0
        # The cap is reported, not hidden.
        assert capped.tail_bound > 1e-4
        assert explicit.tail_bound > capped.tail_bound

    @pytest.mark.parametrize("tilt_deg", [0.0, 45.0])
    def test_converged_in_r_max(self, tilt_deg):
        """At 1e15 G and 10 keV the former r_max = 800 is short by ~0.1 deg;
        the adaptive one agrees with an 8 times larger edge.

        With the dipole along the line of sight B_perp -> 0 far out while
        the mode angle stays fixed, the case that once misplaced the cut."""
        comps = jnp.array([1e15 * R**3])
        ray = ray_from_image(3.0, 0.4, R)
        t = jnp.deg2rad(tilt_deg)
        orient = jnp.array([[jnp.cos(t), 0.0, -jnp.sin(t)], [0.0, 1.0, 0.0],
                            [jnp.sin(t), 0.0, jnp.cos(t)]])

        def solve(**kw):
            m = put_qed_vacuum(R, M_SOLAR, 10.0, comps, orient,
                               band_samples=16, **kw)
            r_start, _ = m.inner_cut(*m.probe(ray))
            return m, m.band(ray, r_start, 3200)[0]

        m, S = solve()
        _, S_far = solve(r_max=8.0 * float(m.r_max))
        assert jnp.linalg.norm(S - S_far) < 1e-3
        if tilt_deg:
            _, S_800 = solve(r_max=800.0)
            assert jnp.linalg.norm(S_800 - S_far) > 1e-3

    def test_traced_under_jit(self):
        """Energy traced under jit, as the example scripts do."""
        ray = ray_from_image(3.0, 0.4, R)

        @jax.jit
        def solve(energy_keV):
            m = put_qed_vacuum(R, M_SOLAR, energy_keV,
                               jnp.array([1e14 * R**3]), jnp.eye(3),
                               band_samples=8)
            r_start, _ = m.inner_cut(*m.probe(ray))
            return m.r_probe.shape[0], m.r_max, m.band(ray, r_start, 200)[0]

        n_probe, r_max, S = solve(1.0)
        assert n_probe == N_PROBE_TRACED
        assert r_max == pytest.approx(float(adaptive_r_max(
            R, M_SOLAR, 1.0 * 1.05, jnp.array([1e14 * R**3]), 1e-4)),
            rel=1e-2)
        assert jnp.all(jnp.isfinite(S))
