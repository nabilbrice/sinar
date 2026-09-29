import jax
import pytest
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sinar.media.qed_polarization import _magnus_scan, _u_grid


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
