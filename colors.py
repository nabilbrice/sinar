import jax
from jax import Array
import jax.numpy as jnp
from functools import partial
from typing import Callable

def blackbody(temperature: float, energy: float) -> float:
    """Computes the radiance from a blackbody.

    The temperature and energy should be in the same units
    so that (energy / temperature) is dimensionless, e.g. keV and keV.

    The radiance is given in arbitrary units.

    Parameters
    ----------
    temperature : float
        The characteristic temperature of the blackbody.
    energy : float | Array
        The energy at which to compute the radiance.
    
    Returns
    -------
    radiance : float
        The radiance, in arbitrary units.
    """
    return energy**3 / jnp.expm1(energy / temperature)

# Spectrum is obtained by sampling from multiple energies at once
bb_spectrum = jax.vmap(blackbody, in_axes=(None, 0))

def brdf_dpbb(inner_T: float, falloff: float, samples: Array, uv: Array, mu: Array):
    """Computes the color for a disc blackbody.
    
    Parameters
    ----------
    inner_T : float
        The temperature at the inner radius (u = 0).
    samples : Array
        The energy samples at which to get the radiance.

    Return
    ------
    spectrum : Array
        The spectrum (radiance) of a disc blackbody.
    """
    return bb_spectrum(inner_T / (1.0 + uv[0])**falloff, samples)

def set_brdf_dbb(inner_T: float = 1.5, falloff = 1.0, samples = jnp.array([0.1, 0.3, 1.0])):
    return jax.jit(partial(brdf_dpbb, inner_T, falloff, samples))

# BRDF needs to have (uv, mu) to be compatible with assumed calling convention in renderer
# The remaining parameters are configuration options
def is_chequered_region(uv: Array, boxes: int | Array = jnp.array([8, 12])):
    return jnp.sum(jnp.floor(uv * boxes), axis=-1) % 2 == 0

def is_cap_region(uv: Array, semi_ap: float = 0.2):
    return uv[0] < semi_ap

def is_patch_region(uv: Array, ulims: Array, vlims: Array):
    within_ulims = (uv[0] >= ulims[0]) & (uv[0] <= ulims[1])
    within_vlims = (uv[1] >= vlims[0]) & (uv[1] <= vlims[1])
    return within_ulims & within_vlims

# Each set brdf can take on different subsets as required by their parent
def set_brdf_region(region_fn: Callable, *args,
                   on: Callable = lambda mu: jnp.array([1.0, 0.3, 1.0]) * mu,
                   off: Array = bb_spectrum(2.0, jnp.array([0.1, 0.3, 0.9]))
                   ):
    """Set the brdf of the associated entity (by index) to be as a cap.

    The returned partial function of the color satisfies the BRDF interface:
    brdf(uv, mu), where uv is an array and mu is a float.
    
    Parameters
    ----------
    region_fn : Callable
        A function delineating on and off regions, by returning a conditional.
    *args
        Additional arguments for modifying the region_fn.
    on : Callable
        The color on the cap, which depends on the incident angle.
    off : Array
        The color off the cap, which does not depend on the incident angle.
    Returns
    -------
    brdf : Callable
        A function that satisfies the brdf interface brdf(uv, mu) -> Array.
    """
    def brdf(uv: Array, mu: float):
        return jax.lax.select(region_fn(uv, *args),
                            on(uv, mu),
                            off,
        )
    return jax.jit(brdf)