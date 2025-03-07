import jax
from jax import Array
import jax.numpy as jnp
from functools import partial
from typing import Callable
from .loaders import load_checked_interpolators

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

def brdf_dbb(inner_T: float, uv: Array, mu: Array):
    """Computes the color for a disc blackbody.
    
    Parameters
    ----------
    inner_T : float
        The temperature at the inner radius (u = 0).
    outer_T : float
        The temperature at the outer radius (u = 1).
    """
    return bb_spectrum(inner_T / (1.0 + uv[0])**4, jnp.array([0.3, 0.9, 2.1]))

def set_brdf_dbb(inner_T: float):
    return partial(brdf_dbb, inner_T)

# BRDF needs to have (uv, mu) to be compatible with assumed calling convention in renderer
# The remaining parameters are configuration options
def brdf_patch(semi_ap: float, on: Callable, off: Array, uv: Array, mu: float):
    case = uv[0] - semi_ap
    return jax.lax.select(case < 0, on(mu), off)

# Each set brdf can take on different subsets as required by their parent
def set_brdf_patch(semi_ap: float = 0.2,
                   on: Callable = lambda mu: jnp.array([1.0, 0.3, 1.0]) * mu,
                   off: Array = bb_spectrum(2.0, jnp.array([0.1, 0.3, 0.9]))):
    """Set the brdf of the associated entity (by index) to be as a patch.

    The returned partial function of the color satisfies the BRDF interface:
    brdf(uv, mu), where uv is an array and mu is a float.
    
    Parameters
    ----------
    semi_ap : float
        The semi-aperture angle (in radians) of the patch.
    on : Callable
        The color on the patch, which depends on the incident angle.
    off : Array
        The color off the patch, which does not depend on the incident angle.
    Returns
    -------
    color : Callable
        A partial function of the color.
    """
    return partial(brdf_patch, semi_ap, on, off)

def brdf_chequered(boxes: Array, bright: Array, dark: Array, uv: Array, mu) -> Array:
    case = jnp.sum(jnp.floor(uv * boxes), axis=-1) % 2
    return jax.lax.select(case == 0, bright(mu), dark(mu))

def set_brdf_chequered(boxes: int | Array = jnp.array([8, 12]),
                          bright = lambda mu: jnp.array([1.0, 0.3, 1.0]),
                          dark = lambda mu: jnp.array([0.1, 0.1, 0.5])):
    return partial(brdf_chequered, boxes, bright, dark)