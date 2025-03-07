import jax
from jax import Array
import jax.numpy as jnp
from .shapes import uv_sphere
from functools import partial
from typing import NamedTuple, Callable

# TODO: Color fields can be composed like the signed distance function.
# - Multiply the object color by the reciprocal of the signed distance,
#   higher powers mean sharper edges.
# - Colors will blend together naturally.
class ColorMap(NamedTuple):
    brdf: Callable
# set_patch_surface() ->
# cm.brdf(uv, mu) call
# in construction, the brdf is
# return partial(patch_surface, semi_ap)
# while the call dispatches brdf

def chequered_surface(uv: Array, boxes: Array, bright: Array, dark: Array) -> Array:
    case = jnp.sum(jnp.floor(uv * boxes), axis=-1) % 2
    return jax.lax.select(case == 0, bright, dark)

def set_chequered_surface(boxes: int | Array = jnp.array([8, 12]),
                          bright: Array = jnp.array([1.0, 0.3, 1.0]),
                          dark: Array = jnp.array([0.1, 0.1, 0.5])):
    return partial(chequered_surface, 
                           boxes = boxes, 
                           bright = bright, dark = dark)

def patch_surface(semi_ap: float, on: Array, off: Array, uv):
    # It's a circle on the surface!
    case = uv[0] - semi_ap
    return jax.lax.select(case < 0, on, off)

def set_patch_surface(semi_ap: float = 0.2,
                      on: Array = jnp.array([1.0, 0.3, 1.0]),
                      off: Array = jnp.array([0.1, 0.1, 0.5])):
    return partial(patch_surface, semi_ap, on, off)

def blackbody(temperature: float, energy: float) -> float:
    """Computes a dimensionless radiance from a blackbody.
    """
    return energy**3 / jnp.expm1(energy / temperature)

# Spectrum is obtained by sampling from multiple energies at once
bb_spectrum = jax.vmap(blackbody, in_axes=(None, 0))

def sample_bb() -> float:
    return bb_spectrum(0.3, jnp.array([0.1, 0.3, 1.0]))

def sample_dbb(position) -> float:
    return bb_spectrum(1.0/jnp.linalg.vector_norm(position), jnp.array([0.5, 1.0, 2.0]))

def brdf_patch(semi_ap: float, on: Callable, off: Array, uv: Array, mu: float):
    case = uv[0] - semi_ap
    return jax.lax.select(case < 0, on(mu), off)

def set_brdf_patch(semi_ap: float = 0.2,
                           on: Callable = lambda mu: jnp.array([1.0, 0.3, 1.0]) * mu,
                           off: Array = sample_bb()):
    return partial(brdf_patch, semi_ap, on, off)

def brdf_chequered(boxes: Array, bright: Array, dark: Array, uv: Array, mu) -> Array:
    case = jnp.sum(jnp.floor(uv * boxes), axis=-1) % 2
    return jax.lax.select(case == 0, bright(mu), dark(mu))

def set_brdf_chequered(boxes: int | Array = jnp.array([8, 12]),
                          bright = lambda mu: jnp.array([1.0, 0.3, 1.0]),
                          dark = lambda mu: jnp.array([0.1, 0.1, 0.5])):
    return partial(brdf_chequered, boxes, bright, dark)