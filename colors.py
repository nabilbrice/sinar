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

def chequered_surface(uv: Array, boxes: Array, bright: Array, dark: Array) -> Array:
    case = jnp.sum(jnp.floor(uv * boxes), axis=-1) % 2
    return jax.lax.select(case == 0, bright, dark)

def set_chequered_surface(boxes: int | Array = jnp.array([8, 12]),
                          bright: Array = jnp.array([1.0, 0.3, 1.0]),
                          dark: Array = jnp.array([0.1, 0.1, 0.5])):
    return jax.jit(partial(chequered_surface, 
                           boxes = boxes, 
                           bright = bright, dark = dark))

def patch_surface(uv: Array, semi_ap: float, on: Array, off: Array):
    # It's a circle on the surface!
    case = uv[0] - semi_ap
    return jax.lax.select(case > 0, on, off)

def set_patch_surface(semi_ap: float = 0.2,
                      on: Array = jnp.array([1.0, 0.3, 1.0]),
                      off: Array = jnp.array([0.1, 0.1, 0.5])):
    return jax.jit(partial(patch_surface, semi_ap = semi_ap, 
                           on = on, off = off))

def blackbody(temperature: float, energy: float) -> float:
    """Computes a dimensionless radiance from a blackbody.
    """
    return energy**3 / jnp.expm1(energy / temperature)

# Spectrum is obtained by sampling from multiple energies at once
bb_spectrum = jax.vmap(blackbody, in_axes=(None, 0))

def sample_bb() -> float:
    return bb_spectrum(0.1, jnp.array([0.1, 1.0, 10.0]))

def sample_dbb(position) -> float:
    return bb_spectrum(1.0/jnp.linalg.vector_norm(position), jnp.array([0.5, 1.0, 2.0]))