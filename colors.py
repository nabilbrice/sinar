import jax
from jax import Array
import jax.numpy as jnp
from .shapes import uv_sphere
from functools import partial

# TODO: Color fields can be composed like the signed distance function.
# - Multiply the object color by the reciprocal of the signed distance,
#   higher powers mean sharper edges.
# - Colors will blend together naturally.

# This is a template for applying a a surface map
# TODO: Remove hardcoded functionality:
# - Bright color, dark color
@jax.jit
def chequered_surface(position: Array,
                      orient: Array,
                      boxes: int | Array = jnp.array([30, 30])) -> Array:
    uv = uv_sphere(position, orient)
    case = jnp.sum(jnp.floor(uv * boxes), axis=-1) % 2
    bright = jnp.array([1.0, 0.3, 1.0])
    dark = jnp.array([0.1, 0.1, 0.5])
    return jax.lax.select(case == 0, bright, dark)

# TODO: Add a surface patch with a semi-apeture and centre
# - It is a block color
@partial(jax.jit, static_argnums=2)
def patch_surface(position: Array, orient: Array, semi_ap: float = 0.2):
    # It's a circle on the surface!
    uv = uv_sphere(position, orient)
    case = uv[0] - semi_ap
    bright = jnp.array([1.0, 0.3, 1.0])
    dark = jnp.array([0.1, 0.1, 0.5])
    return jax.lax.select(case > 0, bright, dark)

def blackbody(temperature: float, energy: float) -> float:
    """Computes a dimensionless radiance from a blackbody.
    """
    return energy**3 / jnp.expm1(energy / temperature)

def sample_blackbody() -> float:
    return blackbody(0.1, jnp.array([0.1, 1.0, 10.0]))