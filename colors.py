import jax
from jax import Array
import jax.numpy as jnp
from .geoms import uv_sphere

# This is a template for applying a a surface map
# TODO: Remove hardcoded functionality:
# - Bright color, dark color
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
def patch_surface(position: Array, orient: Array, semi_ap: float = 0.2):
    # It's a circle on the surface!
    uv = uv_sphere(position, orient)
    case = uv[0] - semi_ap
    bright = jnp.array([1.0, 0.3, 1.0])
    dark = jnp.array([0.1, 0.1, 0.5])
    return jax.lax.select(case > 0, bright, dark)