import jax
from jax import Array
import jax.numpy as jnp

# TODO: Remove hardcoded functionality:
# - Bright color, dark color
def chequered_surface(uv: Array,
                      boxes: int | Array = jnp.array([30, 30])) -> Array:
    case = jnp.sum(jnp.floor(uv * boxes), axis=-1) % 2
    bright = jnp.array([1.0, 0.3, 1.0])
    dark = jnp.array([0.1, 0.1, 0.5])
    return jax.lax.select(case == 0, bright, dark)