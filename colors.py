import jax
from jax import Array
import jax.numpy as jnp

def chequered_surface(uv: Array) -> Array:
    case = jnp.sum(jnp.floor(uv * 10), axis=-1) % 2
    bright = jnp.array([1.0, 0.3, 1.0])
    dark = jnp.array([0.1, 0.1, 0.5])
    return jax.lax.select(case == 0, bright, dark)