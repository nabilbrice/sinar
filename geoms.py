import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

def new_sphere(location = jnp.array([0.,0.,0.]), radius = 1.0) -> partial:
    return partial(sd_sphere, location, radius)

@jax.jit
def sd_sphere(location: Array, radius : float, position: Array) -> float:
    """Computes the signed distance of a sphere.
    
    Parameters
    ----------
    location : Array [3,a]
    radius : float[a]
        The radius of the sphere.
    position : Array [3,b]
        The position to compute the distance from.
    Returns
    -------
    distance : float[b]
        The signed distance for a given position.
    """
    return jnp.linalg.vector_norm(position - location, axis=-1) - radius