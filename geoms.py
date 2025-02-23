import jax
import jax.numpy as jnp
from jax import Array
from functools import partial

def put_sphere(location = jnp.array([0.,0.,0.]), radius = 1.0) -> partial:
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

def sdmin_scene(sdfs: dict, position: Array):
    return jnp.min(jnp.array([sdf(position) for sdf in sdfs]))

# For a smoother blending of objects, but it is slower
def sdsmin_scene(sdfs: list, position: Array):
    sdistances = jnp.array([sdf(position) for sdf in sdfs])
    return -jax.nn.logsumexp(-sdistances*16.0)/16