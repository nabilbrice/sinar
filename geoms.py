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

# An identity matrix (z-axis is north)
id_mat = jnp.array(
    [[1.,0.,0.],
     [0.,1.,0.],
     [0.,0.,1.]]
    )
# The matrix corresponding to y-axis being the north
y_up_mat = jnp.array(
    [[1.,0.,0.],
     [0.,0.,1.],
     [0.,1.,0.]]
    )

@partial(jax.jit, static_argnums=1)
def uv_sphere(position: Array, orient = y_up_mat) -> Array:
    """Computes the local surface coordinates at a sphere.

    Parameters
    ----------
    position : Array [3,]
        The position to be tested at the surface.
    orient : Array [3,3]
        The orientation matrix of the sphere.
    Returns
    -------
    uv : Array [2]
        The local surface coordinates.
    """
    oriented = jnp.linalg.matmul(orient, position)

    # u = theta / pi, running from 0 to 1
    u = jnp.acos(oriented[2]) / jnp.pi
    
    # v = phi / 2 pi + 0.5, running from 0 to 1
    v = jnp.atan2(oriented[1], oriented[0]) * 0.5 / jnp.pi + 0.5
    return jnp.array([u,v])

def sdmin_scene(sdfs: list, position: Array):
    return jnp.min(jnp.array([sdf(position) for sdf in sdfs]))

# For a smoother blending of objects, but it is slower
def sdsmin_scene(sdfs: list, position: Array):
    sdistances = jnp.array([sdf(position) for sdf in sdfs])
    return -jax.nn.logsumexp(-sdistances*16.0)/16