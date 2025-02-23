import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jax import Array
from functools import partial

def sdmin_scene(sdfs: list, position: Array):
    return jnp.min(jnp.array([sdf(position) for sdf in sdfs]))

# For a smoother blending of objects, but it is slower
def sdsmin_scene(sdfs: list, position: Array):
    sdistances = jnp.array([sdf(position) for sdf in sdfs])
    return -jax.nn.logsumexp(-sdistances*16.0)/16

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

@partial(jax.jit, static_argnums=[0,1])
def rotation(theta: float = jnp.pi * 0.5, phi: float = 0.0) -> Array:
    # Sign reversal to get the appropriate rotation direction
    rot_phi = Rotation.from_euler('y', -phi)
    rot_the = Rotation.from_euler('x', theta)
    rot = rot_phi * rot_the
    return rot.as_matrix()

#########
# Spheres
#########
@partial(jax.jit, static_argnums=1, inline=True)
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

def put_sphere(location = jnp.array([0.,0.,0.]), radius = 1.0) -> partial:
    return partial(sd_sphere, location, radius)

@jax.jit
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

######
# Disc
######
@jax.jit
def sd_disc(radius: float, height: float, position: Array) -> Array:
    """Computes the signed distance of a cylinder.
    
    Parameters
    ----------
    radius : float
        The radius of the disc face.
    height : float
        The disc (half) height.
    Returns
    -------
    distance : float
        The signed distance for a given position.
    """
    proj_radius_dist = jnp.sqrt(position[0]**2 + position[1]**2) - radius
    proj_height_dist = jnp.abs(position[2]) - height
    return jnp.sqrt(jnp.maximum(proj_radius_dist, 0.0)**2 + jnp.maximum(proj_height_dist, 0.0)
                    + jnp.minimum(jnp.maximum(proj_radius_dist, proj_height_dist), 0.0))

def put_disc(radius: float = 1.0, height: float = 0.5) -> partial:
    return partial(sd_disc, radius, height)