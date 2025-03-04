import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jax import Array
from functools import partial

# TODO: Object geometry can be changed to provide
# a transformation of the position.

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

@partial(jax.jit, static_argnums=[0,1], inline=True)
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

def put_sphere(location = jnp.array([0.,0.,0.]),
               radius = 1.0,
               color = jnp.array([1.0, 0.0, 0.0])) -> partial:
    return partial(sd_sphere, location, radius)

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
def sd_cylinder(radius: float, height: float, orient: Array, position: Array, tol = -1e-6) -> Array:
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
    position = jnp.matmul(orient, position)
    dists = jnp.array([
        jnp.linalg.vector_norm(position[:2]) - radius,
        jnp.abs(position[2]) - height
    ])
    return jnp.minimum(jnp.max(dists), 0.0) + jnp.linalg.vector_norm(jnp.maximum(dists, tol))

def put_cylinder(radius: float = 1.0, height: float = 0.5, orient: Array = id_mat, tol = -1e-6) -> partial:
    return partial(sd_cylinder, radius, height, orient, tol=tol)

# subtraction is: max(-d1, d2)

def sd_disc(inner: float, outer: float, height: float, orient: Array, position: Array, tol=-1e-6) -> Array:
    sd_inner = sd_cylinder(inner, height, orient, position, tol=-tol)
    sd_outer = sd_cylinder(outer, height, orient, position, tol=tol)
    return jnp.maximum(-sd_inner, sd_outer)

def put_thindisc(inner: float = 3.0, outer: float = 5.0, height: float = 0.25, 
                 orient: Array = id_mat, tol = -1e-6) -> partial:
    return partial(sd_disc, inner, outer, height, orient, tol=tol)