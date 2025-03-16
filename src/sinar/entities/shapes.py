import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from jax import Array
from functools import partial
from typing import NamedTuple, Callable
from ..rays import normalize

# The Shape tuple can hold all the geometry information
# and be calculated JIT as needed
class Shape(NamedTuple):
    sdf: Callable
    uv: Callable
    sn: Callable = None

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
def rotation(phi: float = 0.0, theta: float = jnp.pi * 0.5) -> Array:
    """Rotates a shape around the y-axis and then x-axis.

    phi and theta are specified in radians.
    
    Parameters
    ----------
    phi : float
        Euler angle of rotation about the y-axis.
    theta : float
        Euler angle of rotation about the x-axis.

    Return
    ------
    matrix : Array
        A 2D array which represents the combined rotation.
    """
    # Sign reversal to get the appropriate rotation direction
    rot_phi = Rotation.from_euler('y', -phi)
    rot_the = Rotation.from_euler('x', theta)
    rot = rot_the * rot_phi
    return rot.as_matrix()

#########
# Spheres
#########
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

def uv_sphere(location: Array, radius: float, orient: Array, position: Array) -> Array:
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
    local_position = normalize(position - location)
    oriented = jnp.linalg.matmul(orient, local_position)

    # u = theta / pi, running from 0 to 1
    u = jnp.acos(oriented[2]) / jnp.pi
    
    # v = phi / 2 pi + 0.5, running from 0 to 1
    v = jnp.atan2(oriented[1], oriented[0]) * 0.5 / jnp.pi + 0.5
    return jnp.array([u,v])

def put_sphere(location = jnp.array([0.,0.,0.]),
               radius = 1.0,
               orient = y_up_mat) -> partial:
    return Shape(
        sdf=jax.jit(partial(sd_sphere, location, radius), inline=True),
        uv=jax.jit(partial(uv_sphere, location, radius, orient), inline=True),
        sn=jax.jit(jax.grad(partial(sd_sphere, location, radius)), inline=True),
    )

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

def uv_cylinder(position: Array, orient = y_up_mat):
    pass

def put_cylinder(radius: float = 1.0, height: float = 0.5, orient: Array = id_mat, tol = -1e-6) -> partial:
    return Shape(
        sdf=partial(sd_cylinder, radius, height, orient, tol=tol),
        uv=jax.jit(uv_cylinder),
        sn=jax.jit(jax.grad(partial(sd_cylinder, radius, height, orient, tol)))
    )

def sd_disc(inner: float, outer: float, height: float, orient: Array, position: Array) -> Array:
    oriented = jnp.linalg.matmul(orient, position)
    # Compute the radial distance in the plane:
    r = jnp.linalg.vector_norm(oriented[:2])
    # Compute the distance to the outer and inner boundaries:
    d_plane = jnp.maximum(r - outer, inner - r)
    # Compute the distance to the top / bottom surface:
    d_height = jnp.abs(oriented[2]) - height

    d = jnp.maximum(d_plane, d_height)

    return jnp.minimum(d, 0.0) + jnp.linalg.vector_norm(jnp.maximum(jnp.array([d_plane, d_height]), 0.0))


def uv_disc(inner, outer, height, orient: Array, position: Array):
    oriented = jnp.linalg.matmul(orient, position)
    u = (jnp.linalg.vector_norm(oriented[:2]) - inner) / (outer - inner)
    v = jnp.atan2(oriented[1], oriented[0]) * 0.5 / jnp.pi + 0.5
    return jnp.array([u, v])

def put_thindisc(inner: float = 3.0, outer: float = 5.0, height: float = 0.25, 
                 orient: Array = y_up_mat) -> partial:
    return Shape(
        sdf=jax.jit(partial(sd_disc, inner, outer, height, orient), inline=True),
        uv=jax.jit(partial(uv_disc, inner, outer, height, orient), inline=True),
        sn=jax.jit(jax.grad(partial(sd_disc, inner, outer, height, orient)), inline=True),
    )
