from typing import Callable
from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
from .geoms import sdsmin_scene, y_up_mat, rotation
from .colors import patch_surface, chequered_surface

@partial(jax.jit, static_argnums=[2, 3])
def raymarch(origin: Array, direct: Array, scene_sdf: Callable,
             max_steps=80) -> float:
    """Marches a ray from origin to termination along direct.
    
    The ray marching is done in Euclidean space with a fixed direction
    for each ray. Each ray is a tuple[Array[3], Array[3]].

    Parameters
    ----------
    origin : Array [3,]
        The origin coordinates of the ray at the observer.
    direct : Array [3,]
        The direction vector of the ray. Its norm should be 1
        to ensure correct computation of the signed-distance-function.
    scene_sdf : callable
        The signed-distance-function of the 'scene',
        which defines the scene geometry.
    dtol : float = 1e-4
        The distance tolerance for when a ray is considered to be
        close enough to a surface.
    Returns
    -------
    t : float
        The parameter along the ray after max_steps.
    """
    # Body function for the loop, a single step
    def raystep(_, position: Array):
        dt = scene_sdf(position)
        return position + dt * direct

    return jax.lax.fori_loop(0, max_steps, raystep, origin)

@partial(jax.jit, static_argnums=1)
def normalize(v: Array, axis: int = -1) -> Array:
    """Compute a normalized vector from the given input vector.

    By default, the outermost axis is chosen
    so v.shape must be (3,)

    Parameters
    ----------
    v : Array[3,]
    Returns
    -------
    n : Array[3,]
        The normalized vector.
    """
    return v/jnp.linalg.vector_norm(v, axis=axis, keepdims=True)

# The render is meant to return a color, so will need to give a surface map
# Currently, it constructs the colour inside
@partial(jax.jit, static_argnums=[0, 2])
def render(sdfs: tuple, pixloc: Array, dtol: float = 1e-2) -> Array:
    """Renders a color for a pixel.

    Parameters
    ----------
    sdfs : tuple
        A container of the signed distance functions.
    """
    # Initialise a ray from the focus pointing to the screen.
    ro = jnp.array([0.,0.,10.])
    rd = normalize(jnp.array([*pixloc, -10.]))

    # Construct the scene sdf from the list of items
    scene_sdf = partial(sdsmin_scene, sdfs)
    scene_sdf = jax.jit(scene_sdf)
    
    # Construct the color
    # Note the color does not actually have to be a 3D array,
    # this is only for convenience of the image viewing.
    position = raymarch(ro, rd, scene_sdf)
    # TODO: Make color of surfaces composable like sdf
    color_sf = partial(patch_surface, rotation(jnp.pi*0.3, jnp.pi*0.05))
    color_sf = jax.grad(scene_sdf)
    color_surf = normalize(color_sf(position))
    color_back = jnp.array([0.3, 0.5, 0.7]) # Can be selected anything

    dist = scene_sdf(position)
    return jax.lax.select(dist < dtol,
              jnp.array([*color_surf, 1.0]),
              jnp.array([*color_back, 1.0]))

# Batch renderer for each pixel, since it will often be used
batch_render = jax.vmap(render, in_axes=(None, 0))