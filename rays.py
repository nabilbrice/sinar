import jax
from jax import Array
import jax.numpy as jnp

def raymarch(origin: Array, direct: Array, scene_sdf, 
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
    max_steps : int = 80
        The maximum number of steps to be taken
        before the ray marching is halted.
        This is actually the number of times the marching is looped.

    Returns
    -------
    t : float
        The parameter along the ray after max_steps.
    """
    # Body function for the while loop, a single step
    def raystep(_, position: Array):
        dt = scene_sdf(position)
        return position + dt * direct

    return jax.lax.fori_loop(0, max_steps, raystep, origin)

@jax.jit
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

def render(scene_sdf, pixloc: Array) -> Array:
    """Renders a color for a pixel.
    """
    # Initialise a ray from the focus pointing to the screen.
    ro = jnp.array([0.,0.,10.])
    rd = normalize(jnp.array([*pixloc, -10.]))

    # return the color, these are normalized to 0 .. 1
    zbuff = normalize(raymarch(ro, rd, scene_sdf))[2]
    return jnp.array([zbuff, 0., 0., 1.0])