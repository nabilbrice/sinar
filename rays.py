import jax
from jax import Array
import jax.numpy as jnp

def raymarch(origin: Array, direct: Array, scene_sdf,
             dtol = 1e-4) -> float:
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
    def should_step(state: tuple[Array, float]) -> bool:
        _, dt = state
        # A ray is considered escaped the scene if the step is >100.
        return jnp.logical_and(dt < 100., dt > dtol)

    def raystep(state: tuple[Array, float]) -> tuple[Array, float]:
        position, _ = state
        dt = scene_sdf(position)
        return position + dt * direct, dt

    return jax.lax.while_loop(should_step, raystep, (origin, 1.0))

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

# The render is meant to return a color, so will need to give a surface map
# Currently, it constructs the colour inside
def render(scene_sdf, pixloc: Array) -> Array:
    """Renders a color for a pixel.
    """
    # Initialise a ray from the focus pointing to the screen.
    ro = jnp.array([0.,0.,10.])
    rd = normalize(jnp.array([*pixloc, -10.]))

    normals_fn = jax.grad(scene_sdf)
    position, _ = raymarch(ro, rd, scene_sdf)
    normal = normalize(normals_fn(position))
    return jnp.array([*normal, 1.0])

# Batch renderer, since it will often be used
batch_render = jax.vmap(render, in_axes=(None, 0))