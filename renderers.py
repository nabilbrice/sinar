import jax
from jax import Array
import jax.numpy as jnp
from functools import partial
from .rays import raymarch, gr_raymarch, normalize
from .scenes import sdmin_scene, sdsmin_scene, sdargmin_scene

from .shapes import y_up_mat, id_mat

# The render function has two parts:
# (1) casting stage, which probes the geometry
# (2) shading stage, which probes the color maps
# These can be separated into two separate calls
#@partial(jax.jit, static_argnums=[0, 1, 3, 4])
def render(shapes: tuple, brdfs: tuple, pixloc: Array, focal_distance = 10.0, dtol: float = 1e-4) -> Array:
    """Renders a color for a pixel.

    Parameters
    ----------
    shapes : tuple
        A container of the signed distance functions.
    cms: tuple
        A container of color maps.
    """
    # Initialise a ray from the focus pointing to the screen.
    # Non-stereographic projection for black hole
    ro = jnp.array([*pixloc, focal_distance])
    rd = jnp.array([0.,0.,-1.])

    # Construct the scene sdf from the list of items
    scene_sdf = partial(sdmin_scene, shapes)
    scene_sdf = jax.jit(scene_sdf)

    scene_argmin = partial(sdargmin_scene, shapes)
    
    phase = gr_raymarch(ro, rd, scene_sdf)
    position = phase[:3]
    # The position is fed into the coloring stage right now
    # but this should be made into the helpful parameters:
    # (entity_ID, (u, v), angle)
    # all of this is handled by the casting aspect
    entity_idx = scene_argmin(position)

    # Up to this point, the render is able to return (ID, (u, v), mu)

    # So now dispatch the BRDF (ID, (u, v), mu)

    # Then the shading can occur, dispatching on the appropriate BRDF
    # The entity_idx is used again to switch on the right color
    #color_sf = partial(chequered_surface, y_up_mat, boxes = jnp.array([6,12]))
    #color_sf = jax.grad(scene_sdf)

    #color_surf = normalize(sample_dbb(position))
    color_surf = jnp.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ])
    color_surf = color_surf[entity_idx]
    color_back = jnp.array([0.0, 0.0, 0.0]) # Can be selected anything

    # These two selections should be merged
    # Currently, the black hole is coded as entity 0
    dist = jax.lax.select(entity_idx >= 0,
                          scene_sdf(position),
                          1.0
    )
    return jax.lax.select(dist < dtol,
              jnp.append(color_surf, 1.0),
              jnp.append(color_back, 1.0))

# Batch renderer for each pixel, since it will often be used
batch_render = jax.vmap(
    jax.jit(render, static_argnums=[0, 1, 3, 4]),
    in_axes=(None, None, 0))

def construct_pixlocs(xres = 400, yres = 400, size = 10.0):
    xs = jnp.linspace(-1., 1., xres)*size
    ys = jnp.linspace(1., -1., yres)*size # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    return jnp.stack([X.ravel(), Y.ravel()], axis=-1)