import jax
from jax import Array
import jax.numpy as jnp
from functools import partial
from .rays import gr_raymarch, normalize
from .scenes import sdmin_scene, sdsmin_scene, sdargmin_scene
from .colors import patch_surface, chequered_surface, sample_dbb

# The render is meant to return a color, so will need to give a surface map
# Currently, it constructs the colour inside
@partial(jax.jit, static_argnums=[0, 2, 3])
def render(sdfs: tuple, pixloc: Array, focal_distance = 10.0, dtol: float = 1e-3) -> Array:
    """Renders a color for a pixel.

    Parameters
    ----------
    sdfs : tuple
        A container of the signed distance functions.
    """
    # Initialise a ray from the focus pointing to the screen.
    # Non-stereographic projection for black hole
    ro = jnp.array([*pixloc, focal_distance])
    rd = jnp.array([0.,0.,-1.])

    # Construct the scene sdf from the list of items
    scene_sdf = partial(sdmin_scene, sdfs)
    scene_sdf = jax.jit(scene_sdf)

    scene_argmin = partial(sdargmin_scene, sdfs)
    
    # Construct the color
    # Note the color does not actually have to be a 3D array,
    # this is only for convenience of the image viewing.
    phase, _ = gr_raymarch(ro, rd, scene_sdf)
    position = phase[0][:3]

    entity_idx = scene_argmin(position)
    # TODO: Make color of surfaces composable like sdf
    #color_sf = partial(chequered_surface, rotation(jnp.pi*0.3, jnp.pi*0.05))
    #color_sf = partial(chequered_surface, y_up_mat, boxes = jnp.array([6,12]))
    color_sf = jax.grad(scene_sdf)
    #color_surf = normalize(color_sf(position))
    color_surf = normalize(sample_dbb(position))
    color_back = jnp.array([0.0, 0.0, 0.0]) # Can be selected anything

    # These two selections should be merged
    # Currently, the black hole is coded as entity 0
    dist = jax.lax.select(entity_idx != 0,
                          scene_sdf(position),
                          1.0
    )
    return jax.lax.select(dist < dtol,
              jnp.array([*color_surf, 1.0]),
              jnp.array([*color_back, 1.0]))

# Batch renderer for each pixel, since it will often be used
batch_render = jax.vmap(render, in_axes=(None, 0))