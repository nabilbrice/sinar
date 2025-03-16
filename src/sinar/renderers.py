import jax
from jax import Array
import jax.numpy as jnp
from functools import partial
from .rays import raymarch, gr_raymarch, normalize
from .entities.scenes import sdmin_scene, sdsmin_scene, sdargmin_scene

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

    # Find the closest entity for shading
    entity_idx = scene_argmin(position)
    uv = jnp.array([shape.uv(position) for shape in shapes])[entity_idx]
    sn = jnp.array([shape.sn(position) for shape in shapes])[entity_idx]
    mu = jnp.vecdot(-normalize(phase[3:6]), normalize(sn))

    color_surf = jnp.array([brdf(uv, mu) for brdf in brdfs])[entity_idx]
    color_back = jnp.zeros_like(color_surf)

    # Currently, the black hole is coded as entity 0
    dist = jax.lax.select(entity_idx >= 0,
                          scene_sdf(position),
                          1.0
    )
    # The returned array is appended a 1 to match RGBA,
    # assumed to have length 4.
    return jax.lax.select(dist < dtol,
              color_surf, color_back)

# Batch renderer for each pixel, since it will often be used
batch_render = jax.vmap(
    jax.jit(render, static_argnums=[0, 1, 3, 4]),
    in_axes=(None, None, 0))

def construct_pixlocs(xres = 400, yres = 400, size = 10.0):
    xs = jnp.linspace(-1., 1., xres)*size
    ys = jnp.linspace(1., -1., yres)*size # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    return jnp.stack([X.ravel(), Y.ravel()], axis=-1)