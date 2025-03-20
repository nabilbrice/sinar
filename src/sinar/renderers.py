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
def render_by_surface(pixloc: Array, focal_distance, shapes: tuple, brdfs: tuple, dtol: float = 1e-4) -> Array:
    """Renders a color for a pixel.

    The rendering is computed using a surface brdf,
    which takes surface coordinates and incident angle as input.
    These are computed in the render_by_surface from the terminal ray phase.

    Parameters
    ----------
    pixloc : Array
        The pixel location in 2D.
    focal_distance : float
        The distance of the screen to the origin.
    shapes : tuple
        A container of the signed distance functions.
    brdfs : tuple
        A container of brdfs which are matched with the shapes in index.
    """
    # Initialise a ray from the focus pointing to the screen.
    # Non-stereographic projection for black hole
    ro = jnp.array([*pixloc, focal_distance])
    rd = jnp.array([0.,0.,-1.])

    # Construct the scene sdf from the list of items
    @jax.jit
    def scene_sdf(position):
        return sdmin_scene(shapes, position)
    
    phase = gr_raymarch(ro, rd, scene_sdf)
    position = phase[:3]

    @jax.jit
    def scene_argmin(position):
        return sdargmin_scene(shapes, position)

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
batch_render_by_surface = jax.vmap(
    jax.jit(render_by_surface, static_argnums=[1, 2, 3, 4]),
    in_axes=(0, None, None, None))

def render_by_rayphase(shapes: tuple, rot: callable, pixloc: Array, focal_distance = 20.0, dtol: float = 1e-4) -> Array:
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
    @jax.jit
    def scene_sdf(position):
        return sdmin_scene(shapes, position)
    
    phase = gr_raymarch(ro, rd, scene_sdf)
    position = phase[:3]

    # Find the closest entity for shading

    color_surf = rot(position, phase[3:])
    color_back = jnp.complex64(0.0)

    dist = scene_sdf(position)
    # The returned array is appended a 1 to match RGBA,
    # assumed to have length 4.
    return jax.lax.select(dist < dtol,
              color_surf, color_back)

batch_render_by_rayphase = jax.vmap(
    jax.jit(render_by_rayphase, static_argnums=[0, 1, 3, 4]),
    in_axes=(None, None, 0))

def construct_pixlocs(xres = 400, yres = 400, size = 10.0):
    xs = jnp.linspace(-1., 1., xres)*size
    ys = jnp.linspace(1., -1., yres)*size # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    return jnp.stack([X.ravel(), Y.ravel()], axis=-1)