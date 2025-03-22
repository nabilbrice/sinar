import jax
from jax import Array
import jax.numpy as jnp
from .rays import raymarch, sdf_gr_raymarch, staged_gr_raymarch, normalize
from .entities.scenes import sdmin_scene, sdsmin_scene, sdargmin_scene

# The render function has two parts:
# (1) casting stage, which probes the geometry
# (2) shading stage, which probes the color maps
def render_by_surface(start_phase,
                      staged_shapes: tuple, brdfs: tuple,
                      dtol: float = 1e-4) -> Array:
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
    # Construct the scene sdf from the list of items
    def staged_sdf():
        return tuple(lambda p: sdmin_scene(shapes, p) for shapes in staged_shapes)
    
    scene_sdf = staged_sdf()[-1]
    shapes = staged_shapes[-1]
    
    phase = staged_gr_raymarch(start_phase,
                               staged_sdf(), end_times = jnp.array([18.0, 2.0]),
                               dtol = dtol / 2)[-1]
    position = phase[:3]
    # Actually the early termination condition already gives the is_hit...
    is_hit = scene_sdf(position) < dtol

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

    return jax.lax.select(is_hit,
              color_surf, color_back)

batch_render_by_surface = jax.vmap(
    jax.jit(render_by_surface, static_argnums=[1, 2, 3]),
    in_axes=(0, None, None))

def render_by_rayphase(pixloc: Array, focal_distance: float,
                       shapes: tuple, brdf: callable,
                       dtol: float = 1e-4) -> Array:
    """Renders a color for a pixel.

    The rendering is computed using a ray phase brdf,
    which takes the terminal ray phase as input.

    Parameters
    ----------
    pixloc : Array
        The pixel location in 2D.
    focal_distance : float
        The distance of the screen to the origin.
    shapes : tuple
        A container of the signed distance functions.
    brdf : callable
        The brdf which takes the terminal ray phase as input.
    """
    # Initialise a ray from the focus pointing to the screen.
    # Non-stereographic projection for black hole
    phase0 = init_rayphase(pixloc, focal_distance)

    # Construct the scene sdf from the list of items
    @jax.jit
    def scene_sdf(position):
        return sdmin_scene(shapes, position)
    
    phase = sdf_gr_raymarch(phase0, scene_sdf, dtol = dtol / 2)
    position = phase[:3]

    color_surf = brdf(position, phase[3:])
    color_back = jnp.zeros_like(color_surf)

    return jax.lax.select(scene_sdf(position) < dtol,
              color_surf, color_back)

batch_render_by_rayphase = jax.vmap(
    jax.jit(render_by_rayphase, static_argnums=[1, 2, 3, 4]),
    in_axes=(0, None, None, None))

def construct_pixlocs(xres = 400, yres = 400, size = 10.0) -> Array:
    """Constructs a grid of pixel locations.
    
    Parameters
    ----------
    xres : int
        The number of pixels in the x-direction.
    yres : int
        The number of pixels in the y-direction.
    size : float
        The half-size of the screen in the x and y directions.
    """
    xs = jnp.linspace(-1., 1., xres)*size
    ys = jnp.linspace(1., -1., yres)*size # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    return jnp.stack([X.ravel(), Y.ravel()], axis=-1)

def init_rayphase(pixloc, focal_distance) -> Array:
    """Initialises a ray phase from a pixel.
    """
    return jnp.array([*pixloc, focal_distance, 0.0, 0.0, -1.0])

def construct_screen_rays(xres = 400, yres = 400, size = 10.0, focal_distance = 10.0) -> Array:
    """Constructs the initial ray phases at a screen of pixels.

    Parameters
    ----------
    xres : int
        The number of pixels in the x-direction.
    yres : int
        The number of pixels in the y-direction.
    size : float
        The half-size of the screen in the x and y directions.
    focal_distance : float
        The distance of the screen from the coordinate origin.
    
    Returns
    -------
    rayphases : Array
        The ray phases at the screen.
    """
    pixlocs = construct_pixlocs(xres, yres, size)
    rayphases = jax.vmap(init_rayphase, in_axes =(0, None))(pixlocs, focal_distance)
    return rayphases