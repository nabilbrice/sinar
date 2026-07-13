import jax
from jax import Array
import jax.numpy as jnp
from .rays import (raymarch, gr_raymarch, terminate_by_position, aniso_sdf_gr_raymarch, normalize, 
                    obs_ray_invariants, adaptive_stepper)
from .entities.scenes import sdmin_scene, sdsmin_scene, sdargmin_scene
from functools import partial

# Standard render function, whith a color given by the surface position:
# The render function has two parts:
# (1) casting stage, which probes the geometry
# (2) shading stage, which probes the color maps
def render_by_surface(start_phase,
                      shapes: tuple, brdfs: tuple,
                      aux_phase_dyn = None,
                      dtol: float = 1e-4, timespan = 24.0) -> Array:
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
    def scene_sdf(position):
        return sdmin_scene(shapes, position)

    terminal_event = terminate_by_position(lambda position: scene_sdf(position) < dtol / 2)
    phase = gr_raymarch(start_phase, terminal_event, end_time=timespan, aux_fn=aux_phase_dyn,
                        stepsize_controller=adaptive_stepper(scene_sdf))
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

    return phase, jax.lax.select(is_hit, color_surf, color_back)

@partial(jax.jit, static_argnames=['shapes', 'brdfs', 'aux_phase_dyn'])
def batch_render_by_surface(start_phase, shapes, brdfs, 
                            aux_phase_dyn=None, dtol=1e-4, timespan=24.0):
    batch_render = jax.vmap(render_by_surface, 
        in_axes=(0, None, None, None, None, None)
        )(start_phase, shapes, brdfs, aux_phase_dyn, dtol, timespan)
    return batch_render

def staged_batch_render(phases, staged_shapes, brdfs, aux_phase_dyn=None, dtol = 1e-4, timespans=24.0):
    """Multi-stage rendering.
    """
    n_stages = len(staged_shapes)
    timespans = jnp.broadcast_to(jnp.array(timespans), n_stages)
    staged_colors = []
    for i, shapes in enumerate(staged_shapes):
        phases, colors = batch_render_by_surface(phases, shapes, brdfs, aux_phase_dyn, dtol, timespans[i])
        staged_colors.append(colors)
    return phases, jnp.array(staged_colors)

# Render function for when the color is obtained from the rayphase
def render_by_rayphase(start_phase,
                       shapes: tuple, brdf: callable,
                       aux_phase_dyn = None,
                       dtol: float = 1e-4, timespan = 24.0) -> Array:
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
    # Construct the scene sdf from the list of items
    @jax.jit
    def scene_sdf(position):
        return sdmin_scene(shapes, position)
    
    terminal_event = terminate_by_position(lambda position: scene_sdf(position) < dtol / 2)
    phase = gr_raymarch(start_phase, terminal_event, end_time = timespan, aux_fn = aux_phase_dyn,
                        stepsize_controller=adaptive_stepper(scene_sdf))
    position = phase[:3]

    color_surf = brdf(position, phase[3:6])
    color_back = jnp.zeros_like(color_surf)

    return phase, jax.lax.select(scene_sdf(position) < dtol, color_surf, color_back)

@partial(jax.jit, static_argnames=['shapes', 'brdfs', 'aux_phase_dyn'])
def batch_render_by_rayphase(start_phase, shapes, brdfs, 
                            aux_phase_dyn=None, dtol=1e-4, timespan=24.0):
    batch_render = jax.vmap(render_by_rayphase, 
        in_axes=(0, None, None, None, None, None)
        )(start_phase, shapes, brdfs, aux_phase_dyn, dtol, timespan)
    return batch_render

@partial(jax.jit, static_argnames=['staged_shapes', 'brdfs', 'aux_phase_dyn'])
def staged_batch_render_by_rayphase(phases, staged_shapes, brdfs, aux_phase_dyn=None, dtol = 1e-4, timespans=24.0):
    """Multi-stage rendering.
    """
    n_stages = len(staged_shapes)
    timespans = jnp.broadcast_to(jnp.array(timespans), n_stages)

    probe_brdf = brdfs if callable(brdfs) else brdfs[0]
    probe_color = jnp.array(probe_brdf(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])))

    full_color_shape = (n_stages,) + phases.shape[:-1] + probe_color.shape
    staged_colors = jnp.zeros(full_color_shape, dtype=probe_color.dtype)
    for i, shapes in enumerate(staged_shapes):
        phases, colors = batch_render_by_rayphase(phases, shapes, brdfs, aux_phase_dyn, dtol, timespans[i])
        staged_colors = staged_colors.at[i].set(colors)
    return phases, staged_colors

# Render anisotropic by rayphase
def render_aniso_by_rayphase(start_phase,
                             shapes: tuple, brdf: callable,
                             aux_phase_dyn = None,
                             dtol: float = 1e-4, timespan = 24.0) -> Array:
    """Renders using anisotropic (direction-dependent) shapes."""
    @jax.jit
    def scene_sdf(phase):
        return sdmin_scene(shapes, phase)
    
    phase = aniso_sdf_gr_raymarch(start_phase, scene_sdf, end_time=timespan, 
                                   aux_fn=aux_phase_dyn, dtol=dtol/2)
    position = phase[:3]
    direction = phase[3:6]

    color_surf = brdf(position, direction)
    color_back = jnp.zeros_like(color_surf)

    return phase, jax.lax.select(scene_sdf(phase) < dtol, color_surf, color_back)

@partial(jax.jit, static_argnames=['shapes', 'brdf', 'aux_phase_dyn'])
def batch_render_aniso_by_rayphase(start_phase, shapes, brdf, 
                                    aux_phase_dyn=None, dtol=1e-4, timespan=24.0):
    batch_render = jax.vmap(render_aniso_by_rayphase, 
        in_axes=(0, None, None, None, None, None)
        )(start_phase, shapes, brdf, aux_phase_dyn, dtol, timespan)
    return batch_render

@partial(jax.jit, static_argnames=['staged_shapes', 'brdf', 'aux_phase_dyn'])
def staged_batch_render_aniso_by_rayphase(phases, staged_shapes, brdf, 
                                           aux_phase_dyn=None, dtol=1e-4, timespans=24.0):
    """Multi-stage rendering with anisotropic shapes."""
    n_stages = len(staged_shapes)
    timespans = jnp.broadcast_to(jnp.array(timespans), n_stages)

    probe_color = jnp.array(brdf(jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])))

    full_color_shape = (n_stages,) + phases.shape[:-1] + probe_color.shape
    staged_colors = jnp.zeros(full_color_shape, dtype=probe_color.dtype)
    
    for i, shapes in enumerate(staged_shapes):
        phases, colors = batch_render_aniso_by_rayphase(phases, shapes, brdf, 
                                                         aux_phase_dyn, dtol, timespans[i])
        staged_colors = staged_colors.at[i].set(colors)
    return phases, staged_colors

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

def init_rayphase(pixloc, focal_distance, n_aux=0) -> Array:
    """Initialises a ray phase from a pixel.
    """
    return jnp.array([*pixloc, focal_distance, 0.0, 0.0, -1.0, *jnp.zeros(n_aux)])

def construct_screen_rays(xres = 400, yres = 400, size = 10.0, focal_distance = 10.0, n_aux=0) -> Array:
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
    rayphases = jax.vmap(init_rayphase, in_axes =(0, None, None))(pixlocs, focal_distance, n_aux)
    return rayphases

def _obs_sky_frame(los: Array):
    """Builds an orthonormal (e1, e2) frame for the sky plane perpendicular to los."""
    z_hat = jnp.array([0.0, 0.0, 1.0])
    y_hat = jnp.array([0.0, 1.0, 0.0])
    ref = jnp.where(jnp.abs(jnp.dot(los, z_hat)) < 0.99, z_hat, y_hat)
    e1 = normalize(jnp.cross(ref, los))
    e2 = jnp.cross(los, e1)
    return e1, e2

def construct_obs_rays(xres=400, yres=400, size=10.0, los=jnp.array([0.0, 0.0, -1.0])):
    """Constructs ray invariants for an observer at infinity.

    Parameters
    ----------
    xres, yres : int
        Pixel resolution.
    size : float
        Half-size of the sky plane.
    los : Array (3,)
        Line-of-sight unit vector (observer toward origin).

    Returns
    -------
    b2s : Array (N,)
        Impact parameter squared per ray.
    los_perps : Array (N, 3)
        Per-ray perpendicular direction in the sky plane.
    los : Array (3,)
        The line-of-sight vector (passed through for convenience).
    """
    pixlocs = construct_pixlocs(xres, yres, size)
    e1, e2 = _obs_sky_frame(los)
    b2s, los_perps = obs_ray_invariants(pixlocs, e1, e2)
    return b2s, los_perps, los