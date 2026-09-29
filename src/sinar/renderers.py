import jax
from jax import Array
import jax.numpy as jnp
from .rays import (raymarch, gr_raymarch, terminate_by_position, terminate_by_phase, normalize,
                    obs_ray_invariants, adaptive_stepper, coordframe_to_staticframe,
                    approx_coordphase_at_r)
from .entities.scenes import Scene, sdmin_scene, sdsmin_scene, sdargmin_scene
from functools import partial

# Frames
# ------
# gr_raymarch integrates a coordinate-frame phase: position, then the
# backward-traced tangent of the ray in the Euclidean embedding of
# Schwarzschild coordinates.  Everything a scene evaluates -- SDFs, surface
# colours, BRDFs -- is physics, and physics needs the static-frame phase:
# position, then the photon's propagation direction as a static observer
# measures it.  This module is the boundary between the two:
#
# * scenes, shapes and BRDFs only ever receive static-frame phases, so their
#   directions point along the photon (away from a surface it left), and an
#   emission cosine is mu = k . n_hat, with no minus sign;
# * phases handed back for further marching (e.g. staged rendering) stay in
#   the coordinate frame.


def _static(fn):
    """Wrap a phase function so it receives the static-frame phase."""
    return lambda phase: fn(coordframe_to_staticframe(phase))

# Standard render function, whith a color given by the surface position:
# The render function has two parts:
# (1) casting stage, which probes the geometry
# (2) shading stage, which probes the color maps
def render_by_surface(start_phase,
                      scene: Scene,
                      aux_phase_dyn = None,
                      dtol: float = 1e-4, timespan = 24.0) -> Array:
    """Renders a color for a pixel.

    The rendering is computed using a surface brdf,
    which takes surface coordinates and incident angle as input.
    These are computed in the render_by_surface from the terminal ray phase.

    Parameters
    ----------
    start_phase : Array
        Coordinate-frame phase to march from.
    scene : Scene
        The entities; their SDFs and surface colours receive static-frame
        phases.

    Returns
    -------
    phase : Array
        Terminal coordinate-frame phase, for further marching.
    color : Array
    """
    sdf = _static(scene.sdf)
    terminal_event = terminate_by_phase(lambda phase: sdf(phase) < dtol / 2)
    phase = gr_raymarch(start_phase, terminal_event, end_time=timespan, aux_fn=aux_phase_dyn,
                        stepsize_controller=adaptive_stepper(sdf))
    # Actually the early termination condition already gives the is_hit...
    is_hit = sdf(phase) < dtol

    # Find the closest entity for shading
    color_surf = scene.surface_color(coordframe_to_staticframe(phase))
    color_back = jnp.zeros_like(color_surf)

    return phase, jax.lax.select(is_hit, color_surf, color_back)

@partial(jax.jit, static_argnames=['entities', 'aux_phase_dyn'])
def batch_render_by_surface(start_phase, entities, 
                            aux_phase_dyn=None, dtol=1e-4, timespan=24.0):
    batch_render = jax.vmap(render_by_surface, 
        in_axes=(0, None, None, None, None)
        )(start_phase, entities, aux_phase_dyn, dtol, timespan)
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
    which takes the terminal position and static-frame photon direction.

    Parameters
    ----------
    start_phase : Array
        Coordinate-frame phase to march from.
    shapes : tuple
        A container of the signed distance functions.
    brdf : callable
        brdf(position, direction), with direction the static-frame photon
        direction (pointing away from the surface for emitted light).

    Returns
    -------
    phase : Array
        Terminal coordinate-frame phase, for further marching.
    color : Array
    """
    # Construct the scene sdf from the list of items
    def scene_sdf(phase):
        return sdmin_scene(shapes, phase)
    
    terminal_event = terminate_by_position(lambda position: scene_sdf(position) < dtol / 2)
    phase = gr_raymarch(start_phase, terminal_event, end_time = timespan, aux_fn = aux_phase_dyn,
                        stepsize_controller=adaptive_stepper(scene_sdf))
    static_phase = coordframe_to_staticframe(phase)

    color_surf = brdf(static_phase[:3], static_phase[3:6])
    color_back = jnp.zeros_like(color_surf)

    return phase, jax.lax.select(scene_sdf(phase) < dtol, color_surf, color_back)

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
    """Renders using anisotropic (direction-dependent) shapes.

    The shapes' SDFs depend on the photon direction (e.g. adiabatic
    surfaces, which project the field onto it), so they are evaluated on the
    static-frame phase at every step of the march.
    """
    scene_sdf = _static(lambda phase: sdmin_scene(shapes, phase))

    terminal_event = terminate_by_phase(lambda phase: scene_sdf(phase) < dtol / 2)
    phase = gr_raymarch(start_phase, terminal_event, end_time=timespan, aux_fn=aux_phase_dyn,
                        stepsize_controller=adaptive_stepper(scene_sdf))
    static_phase = coordframe_to_staticframe(phase)

    color_surf = brdf(static_phase[:3], static_phase[3:6])
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
    """Initialises a coordinate-frame ray phase from a pixel.

    Rays start parallel to -z at z = focal_distance: a screen at finite
    distance, not an observer at infinity (see ``construct_obs_phases``).
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
        The coordinate-frame ray phases at the screen.
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

def construct_obs_rays(xres=400, yres=400, size=10.0, los=jnp.array([0.0, 0.0, 1.0])):
    """Constructs ray invariants for an observer at infinity.

    Parameters
    ----------
    xres, yres : int
        Pixel resolution.
    size : float
        Half-size of the sky plane.
    los : Array (3,)
        Unit direction from the origin toward the observer, as taken by
        ``rays.observer_anchor`` and ``rays.approx_coordphase_at_r``.

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

def construct_obs_phases(xres=400, yres=400, size=10.0, los=jnp.array([0.0, 0.0, 1.0]),
                         r_start=200.0, n_aux=0):
    """Coordinate-frame start phases for an observer at infinity.

    Each pixel's ray is placed at radius ``r_start`` on its Beloborodov
    trajectory, ready for ``gr_raymarch`` to take over.  At large r_start the
    approximation is excellent (psi -> 0), so the march starts on the ray an
    observer at infinity would see, unlike the parallel screen of
    ``construct_screen_rays``.  Pixels whose ray never reaches r_start are
    NaN, and ``r_start`` must lie outside every scene object.

    Returns
    -------
    phases : Array (N, 6 + n_aux)
    """
    b2s, los_perps, los = construct_obs_rays(xres, yres, size, los)
    phases = jax.vmap(approx_coordphase_at_r, in_axes=(None, 0, None, 0))(
        r_start, b2s, los, los_perps)
    return jnp.concatenate([phases, jnp.zeros((phases.shape[0], n_aux))], axis=-1)