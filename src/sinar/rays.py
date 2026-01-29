from typing import Callable
from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5, diffeqsolve, Event, PIDController

@partial(jax.jit, static_argnums=[2, 3])
def raymarch(origin: Array, direct: Array, scene_sdf: Callable,
             max_steps=160) -> float:
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
    phase0 = jnp.concatenate([origin, direct])
    # Body function for the loop, a single step
    def raystep(_, phase: Array):
        dt = scene_sdf(phase[0:3])*0.9
        return jnp.concatenate([phase[0:3] + dt * phase[3:6], phase[3:6]])

    return jax.lax.fori_loop(0, max_steps, raystep, phase0)

@partial(jax.jit, static_argnums=[1, 2, 3], inline=True)
def raymarch_to_screen(start_phase: Array, scene_sdf: Callable, screen_distance: float = 10.0,
                       max_steps: int = 160) -> Array:
    """Marches a ray to a screen specified a distance.
    
    The ray marching is done in Euclidean space with a fixed direction
    for each ray. The screen is defined as a distance from the origin
    in the direction of the ray.
    
    Parameters
    ----------
    start_phase : Array [6 +,]
        Initial ray phase [x, y, z, dx, dy, dz, ...].
    scene_sdf : Callable
        The signed distance function.
    screen_distance : float
        Distance from the origin to terminate.
    max_steps : int
        Maximum number of ray marching steps.
        
    Returns
    -------
    phase : Array [6,]
        Final ray phase when reaching screen or max steps.
    """
    def raystep(i: int, phase: Array) -> Array:
        current_pos = phase[:3]
        # Calculate distance from the plane
        distance = jnp.dot(current_pos, -phase[3:6])

        should_continue = distance > screen_distance

        dt = jnp.where(should_continue, scene_sdf(current_pos) * 0.9, 0.0)
        new_pos = current_pos + dt * phase[3:6]

        return jnp.where(should_continue,
                         jnp.concatenate([new_pos, phase[3:]]), # concatenate remaining phase too
                         phase)
    
    return jax.lax.fori_loop(0, max_steps, raystep, start_phase)

@partial(jax.jit, static_argnums=1, inline=True)
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

batch_normalize = jax.vmap(normalize)

def potential(t, q, l2) -> float:
    """Computes the value of the Schwarzschild null-geodesic potential.

    Parameters
    ----------
    t : float
        The parameterization of the geodesic.
    q : Array[...,3]
        The position of the ray.
    l2 : float
        The initial angular momentum squared of the ray.
    
    Returns
    -------
    V : float
        The potential.
    """
    return l2*0.5/jnp.sqrt(q[0]**2 + q[1]**2 + q[2]**2)**3

# "Pseudo-acceleration" acting on the ray veloctiy.
accel = jax.jit(jax.grad(potential, argnums=1))

def initial_l2(q, p):
    lvec = jnp.linalg.cross(q, p)
    return jnp.linalg.vecdot(lvec, lvec)

def terminate_by_position(fn: Callable) -> Event:
    """Constructs an event to terminate the marching by the ray position.
    """
    return Event(lambda t, y, args, **kwargs: fn(y[:3]),)

def terminate_by_phase(fn: Callable) -> Event:
    """Constructs an event to terminate the marching by ray phase.
    """
    return Event(lambda t, y, args, **kwargs: fn(y))

@partial(jax.jit, static_argnames=["terminal_event", "aux_fn"])
def gr_raymarch(phase, terminal_event: Event, end_time=24.0, aux_fn = None) -> float:
    # Initial conditions
    l2 = initial_l2(phase[:3], phase[3:6])

    def phase_dyn(t, y, l2):
        if aux_fn is not None:
            aux_fn(y)
        return jnp.concatenate([y[3:6], accel(t, y[:3], l2), y[6:]])

    term = ODETerm(phase_dyn)
    
    solution = diffeqsolve(
        term,
        Tsit5(),
        t0=0.0,
        t1=end_time,
        dt0=0.1,
        y0=phase,
        args=l2,
        stepsize_controller=PIDController(dtmax=1/8, rtol=1e-6, atol=1e-8),
        event=terminal_event,
        throw=False
    )

    return solution.ys[0]

@partial(jax.jit, static_argnames=["sdf", "aux_fn"])
def sdf_gr_raymarch(start_phase, sdf, end_time = 24.0, aux_fn = None, dtol=1e-4):
    terminal_event = terminate_by_position(lambda p: sdf(p))
    return gr_raymarch(start_phase, terminal_event, end_time, aux_fn=aux_fn)

@partial(jax.jit, static_argnames=["sdf", "aux_fn"])
def quick_sdf_gr_raymarch(start_phase: Array, sdf: Callable, 
                          end_time = 24.0, aux_fn = None, dtol = 1e-4,
                          screen_distance = 50.0, max_euclidean_steps = 160) -> Array:
    """Hybrid ray marching: Euclidean until screen distance, then GR.
    
    Uses Euclidean ray marching until the ray advances to a specified screen,
    then switches to the full GR ray marching with recalculated angular momentum.
    This should be more computationally efficient for rays that start farther
    from the gravitating body where spacetime is approximately flat.
    
    Parameters
    ----------
    start_phase : Array [6,]
        Initial ray phase [x, y, z, dx, dy, dz].
    sdf : Callable
        The signed distance function of the scene.
    dtol : float
        The distance tolerance for when a ray is considered to be
        close enough to a surface.
    screen_distance : float
        Distance from the origin to terminate the Euclidean ray marching.
    end_time : float
        The time to terminate the GR ray marching.
    max_euclidean_steps : int
        Maximum number of steps for the Euclidean ray marching.
    
    Returns
    -------
    phase : Array [6,]
        Final ray phase after the ray marching.
    """
    # Part 1: Euclidean ray marching to screen
    screen_phase = raymarch_to_screen(start_phase, sdf, screen_distance, max_euclidean_steps)

    surface_hit_euclidean = sdf(screen_phase[:3]) < dtol

    # Part 2: GR ray marching from screen position (if no surface is hit only)
    def continue_with_gr():
        terminal_event = terminate_by_position(lambda p: sdf(p) - dtol)
        return gr_raymarch(screen_phase, terminal_event, end_time=screen_distance*2.0, aux_fn = aux_fn)
    
    def return_euclidean_result():
        return screen_phase
    
    final_phase = jax.lax.cond(
        surface_hit_euclidean,
        return_euclidean_result,
        continue_with_gr
    )

    return final_phase

# multiple stage gr_raymarch
def staged_gr_raymarch(phase, staged_sdf, dtol = 1e-4, end_times=jnp.array([24.0])):
    """Multi-stage GR raymarch.
    """
    n_stages = len(staged_sdf)
    end_times = jnp.broadcast_to(end_times, n_stages)
    phases = jnp.zeros((n_stages, 6))
    for i, sdf in enumerate(staged_sdf):
        phase = gr_raymarch(phase,
                            terminate_by_position(lambda p: sdf(p) - dtol),
                            end_time = end_times[i]
                            )
        phases = phases.at[i].set(phase)
    return phases