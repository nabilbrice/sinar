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
accel = jax.grad(potential, argnums=1)

def initial_l2(q, p):
    lvec = jnp.linalg.cross(q, p)
    return jnp.linalg.vecdot(lvec, lvec)

def hamiltonian(t, y, l2):
    return jnp.concatenate([normalize(y[3:]), accel(t, y[...,:3], l2)])

term = ODETerm(hamiltonian)

def terminate_by_position(fn: Callable) -> Event:
    """Constructs an event to terminate the marching by the ray position.
    """
    return Event(lambda t, y, args, **kwargs: fn(y[:3]))

def terminate_by_phase(fn: Callable) -> Event:
    """Constructs an event to terminate the marching by ray phase.
    """
    return Event(lambda t, y, args, **kwargs: fn(y))

def gr_raymarch(phase, terminal_event: Event, end_time=24.0) -> float:
    # Initial conditions
    l2 = initial_l2(phase[:3], phase[3:])
    
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
    )

    return solution.ys[0]

def sdf_gr_raymarch(start_phase, sdf, dtol = 1e-4, end_time = 24.0):
    terminal_event = terminate_by_position(lambda p: sdf(p) < dtol)
    return gr_raymarch(start_phase, terminal_event, end_time)

# multiple stage gr_raymarch
def staged_gr_raymarch(phase, staged_sdf, dtol = 1e-4, end_times=jnp.array([24.0])):
    """Multi-stage GR raymarch.
    """
    n_stages = len(staged_sdf)
    end_times = jnp.broadcast_to(end_times, n_stages)
    phases = jnp.zeros((n_stages, 6))
    for i, sdf in enumerate(staged_sdf):
        phase = gr_raymarch(phase,
                            terminate_by_position(lambda p: sdf(p) < dtol),
                            end_time = end_times[i]
                            )
        phases = phases.at[i].set(phase)
    return phases