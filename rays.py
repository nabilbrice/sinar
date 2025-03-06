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
    # Body function for the loop, a single step
    def raystep(_, position: Array):
        dt = scene_sdf(position)*0.9
        return position + dt * direct

    return jax.lax.fori_loop(0, max_steps, raystep, origin)

@partial(jax.jit, static_argnums=1)
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

def gr_raymarch(origin, direct, scene_sdf, end_time=24.0) -> float:
    # Initial conditions
    l2 = initial_l2(origin, direct)
    phase0 = jnp.concatenate([origin, direct])
    
    # Define the condition function for termination
    def cond_fn(t, y, args, **kwargs):
        q = y[:3]
        return scene_sdf(q) < 1e-6
    
    event = Event(cond_fn)
    
    solution = diffeqsolve(
        term,
        Tsit5(),
        t0=0.0,
        t1=end_time,
        dt0=0.1,
        y0=phase0,
        args=l2,
        stepsize_controller=PIDController(dtmax=1/8, rtol=1e-6, atol=1e-8),
        event=event
    )

    return solution.ys[0]