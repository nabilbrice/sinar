from typing import Callable
from functools import partial
import jax
import equinox as eqx
from jax import Array
import jax.numpy as jnp
from diffrax import ODETerm, Tsit5, Bosh3, Ralston, diffeqsolve, Event, PIDController
from diffrax._step_size_controller.base import AbstractStepSizeController

def raymarch(phase, terminate_fn: Callable, dist_fn: Callable, max_steps=160,
             safety = 1.0, dt_floor=1e-4) -> float:
    """Marches a ray in Euclidean space until termination.

    Returns
    -------
    t : float
        The parameter along the ray after max_steps.
    """
    def cond(carry):
        t, y, i = carry
        return (~terminate_fn(y[0:3])) & (i < max_steps)

    def body(carry):
        t, y, i = carry
        dt = jnp.maximum(safety * dist_fn(y[0:3]), dt_floor)
        return (t + dt, y.at[0:3].add(dt * y[3:6]), i+1)

    _, phase, _ = jax.lax.while_loop(
        cond, body, (jnp.asarray(0.0), phase, jnp.asarray(0))
    )
    return phase

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
    return l2/jnp.sqrt(q[0]**2 + q[1]**2 + q[2]**2)**3

# "Pseudo-acceleration" acting on the ray veloctiy.
accel = jax.jit(jax.grad(potential, argnums=1))

def initial_l2(q, p):
    lvec = jnp.linalg.cross(q, p)
    return jnp.linalg.vecdot(lvec, lvec)

def terminate_by_position(fn: Callable) -> Event:
    """Constructs an event to terminate the marching by the ray position.

    This function allows functions that take more than the ray phase as input.
    """
    return Event(lambda t, y, args, **kwargs: fn(y[:3]))

def terminate_by_phase(fn: Callable) -> Event:
    """Constructs an event to terminate the marching by ray phase.
    """
    return Event(lambda t, y, args, **kwargs: fn(y))

def gr_raymarch(phase, terminal_event: Event, end_time=24.0, aux_fn = None,
                stepsize_controller=PIDController(dtmax=1/16, rtol=1e-6, atol=1e-8)) -> float:
    # Initial conditions
    l2 = initial_l2(phase[:3], phase[3:6])

    if aux_fn is None:
        def phase_dyn(t, y, l2):
            return jnp.concatenate([y[3:6], accel(t, y[:3], l2), y[6:]])
    else:
        def phase_dyn(t, y, l2):
            aux_dy = aux_fn(t, y, l2)
            return jnp.concatenate([y[3:6], accel(t, y[:3], l2), aux_dy])

    term = ODETerm(phase_dyn)
    
    solution = diffeqsolve(
        term,
        Bosh3(), # Possibly using a lower-order solver is just as good when dtmax is low
        # As a baseline, it seems like 3rd order is sufficient for dtmax = 1/8
        t0=0.0,
        t1=end_time,
        dt0=1.0,
        y0=phase,
        args=l2,
        # dtmax has to be low in order to safely resolve the features anyway
        stepsize_controller=stepsize_controller,
        event=terminal_event,
        throw=False
    )

    return solution.ys[0]

class SDFClipController(AbstractStepSizeController):
    """Step size controller that clips to ensure the scene SDF.

    Wraps an inner (e.g. PID) controller, to which the accuracy-driven step selection
    is delegated. Every step is capped so that the Euclidean arc length of next advancement
    is not exceeded by `safety * |sdf(pos)|`.

    Far from surfaces, the geodesic error control sets the step, so no global dtmax required.

    Parameters
    ----------
    inner : AbstractStepController
        The wrapped accuracy controller, e.g. `PIDController(rtol=1e-6, atol=1e-8)`.
    dist_fn: Callable
        Distance bound as a function of the full phase y.
        For isotropic scene SDF pass `lambda y: sdf(y[:3])`.
    safety : float
        Fraction of the SDF bound to advance per step (< 1).
    dt_floor :
        Minimum step, preventing a stall as sdf -> 0 and letting terminal event (sdf < dtol)
        trigger. Should be order of dtol.
    """
    inner: AbstractStepSizeController
    dist_fn: Callable = eqx.field(static=True)
    safety: float = 1.0
    dt_floor: float = 1e-4

    def wrap(self, direction):
        return eqx.tree_at(lambda s: s.inner, self, self.inner.wrap(direction))

    def _cap(self, y):
        d = self.dist_fn(y[0:3])
        speed = jnp.linalg.vector_norm(y[3:6]) # d(arclength)/d(parameter)
        return jnp.maximum(self.safety * d / speed, self.dt_floor)

    def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
        t1_, state = self.inner.init(terms, t0, t1, y0, dt0, args,
                                     func, error_order)
        return t0 + jnp.maximum(t1_ - t0, self._cap(y0)), state

    def adapt_step_size(self, t0, t1, y0, y1, args, y_error, error_order, state):
        keep, nt0, nt1, jump, state, result = self.inner.adapt_step_size(
            t0, t1, y0, y1, args, y_error, error_order, state
        )
        y_next = jax.tree.map(lambda a, b: jnp.where(keep, a, b), y1, y0)
        nt1 = jnp.minimum(nt1, nt0 + self._cap(y_next))
        return keep, nt0, nt1, jump, state, result

def adaptive_stepper(dist_fn, safety=1.0, rtol=1e-6, atol=1e-8):
    return SDFClipController(inner=PIDController(rtol=rtol, atol=atol), dist_fn=dist_fn, safety=safety)

def aniso_sdf_gr_raymarch(start_phase, sdf, end_time=24.0, aux_fn=None, dtol=1e-4):
    terminal_event = terminate_by_phase(lambda phase: sdf(phase) < dtol)
    return gr_raymarch(start_phase, terminal_event, end_time=end_time, aux_fn=aux_fn)

# Approximation of Beloborodov (2002)
@jax.jit
def _approx_raytraj_at_r(r, b2):
    g2 = 1.0 - 2.0 / r
    q = jnp.sqrt(1.0 - b2 * g2 / (r * r))

    los_para = r * (q + g2 - 1.0) / g2
    los_perp = jnp.sqrt(r * r - los_para * los_para)

    cpsi = los_para / r
    spsi = los_perp / r

    calpha = q * jnp.sqrt(g2) # radial direction needs metric g00
    salpha = jnp.sqrt(1.0 - q * q)

    dir_perp = -calpha * spsi + salpha * cpsi
    dir_para = -calpha * cpsi - salpha * spsi

    # further simplification of the norm doesn't avoid the sqrt
    norm = jnp.sqrt(dir_perp * dir_perp + dir_para * dir_para)
    dir_perp = dir_perp / norm
    dir_para = dir_para / norm

    return los_perp, los_para, dir_perp, dir_para

@jax.jit
def _approx_rayphase_at_r(r, b2, los = jnp.array([0.0, 0.0, 1.0]), los_perp = jnp.array([0.0, 1.0, 0.0])):
    pos_perp, pos_para, dir_perp, dir_para = _approx_raytraj_at_r(r, b2)
    q = pos_perp * los_perp + pos_para * los
    p = dir_perp * los_perp + dir_para * los
    return jnp.concatenate([q, p])

@jax.jit
def _approx_raytraj_at_altitude(altitude, b2):
    c = jnp.cos(altitude)
    s = jnp.sin(altitude)
    t = (1.0 - c) / (1.0 + c)
    r = jnp.sqrt(t * t + b2 / (s * s)) - t
    
    los_para = r * c
    los_perp = r * s

    g2 = 1.0 - 2.0 / r
    q = jnp.sqrt(1.0 - b2 * g2 / (r * r))

    calpha = q * jnp.sqrt(g2)
    salpha = jnp.sqrt(1.0 - q * q)

    dir_perp = -calpha * s + salpha * c
    dir_para = -calpha * c - salpha * s

    norm = jnp.sqrt(dir_perp * dir_perp + dir_para * dir_para)
    dir_perp = dir_perp / norm
    dir_para = dir_para / norm

    return los_perp, los_para, dir_perp, dir_para

@jax.jit
def _approx_rayphase_at_altitude(altitude, b2, los = jnp.array([0.0, 0.0, 1.0]), los_perp = jnp.array([0.0, 1.0, 0.0])):
    pos_perp, pos_para, dir_perp, dir_para = _approx_raytraj_at_altitude(altitude, b2)
    q = pos_perp * los_perp + pos_para * los
    p = dir_perp * los_perp + dir_para * los
    return jnp.concatenate([q, p])

@jax.jit
def obs_ray_invariants(pixlocs: Array, e1: Array, e2: Array) -> tuple[Array, Array]:
    """Computes the per-ray invariants from pixel locations and sky-plane basis.

    Parameters
    ----------
    pixlocs : Array (N, 2)
    e1, e2 : Array (3,)

    Returns
    -------
    b2s : Array (N,)
        Impact parameter squared for each ray.
    los_perps : Array (N, 3)
        Per-ray perpendicular direction in the sky plane.
    """
    b_vecs = pixlocs[:, 0:1] * e1 + pixlocs[:, 1:2] * e2  # (N, 3)
    b2s = jnp.sum(b_vecs * b_vecs, axis=-1)                # (N,)
    safe = b2s > 1e-12
    los_perps = jnp.where(safe[:, None], b_vecs / jnp.sqrt(b2s)[:, None], e1)
    return b2s, los_perps
