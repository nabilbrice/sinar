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
        Distance bound as a function of the full phase y (coordinate frame).
        Scene SDFs take a phase and slice its position, so they can be
        passed directly.
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
        d = self.dist_fn(y)
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

# Beloborodov (2002) approximation
#
#     1 - cos(alpha) = (1 - cos(psi)) (1 - 2/r)
#
# psi is the polar angle of the ray's position from its asymptotic direction
# and alpha the static-frame angle between the ray and the local radius.  The
# relation holds at every point along a ray (Beloborodov 2002, sec. 2), and
# only psi(r) is approximate: alpha(r) follows exactly from b.  Lengths are
# in GM/c^2 throughout.
#
# A ray is described once, by its invariants and an anchor:
# * b2, the squared impact parameter;
# * an anchor (r_hat_a, m_hat_a, R_a): the unit radius vector at radius R_a,
#   and the unit vector orthogonal to it in the orbital plane toward which
#   the ray sweeps as r increases.  ``observer_anchor`` gives the anchor at
#   R_a = inf from sky coordinates, the ray constructors give the anchor at
#   the stellar surface, and ``reanchor`` moves between them.  Every anchor
#   of a ray has the same plane normal n_hat = r_hat_a x m_hat_a.
#
# Two kinds of phase (position, direction) exist, and they differ only in the
# direction; positions are Schwarzschild coordinates in both:
# * static-frame phase: the photon's propagation direction measured by a
#   static observer.  This is the physical direction, and the frame the field
#   components are given in: anything that projects onto it (B_perp, mode
#   angles, emission angles) needs this phase.
# * coordinate-frame phase: the backward-traced tangent of the ray in the
#   Euclidean embedding of Schwarzschild coordinates.  This is the state
#   ``gr_raymarch`` integrates, so approximate phases that seed or replace a
#   march must be in this frame.
#
# ``approx_statictraj_at_r`` is the trajectory; ``staticframe_to_coordframe``
# and ``coordframe_to_staticframe`` convert phases at the marcher boundary.
def _psi_trig(r, b2):
    """(cos psi, sin psi, cos alpha) at radius r, on the outgoing branch.

    Solves the cosine relation as y = 1 - cos psi = (b^2/r^2) / (1 + cos alpha)
    and sin psi = (b/r) sqrt((2 - y) / (1 + cos alpha)), which is
    sqrt(y (2 - y)) with the b^2/r^2 factor taken outside the root.  Every
    step is a sum, product or quotient of positive terms, so sin psi keeps
    full relative precision as b^2/r^2 -> 0; the textbook
    sqrt(1 - cos^2 psi) cancels there and can round to NaN at b^2 = 0.
    Taking b outside the root keeps d/db2 finite wherever b2 > 0, including
    at r = inf (psi = 0, alpha = 0), where sqrt(y (2 - y)) with y = 0 would
    give 0 * inf = NaN.

    NaN where the ray does not reach r, b^2 (1 - 2/r) > r^2.  Callers use that
    as a miss mask, so there is deliberately no guard; the ray constructors
    keep b2 off the tangential limit instead.
    """
    q = jnp.sqrt(1.0 - b2 / r**2 * (1.0 - 2.0 / r))
    y = (b2 / r**2) / (1.0 + q)
    return 1.0 - y, jnp.sqrt(b2) / r * jnp.sqrt((2.0 - y) / (1.0 + q)), q

def _sweep(r, b2, r_hat_a, m_hat_a, R_a):
    """(r_hat, phi_hat, cos alpha) at radii r, for the ray anchored at R_a.

    The ray has swept through psi_a - psi(r) since the anchor; cos/sin of it
    come from the addition formulae, so no arccos appears and derivatives in
    r and b2 stay finite as psi -> 0.  phi_hat is the in-plane unit vector
    orthogonal to r_hat in the direction of increasing r.
    """
    r = jnp.asarray(r)
    ca, sa, _ = _psi_trig(R_a, b2)
    cp, sp, q = _psi_trig(r, b2)
    cd = (ca * cp + sa * sp)[..., None]           # cos(psi_a - psi)
    sd = (sa * cp - ca * sp)[..., None]           # sin(psi_a - psi)
    return cd * r_hat_a + sd * m_hat_a, -sd * r_hat_a + cd * m_hat_a, q

def approx_statictraj_at_r(r, b2, r_hat_a, m_hat_a, R_a):
    """Positions and static-frame photon directions at radii r.

    Parameters
    ----------
    r : Array (...)
        Schwarzschild areal radii on the outgoing branch, any shape.
        Evaluation is pointwise, so a grid need not be uniform or sorted.
    b2 : float
        Squared impact parameter, conserved along the ray.  For emission at
        static-frame angle alpha_R to the normal at radius R,
        b^2 = R^2 sin^2(alpha_R) / (1 - 2/R).
    r_hat_a, m_hat_a : Array (3,)
        Anchor unit vectors, orthonormal and not checked: the radius vector
        at R_a, and the in-plane direction the ray sweeps toward as r
        increases.  At a surface anchor m_hat_a is the azimuthal direction at
        emission, not the photon direction; the emitted direction is
        cos(alpha_R) r_hat_a + sin(alpha_R) m_hat_a.  A non-unit r_hat_a
        rescales every position.
    R_a : float
        Anchor radius: the stellar radius for a surface anchor, inf for an
        observer anchor (see ``observer_anchor``).

    Returns
    -------
    pos : Array (..., 3)
        r * r_hat.  Only its azimuthal location carries the Beloborodov
        error; |pos| = r exactly.
    k : Array (..., 3)
        Unit photon direction in the local static frame,
        cos(alpha) r_hat + sin(alpha) phi_hat.  Exact for the given r.
    q : Array (...)
        cos(alpha) >= 0.
        All three are NaN where the ray does not reach r.
    """
    r_hat, phi_hat, q = _sweep(r, b2, r_hat_a, m_hat_a, R_a)
    r = jnp.asarray(r)
    sin_a = (jnp.sqrt(b2) / r * jnp.sqrt(1.0 - 2.0 / r))[..., None]
    return r[..., None] * r_hat, q[..., None] * r_hat + sin_a * phi_hat, q

def observer_anchor(los, los_perp):
    """(r_hat_a, m_hat_a, R_a) anchoring a ray at the observer.

    ``los`` is the unit asymptotic direction toward the observer and
    ``los_perp`` the unit sky-plane offset of the pixel (e.g. from
    ``obs_ray_invariants``).  The sweep vector is -los_perp: seen from the
    observer's end, the ray approaches los from the los_perp side as r
    increases.
    """
    return los, -los_perp, jnp.inf

def reanchor(b2, r_hat_a, m_hat_a, R_a, R_b):
    """(r_hat_b, m_hat_b): the same ray's anchor moved from R_a to R_b.

    ``reanchor(b2, *observer_anchor(los, los_perp), R)`` gives the surface
    anchor of a sky pixel, and ``reanchor(b2, r_hat0, m_hat, R, jnp.inf)``
    gives the observer anchor (los, -los_perp) of a surface ray.  The plane
    normal r_hat x m_hat is unchanged.
    """
    r_hat, phi_hat, _ = _sweep(R_b, b2, r_hat_a, m_hat_a, R_a)
    return r_hat, phi_hat

def _radial_split(pos, d):
    """(r, radial part, tangential part) of directions d at positions pos."""
    r = jnp.linalg.norm(pos, axis=-1, keepdims=True)
    r_hat = pos / r
    d_r = jnp.sum(d * r_hat, axis=-1, keepdims=True) * r_hat
    return r, d_r, d - d_r

def staticframe_to_coordframe(phase):
    """Coordinate-frame phase from a static-frame phase.

    The static observer's radial rod is sqrt(1 - 2/r) times a coordinate
    step, so the tangential part is divided by sqrt(1 - 2/r) relative to
    the radial part; the result is renormalised and reversed to point along
    the march rather than the photon.  Positions and any components after
    the sixth (e.g. auxiliary marcher state) are passed through.

    Parameters
    ----------
    phase : Array (..., 6+)
        Position, then static-frame photon direction.

    Returns
    -------
    phase : Array (..., 6+)
        Position, then unit backward coordinate direction.
    """
    r, k_r, k_t = _radial_split(phase[..., :3], phase[..., 3:6])
    d = -normalize(k_r + k_t / jnp.sqrt(1.0 - 2.0 / r))
    return phase.at[..., 3:6].set(d)

def coordframe_to_staticframe(phase):
    """Static-frame phase from a coordinate-frame phase.

    Inverse of ``staticframe_to_coordframe``.  Apply it to anything from
    ``gr_raymarch`` or ``approx_coordphase_at_r`` before projecting onto the
    direction.  The input direction need not be normalised.

    Parameters
    ----------
    phase : Array (..., 6+)
        Position, then backward coordinate direction.

    Returns
    -------
    phase : Array (..., 6+)
        Position, then unit static-frame photon direction.
    """
    r, d_r, d_t = _radial_split(phase[..., :3], phase[..., 3:6])
    k = -normalize(d_r + jnp.sqrt(1.0 - 2.0 / r) * d_t)
    return phase.at[..., 3:6].set(k)

@jax.jit
def approx_coordphase_at_r(r, b2, los=jnp.array([0.0, 0.0, 1.0]),
                           los_perp=jnp.array([0.0, 1.0, 0.0])):
    """Coordinate-frame phase at radius r for the sky pixel (b2, los_perp).

    The approximate counterpart of a ``gr_raymarch`` state, so it can seed or
    replace a march.  Convert with ``coordframe_to_staticframe`` before any
    physics.

    Parameters
    ----------
    r : float or Array (...)
        Schwarzschild areal radius.
    b2 : float
        Squared impact parameter.
    los : Array (3,)
        Unit asymptotic direction toward the observer.
    los_perp : Array (3,)
        Unit sky-plane offset of the pixel, orthogonal to ``los``.

    Returns
    -------
    phase : Array (..., 6)
        Position, then unit backward coordinate direction; NaN on a miss.
    """
    pos, k, _ = approx_statictraj_at_r(r, b2, *observer_anchor(los, los_perp))
    return staticframe_to_coordframe(jnp.concatenate([pos, k], axis=-1))

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

def ds_dr(r, q):
    """Proper length along the ray per unit coordinate radius, static frame.

    ds/dr = 1 / (sqrt(1 - 2/r) cos(alpha)).  Floored at cos(alpha) = 1e-12
    so a ray launched tangentially at the surface stays finite.
    """
    return 1.0 / (jnp.sqrt(1.0 - 2.0 / r) * jnp.maximum(q, 1e-12))

def screen_basis(n_hat, k):
    """Parallel-transported polarization screen (e1, e2) = (n_hat x k, n_hat).

    For a planar Schwarzschild geodesic the orbital-plane normal is itself
    parallel transported, so this basis carries no gravitational Faraday
    rotation and Stokes vectors on it need no frame-rotation term.

    Parameters
    ----------
    n_hat : Array (3,)
        Plane normal r_hat_a x m_hat_a, the same for every anchor.
    k : Array (..., 3)
        Static-frame photon directions, e.g. from ``approx_statictraj_at_r``.

    Returns
    -------
    e1, e2 : Array (..., 3)
    """
    e2 = jnp.broadcast_to(n_hat, k.shape)
    return normalize(jnp.cross(e2, k)), e2

# surface-anchored ray constructors
def _safe_b2(b2, R, b2_floor):
    """b2 kept clear of the two points where a ray starting at R is singular.

    Floor: sin psi is proportional to b = sqrt(b2), whose derivative is
    infinite at b2 = 0, so a strictly radial ray makes a jvp in b2 NaN.  Tangential limit
    b2 = R^2 / (1 - 2/R): cos(alpha_R) = 0 there, and rounding makes
    _psi_trig(R, b2) NaN in a few percent of cases.  Values within 1e-12 of
    the limit are moved 1e-14 below it (alpha_R short of 90 deg by ~6e-6
    deg); anything further out is a genuine miss and is left alone.
    """
    floor = (1e-6 * R) ** 2 if b2_floor is None else b2_floor
    limit = R**2 / (1.0 - 2.0 / R)
    b2 = max(float(b2), floor)
    return limit * (1.0 - 1e-14) if abs(b2 - limit) <= 1e-12 * limit else b2

def ray_from_surface(r_hat0, m_hat, alpha, R, b2_floor=None):
    """(b2, r_hat0, m_hat, n_hat) for a ray leaving R * r_hat0 at angle alpha.

    The surface anchor of the ray, with R_a = R.  ``alpha`` is the
    static-frame emission angle to the local normal, in [0, pi/2]; the ray
    sweeps toward ``m_hat``, which must be orthonormal to ``r_hat0``.
    """
    b2 = _safe_b2(R**2 * jnp.sin(alpha) ** 2 / (1.0 - 2.0 / R), R, b2_floor)
    r_hat0, m_hat = jnp.asarray(r_hat0, float), jnp.asarray(m_hat, float)
    return b2, r_hat0, m_hat, normalize(jnp.cross(r_hat0, m_hat))

def ray_from_image(b, phi, R, b2_floor=None):
    """(b2, r_hat0, m_hat, n_hat) for the image pixel at sky polar (b, phi).

    The surface anchor of the ray, with R_a = R.  The observer lies along
    los = +z and ``phi`` is measured from x toward y.  For another line of
    sight, take (b2, los_perp) from ``obs_ray_invariants`` and use
    ``reanchor(b2, *observer_anchor(los, los_perp), R)``.
    """
    b2 = _safe_b2(float(b) ** 2, R, b2_floor)
    los = jnp.array([0.0, 0.0, 1.0])
    los_perp = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
    r_hat0, m_hat = reanchor(b2, *observer_anchor(los, los_perp), R)
    return b2, r_hat0, m_hat, normalize(jnp.cross(r_hat0, m_hat))