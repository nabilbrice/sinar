import jax
import jax.numpy as jnp
from diffrax import ODETerm

def potential(t, q, l2):
    return l2*0.5/jnp.sqrt(q[0]**2 + q[1]**2 + q[2]**2)**3

accel = jax.grad(potential, argnums=1)

def initial_l2(q, p):
    lvec = jnp.linalg.cross(q, p)
    return jnp.linalg.vecdot(lvec, lvec)

def hamiltonian(t, y, l2):
    return jnp.concatenate([y[...,3:], accel(t, y[...,:3], l2)])

term = ODETerm(hamiltonian)