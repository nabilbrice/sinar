import jax
import jax.numpy as jnp
from jax import Array
from ..rays import normalize

# If only the direction matters then the distance division is a waste,
# especially if the direction needs to be normalized again afterwards.
def potential(components: Array, orient: Array, position: Array):
    moment = jnp.matmul(orient, jnp.array([0.0, 0.0, 1.0]))
    point_distance = jnp.linalg.norm(position)
    cosine = jnp.vecdot(moment, position) / point_distance

    return -0.5 * components[0] * cosine / point_distance**2 \
            -1./3.*components[1] * (1.5 * cosine**2 - 0.5) / point_distance**3

dipole_vector = jax.grad(potential, argnums=2)

# For mixed harmonics, the vector direction addition needs to be weighted.
def dipole_dir(orient: Array, position: Array):
    moment = jnp.matmul(orient, jnp.array([0.0, 0.0, 1.0]))
    position = position / jnp.linalg.norm(position)
    return 3.0 * jnp.vecdot(moment, position) * position - jnp.vecdot(position, position) * moment

def cross_and_project(orient: Array, position: Array, direction: Array):
    dipole_direction = dipole_vector(jnp.array([1.0, 0.0]), orient, position)

    cross = normalize(jnp.cross(dipole_direction, direction))

    x_proj = jnp.dot(cross, jnp.array([1.0, 0.0, 0.0]))
    y_proj = jnp.dot(cross, jnp.array([0.0, 1.0, 0.0]))

    return x_proj, y_proj