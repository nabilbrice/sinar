import jax
import jax.numpy as jnp
from jax import Array
from ..rays import normalize

# If only the direction matters then the distance division is a waste,
# especially if the direction needs to be normalized again afterwards.
def mag_potential(components: Array, orient: Array, position: Array):
    moment = jnp.array([0.0, 0.0, 1.0])
    position = jnp.matmul(orient, position)
    point_distance = jnp.linalg.norm(position)
    cosine = jnp.vecdot(moment, position) / point_distance

    return (-0.5 * components[0] * cosine / point_distance**2 
            -1./3.*components[1] * (1.5 * cosine**2 - 0.5) / point_distance**3
            -0.25 * components[2] * (2.5 * cosine**3 - 1.5 * cosine) / point_distance**4
            )

mag_vector = jax.grad(mag_potential, argnums=2)

# For mixed harmonics, the vector direction addition needs to be weighted.
def dipole_dir(orient: Array, position: Array):
    moment = jnp.matmul(orient, jnp.array([0.0, 0.0, 1.0]))
    position = position / jnp.linalg.norm(position)
    return 3.0 * jnp.vecdot(moment, position) * position - jnp.vecdot(position, position) * moment

def stokes_rotation(components: Array, orient: Array, position: Array, ray_dir: Array):
    """Computes the rotation for the local Stokes parameters as a complex number.

    The rotation transforms the local Stokes parameters,
    which must be represented by the complex number P = Q + iU,
    to the detector frame.
    """
    mag_dir = mag_vector(components, orient, position)

    cross = normalize(jnp.cross(mag_dir, ray_dir))

    cos = jnp.dot(cross, jnp.array([1.0, 0.0, 0.0]))
    sin = jnp.dot(cross, jnp.array([0.0, -1.0, 0.0]))
    z = cos + 1j * sin

    return z**2

def adiabatic_factor(energy: float, components: Array) -> float:
    """Computes the scaling factor between the adiabatic radius to the surface radius.
    """
    # 7.61
    return 7.61 * (components[0]/1.2**3)**(0.4) * energy**(0.2)