import jax
import jax.numpy as jnp
from jax import Array
from ..rays import normalize
from .shapes import rotation

# If only the direction matters then the distance division is a waste,
# especially if the direction needs to be normalized again afterwards.
def mag_potential(components: Array, orient: Array, position: Array):
    moment = jnp.matmul(jnp.array([0.0, 0.0, 1.0]), orient)
    point_distance = jnp.linalg.norm(position)
    cosine = jnp.vecdot(moment, position) / point_distance

    return (-0.5 * components[0] * cosine / point_distance**2 
            -1./3.*components[1] * (1.5 * cosine**2 - 0.5) / point_distance**3
            -0.25 * components[2] * (2.5 * cosine**3 - 1.5 * cosine) / point_distance**4
            )

mag_vector = jax.grad(mag_potential, argnums=2)

def gr_dipole_vector(orient: Array, position: Array):
    """GR-corrected magnetic dipole field vector.
    From the Appendix of Page & Sarmiento (1996)."""
    moment = jnp.matmul(jnp.array([0.0, 0.0, 1.0]), orient)
    
    r = jnp.linalg.norm(position)
    # Radial vector:
    r_hat = position / r
    cos_theta = jnp.vecdot(moment, r_hat)
    # Latitude vector, blows up at cos_theta = 1, along the moment axis
    theta_hat = (moment - cos_theta * r_hat) / jnp.sqrt(1.0 - cos_theta**2)
    
    # Flat-space field (matching mag_vector)
    B_flat = (0.5 / r**3) * (3.0 * cos_theta * r_hat - moment)
    
    # In mass units of radius, the Schwarzschild radius Rs = 2M
    x = 2.0 / r

    f1 = -3.0 / x**3 * (jnp.log1p(-x) + 0.5 * x * (x + 2.0))
    g1 = -2.0 * f1 + 3.0 / (1.0 - x)
    alpha = jnp.sqrt(1.0 - x)
    
    # Corrections are to the spherical components, then reconstruct
    B_r_gr = jnp.vecdot(B_flat, r_hat) * f1
    B_theta_gr = jnp.vecdot(B_flat, theta_hat) * g1 * alpha
    
    return B_r_gr * r_hat + B_theta_gr * theta_hat

def gr_quadrupole_vector(orient: Array, position: Array):
    """GR-corrected magnetic quadrupole field vector. Axially symmetric case.
    From the Appendix of Page & Sarmiento (1996)."""
    moment = jnp.matmul(jnp.array([0.0, 0.0, 1.0]), orient)

    r = jnp.linalg.norm(position)
    # Radial vector:
    r_hat = position / r
    cos_theta = jnp.vecdot(moment, r_hat)
    # Latitude vector, blows up at cos_theta = 1, along the moment axis
    theta_hat = (moment - cos_theta * r_hat) / jnp.sqrt(1.0 - cos_theta**2)

    # Flat-space field (matching mag_vector for quadrupole component)
    B_flat = (0.5 / r**4) * (
        (5.0 * cos_theta**2 - 1.0) * r_hat 
        - 2.0 * cos_theta * moment
    )
    
    x = 2.0 / r
    log_term = jnp.log1p(-x)
    # These suffer from catastrophic cancellations when x -> 0 if JAX_ENABLE_X64 is not set
    f2 = 10.0/3.0/x**4 * (
        6 * log_term * (3*x - 4) / x + x**2 + 6*x - 24
    )
    g2 = 10.0/x**4 * (
        6 * log_term * (2 - x)/x + (x**2 - 12*x + 12)/(1 - x)
    )
    alpha = jnp.sqrt(1.0 - x)

    # Corrections are to the spherical components, then reconstruct
    B_r_gr = jnp.vecdot(B_flat, r_hat) * f2
    B_theta_gr = jnp.vecdot(B_flat, theta_hat) * g2 * alpha
    
    return B_r_gr * r_hat + B_theta_gr * theta_hat

def magnetic_field_strength(components: Array, orient: Array, position: Array) -> float:
    """Magnitude of the magnetic field vector."""
    B_vector = mag_vector(components, orient, position)
    return jnp.sqrt(jnp.vecdot(B_vector, B_vector))

# Gradient of magnetic field strength with respect to position
grad_mag_field_strength = jax.grad(magnetic_field_strength, argnums=2)

def lengthscale_B(components: Array, orient: Array, position: Array, 
                  direction: Array) -> float:
    """
    Compute length scale along a specific direction.
    
    This computes: |B| / (direction̂ · ∇|B|).
    """
    B_magnitude = magnetic_field_strength(components, orient, position)
    grad_B_magnitude = grad_mag_field_strength(components, orient, position)
    
    direction_normalized = direction / jnp.linalg.norm(direction)
    
    directional_derivative = jnp.abs(jnp.vecdot(direction_normalized, grad_B_magnitude))
    
    return B_magnitude / directional_derivative

def lengthscale_A(energy_keV: float, components: Array, orient: Array, position: Array,
                  direction: Array,
                  M_solar: float = 1.4) -> float:
    """
    QED length scale in mass-scaled geometric units.

    The mass in solar units is required.
    
    Parameters
    ----------
    M_solar : float
        Mass in solar masses (default: 1.4 M_☉ for typical NS)
    """
    # B_gauss = magnetic_field_strength(components, orient, position) # Gauss units
    B_perp_gauss = jnp.linalg.norm(jnp.linalg.cross(direction, mag_vector(components, orient, position)))
    
    C_geometric = 1.00657e+19  # In the geometric mass scaling units of the ray-marcher
    
    return C_geometric / (energy_keV * B_perp_gauss**2 * M_solar) 

def stokes_rotation(components: Array, orient: Array, position: Array, ray_dir: Array):
    """Computes the rotation for the local Stokes parameters as a complex number.

    The rotation transforms the local Stokes parameters,
    which must be represented by the complex number P = Q + iU,
    to the detector frame.
    """
    mag_dir = components[0] * gr_dipole_vector(orient, position) + components[1] * gr_quadrupole_vector(orient, position)

    cross = normalize(jnp.cross(mag_dir, ray_dir))

    cos = jnp.dot(cross, jnp.array([1.0, 0.0, 0.0]))
    sin = jnp.dot(cross, jnp.array([0.0, -1.0, 0.0]))
    z = cos + 1j * sin

    return z**2

def adiabatic_radius(energy_keV: float, components: Array, orient: Array) -> float:
    """
    Computes the adiabatic radius where length_scale_A equals length_scale_B.
    
    The adiabatic radius is defined as the distance from the neutron star center
    where the characteristic length scales of the magnetic field and the 
    photon energy become equal: length_scale_A = length_scale_B.
    
    Args:
        energy_keV: Photon energy in keV
        components: Magnetic field harmonic components array
        orient: Orientation matrix (3x3) for magnetic field geometry
        
    Returns:
        Radius (in units consistent with input position) where the length scales are equal
    """

    orient = rotation(theta = 0.0, phi = 0.0)
    
    def length_scale_difference(radius: float) -> float:
        """
        Compute the difference between length_scale_A and length_scale_B.
        Returns zero at the adiabatic radius.
        """
        # Position at given radius along the z-axis (radial direction)
        position = jnp.array([0.0, 0.0, radius])
        
        # Use radial direction for length_scale_B calculation
        direction = jnp.array([0.0, 0.0, -1.0])
        
        # Compute both length scales
        ls_A = lengthscale_A(energy_keV, components, orient, position)
        ls_B = lengthscale_B(components, orient, position, direction)
        
        # Return difference (zero at adiabatic radius)
        return ls_A - ls_B
    
    # JAX-compatible bisection method
    def bisection_step(carry):
        lower, upper, mid = carry
        f_mid = length_scale_difference(mid)
        f_lower = length_scale_difference(lower)
        
        # Update bounds based on sign of function values
        new_lower = jnp.where(f_mid * f_lower < 0, lower, mid)
        new_upper = jnp.where(f_mid * f_lower < 0, mid, upper)
        new_mid = 0.5 * (new_lower + new_upper)
        
        return new_lower, new_upper, new_mid
    
    def bisection_condition(carry):
        lower, upper, mid = carry
        return jnp.abs(upper - lower) > 1e-3
    
    # Initial bounds and midpoint
    lower = 1.0
    upper = 100.0
    mid = 0.5 * (lower + upper)
    
    # Check if root exists in initial bounds
    f_lower = length_scale_difference(lower)
    f_upper = length_scale_difference(upper)
    
    # If no sign change, try wider bounds
    lower = jnp.where(f_lower * f_upper > 0, 0.1, lower)
    upper = jnp.where(f_lower * f_upper > 0, 1000.0, upper)
    mid = 0.5 * (lower + upper)
    
    # Run bisection using JAX while_loop
    final_lower, final_upper, final_mid = jax.lax.while_loop(
        bisection_condition,
        bisection_step,
        (lower, upper, mid)
    )
    
    return final_mid