from sinar.renderers import construct_screen_rays, staged_batch_render, staged_batch_render_by_rayphase, batch_render_by_surface, batch_render_by_rayphase
from sinar.entities.colors import set_brdf_region , set_brdf_dbb, is_cap_region, is_patch_region, is_chequered_region
from sinar.io.visuals import save_frame_as_png, save_frame_as_gif
import jax.numpy as jnp
from sinar.rays import batch_normalize

# Commonly used configuration for bh marching
def create_bh_frame(xres = 400, yres = 400, size = 10.0,
                    focal_distance = 10.0,
                    theta = jnp.pi/2.1, phi = 0.0):
    from sinar.entities.shapes import put_sphere, put_thindisc, rotation

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    bounds = (
        put_sphere(radius = 12.0),
    )
    shapes = (
        # black hole event horizon is 2.0
        put_sphere(radius = 2.0, orient = rotation(phi = phi, theta = theta)),
        put_thindisc(inner=3.0, outer=12.0, height=0.1, orient = rotation(phi = phi, theta = theta)),
    )
    # The associated colors:
    # bb_spectrum can be given any length array for samples, which is returned.
    brdfs = (
        set_brdf_region(is_chequered_region, jnp.array([6, 12]),
                           #on_brdf = lambda uv, mu: jnp.array([1.0, 0.3, 0.0]),
                           #off_brdf = lambda uv, mu: jnp.array([0.0, 0.3, 1.0])
                        ),
        set_brdf_dbb(),
        #set_brdf_chequered(),
    )

    rayphases = construct_screen_rays(xres = xres, yres = yres,
                                        size = size, focal_distance = focal_distance)
    # Color each pixel
    rayphases, colors = batch_render_by_surface(rayphases, shapes, brdfs, 1e-4,
                                                focal_distance * 2.0)

    # Construct the image for viewing with length 3
    colors = batch_normalize(colors)
    frame = colors.reshape(xres, yres, 3)
    save_frame_as_png(frame, filepath="out/image.png")
    return frame

def create_rotating_bh_gif(num_frames = 36, outfile="out/rotating_bh.gif"):
    thetas = [float(theta) for theta in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    frames = [create_bh_frame(xres=200, yres=200, theta = theta) for theta in thetas]
    save_frame_as_gif(frames, outfile)

def create_wobbling_bh_gif(num_frames = 36, outfile="out/wobbling_bh.gif"):
    phis = [float(phi) for phi in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    thetas = [float(jnp.pi/2 - jnp.pi/8 * jnp.cos(phi)) for phi in phis]
    angles = zip(phis, thetas)
    frames = [create_bh_frame(xres=200, yres=200, phi = phi, theta = theta) for phi, theta in angles]
    save_frame_as_gif(frames, outfile)

def create_ns_frame(xres = 400, yres = 400, size = 10.0, 
                    focal_distance = 10.0,
                    phi = -jnp.pi/8):
    from sinar.entities.shapes import put_sphere, rotation
    from sinar.io.loaders import load_fixed_spec_brdf

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 6.0, orient = rotation(theta = jnp.pi / 3.2, phi = phi)),
    )
    # The associated colors:
    energy_points = jnp.array([0.3, 0.9, 1.2])
    ulims = jnp.array([0.1, 0.3])
    vlims = jnp.array([0.1, 0.2]) # belt configuration
    brdfs = (
        set_brdf_region(is_patch_region, ulims, vlims,
                       on_brdf = load_fixed_spec_brdf("tests/inten_incl_patch0.dat", energy_points),
                       off_brdf = set_brdf_region(is_cap_region)
        ),
    )

    rayphases = construct_screen_rays(xres, yres, size, focal_distance)
    # Color each pixel using the batch_render
    rayphases, colors = batch_render_by_surface(rayphases, shapes, brdfs, 1e-4, focal_distance * 2.0)
    frame = colors.reshape(xres, yres, 3)

    # Construct the image for viewing
    save_frame_as_png(frame, filepath="out/image.png")
    return frame

def create_ns_polspec(xres = 200, yres = 200, size = 10.0, focal_distance = 60.0, phi = -jnp.pi/4):
    from sinar.entities.shapes import put_sphere, rotation
    from sinar.io.loaders import load_full_stokes_brdf, read_checked_intensity_file
    from sinar.entities.colors import bb_spectrum
    import numpy as np
    import matplotlib.pyplot as plt

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    orient = rotation(theta = jnp.pi/3, phi = phi)
    ns_radius = 4.0
    shapes = (
        put_sphere(radius = ns_radius, orient = orient),
    )
    # The associated colors:
    energy_points = read_checked_intensity_file("tests/inten_B13_7T7.dat")[0]
    ulims = jnp.array([0.2, 0.4])
    vlims = jnp.array([0.0, 1.0]) # belt configuration
    # Polarized emission requires an array of output values for each energy point
    brdfs = (
        set_brdf_region(is_cap_region, 0.25,
                       on_brdf = load_full_stokes_brdf("tests/inten_B13_7T7.dat"),
                       off_brdf = lambda uv, mu: jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0])[:, jnp.newaxis],
                       (3,len(energy_points)))
        ),
    )

    rayphases = construct_screen_rays(xres, yres, size, focal_distance)
    # Color each pixel using the batch_render
    _, frame = batch_render_by_surface(rayphases, shapes, brdfs, 1e-4, focal_distance * 2.0)
    save_frame_as_png(frame.reshape(xres, yres, 3, len(energy_points))[:, :, :, 5], filepath="out/image.png")
    from sinar.entities.shapes import put_nested_spheres
    from sinar.entities.harmonics import adiabatic_factor, mag_vector
    magnetic_field_config = jnp.array([1.0 * ns_radius**3, 50.0 * ns_radius**4, 0.0 * ns_radius**5])
    bfield = mag_vector(magnetic_field_config, jnp.eye(3), jnp.array([0.0, 0.0, ns_radius]))
    print(jnp.linalg.norm(bfield))
    # The field strength needs to be calculated properly here to get the proper adiabatic radius factor
    radii = adiabatic_factor(energy_points, magnetic_field_config) * ns_radius
    print(radii)
    staged_shapes = put_nested_spheres(radii)
    from sinar.entities.harmonics import stokes_rotation
    rot = lambda pos, dir: stokes_rotation(magnetic_field_config, orient, pos, dir)
    rayphases, pol = staged_batch_render_by_rayphase(rayphases, staged_shapes, rot,
                                                     1e-4,
                                                     focal_distance * 2.0)

    frame_Q = frame[:, 1, :]
    # F_i1j * P_ji -> Tj (means the j is broadcast multiplied)
    # pol needs to be reversed in energy axis because the radii are nested in reverse order
    total_P = jnp.einsum("ij,ji->j", frame_Q, jnp.flip(pol, axis=0)) / frame_Q.shape[0]

    total_I = jnp.mean(frame[:, 0, :], axis=0)
    # TODO: This should be implemented as saving the array:
    plt.figure(figsize=(10, 6))
    plt.plot(energy_points, total_P / total_I)
    plt.xscale("log")
    plt.ylim(-1.0, 1.0)
    plt.xlabel("Energy (keV)")
    plt.ylabel("Polarization / Intensity")
    plt.title("Polarization Components by Nested Stage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def create_ns_polarization_test(xres=400, yres=400, size=10.0, focal_distance=40.0, adiabatic_factor = 10.0, rot_phase=0.0):
    """Creates a neutron star image showing polarization rotation due to magnetic field.
    
    Initial surface emission is 100% U polarization, then magnetic field rotation
    is applied using a pure dipole field. The final image colors represent polarization
    angle (hue) and magnitude (brightness) using HSV colormap.
    """
    from sinar.entities.shapes import put_sphere, put_nested_spheres, rotation, rotate_about_axis
    from sinar.entities.harmonics import stokes_rotation
    from sinar.renderers import construct_screen_rays, batch_render_by_surface, staged_batch_render_by_rayphase
    from sinar.io.visuals import save_frame_as_png
    from matplotlib import cm
    import jax.numpy as jnp
    
    # Neutron star configuration
    ns_radius = 6.0

    orient = rotation(theta=-jnp.pi/4) @ rotate_about_axis(spin_angle=rot_phase, axis=jnp.array([0., 1., 0.]))
    
    # Create the neutron star sphere pointing in the correct direction for the brdf to work
    shapes = (
        put_sphere(radius=ns_radius, orient=orient),
    )
    
    # BRDF that produces 100% U polarization
    # Returns [I, Q, U, V] where I=1 (isotropic intensity), Q=0, U=1, V=0
    # No dependence on mu (viewing angle) since we want isotropic emission
    def pure_U_brdf(uv, mu):
        return jnp.array([1.0, 0.0, 1.0, 0.0])

    def is_dual_cap_region(uv, semi_ap: float = 0.2):
        """Returns True for both polar caps."""
        return (uv[0] < semi_ap) | (uv[0] > (1.0 - semi_ap))
    
    # brdfs = (pure_U_brdf,)
    brdfs = (
        set_brdf_region(is_dual_cap_region, 1.0,
                       on_brdf=pure_U_brdf,
                       off_brdf=lambda uv, mu: jnp.array([0.0, 0.0, 0.0, 0.0])
        ),
    )
    
    # Render initial surface with 100% U polarization
    rayphases = construct_screen_rays(xres, yres, size, focal_distance)
    
    magnetic_field_config = jnp.array([1.0, 0.0, 0.0])
    
    adiabatic_radius = adiabatic_factor * ns_radius
    radii = jnp.array([adiabatic_radius])
    
    # Create nested spheres for the magnetic field rotation region
    staged_shapes = put_nested_spheres(radii)
    
    # Set up magnetic field orientation for [1, 0, 0] dipole direction
    # The magnetic field moment is rotated from [0, 0, 1] to [1, 0, 0]
    
    # Create rotation function for Stokes parameters
    rot = lambda pos, dir: stokes_rotation(magnetic_field_config, orient, pos, dir)
    
    # Apply magnetic field rotation using staged rendering
    rayphases, pol_rotation = staged_batch_render_by_rayphase(rayphases, staged_shapes, 
                                                              rot, 1e-4, focal_distance * 2.0)

    rayphases, initial_colors = batch_render_by_surface(rayphases, shapes, brdfs, 1e-4, adiabatic_radius * 1.5)
    
    # Save initial surface coloring for verification
    # Create a simple visualization of initial U polarization (should be naturally black background)
    # initial_vis = initial_colors[:, -3:]
    # initial_frame = initial_vis.reshape(xres, yres, 3)
    # save_frame_as_png(initial_frame, filepath="out/ns_initial_polarization.png")
    
    # Extract Q and U from initial colors
    Q_initial = initial_colors[:, 1]  # Should be all zeros
    U_initial = initial_colors[:, 2]  # Should be all ones
    
    # Apply rotation to polarization using complex representation: P = Q + iU
    P_complex = Q_initial + 1j * U_initial
    # pol_rotation has shape (n_stages, n_pixels), we have 1 stage
    P_rotated = P_complex * pol_rotation[0]
    
    # Extract final Q and U components
    Q_final = jnp.real(P_rotated)
    U_final = jnp.imag(P_rotated)
    
    # Find max polarization for normalization (only among non-zero pixels)
    surface_pixels = initial_colors[:, 0] > 0
    if jnp.sum(surface_pixels) > 0:
        max_pol = jnp.max(jnp.sqrt(Q_final[surface_pixels]**2 + U_final[surface_pixels]**2))
    else:
        max_pol = 1.0  # fallback
    
    # Calculate polarization angle from Q and U
    # Factor of 0.5 because polarization angle is arctan(U/Q)/2
    phase = 0.5 * jnp.arctan2(U_final, Q_final)
    
    # Normalize phase to [0, 1] range for colormap
    # arctan2 returns [-π, π], so 0.5*arctan2 returns [-π/2, π/2]
    # We shift and scale to [0, 1] which maps to [0°, 180°] polarization
    phase_norm = (phase + 0.5*jnp.pi) / jnp.pi
    
    # Calculate normalized magnitude
    pol_magnitude = jnp.sqrt(Q_final**2 + U_final**2) / max_pol
    
    # Use HSV colormap: hue = polarization angle
    cmap = cm.hsv
    rgba_colors = cmap(phase_norm)
    
    # Modulate brightness by polarization magnitude
    # Multiply RGB channels by magnitude (alpha channel ignored)
    rgb_colors = rgba_colors[..., :3] * pol_magnitude[..., jnp.newaxis]
    
    # Reshape to image
    frame = rgb_colors.reshape(xres, yres, 3)
    
    # Save the image
    save_frame_as_png(frame, filepath="out/ns_polarization_test.png")
    
    return frame


def create_rotating_ns_gif(num_frames = 36, outfile="out/rotating_ns.gif"):
    phis = [float(phi) for phi in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    frames = [create_ns_frame(xres=200, yres=200, phi = phi) for phi in phis]
    save_frame_as_gif(frames, outfile)

def create_rotating_ns_polarization_gif(num_frames=36, outfile="out/rotating_ns_polarization.gif", 
                                        adiabatic_factor=10.0):
    rot_phases = [float(phase) for phase in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    frames = [
        create_ns_polarization_test(xres=200, yres=200, size=10.0, 
                                    focal_distance=12.0*adiabatic_factor, 
                                    adiabatic_factor=adiabatic_factor,
                                    rot_phase=phase)
        for phase in rot_phases
    ]
    save_frame_as_gif(frames, outfile)

def create_rotating_ns_polarization_gif_optimized(num_frames=36, 
                                                  outfile="out/rotating_ns_polarization.gif", 
                                                  adiabatic_factor=10.0,
                                                  capsize = 0.4 / jnp.pi,
                                                  save_pol_data=True):
    """Optimized version that caches ray phases and only re-evaluates orientation-dependent quantities."""
    from sinar.entities.shapes import put_sphere, put_nested_spheres, rotation, rotate_about_axis
    from sinar.entities.harmonics import stokes_rotation
    from sinar.rays import normalize
    import jax
    import numpy as np
    
    # Configuration
    ns_radius = 5.8
    xres, yres = 200, 200
    size = 8.0
    focal_distance = 20.0 * adiabatic_factor
    magnetic_field_config = jnp.array([1.0, 100.0, 0.0])
    
    # === Cache ray phases (expensive GR raytracing done once) ===
    rayphases_init = construct_screen_rays(xres, yres, size, focal_distance, n_aux=1)
    
    # March through adiabatic zone
    adiabatic_radius = adiabatic_factor * ns_radius
    staged_shapes = put_nested_spheres(jnp.array([adiabatic_radius]))
    
    dummy_rot = lambda pos, dir: 1.0 + 0j
    rayphases_adiabatic, _ = staged_batch_render_by_rayphase(
        rayphases_init, staged_shapes, dummy_rot, 1e-5, focal_distance * 2.0
    )
    
    # Continue to surface and use to check if hit
    orient_identity = jnp.eye(3)
    shapes_identity = (put_sphere(radius=ns_radius, orient=orient_identity),)
    # This is only used to zero out and as an identity multiplication for success
    is_surface_brdf = (lambda uv, mu: jnp.array([1., 1., 1., 1.]),)
    rayphases_surface, is_surface = batch_render_by_surface(
        rayphases_adiabatic, shapes=shapes_identity, brdfs=is_surface_brdf, dtol=1e-5, timespan=focal_distance * 2.0,
    )
    
    # Extract cached arrays
    pos_adiabatic = rayphases_adiabatic[:, :3]
    dir_adiabatic = rayphases_adiabatic[:, 3:6]
    pos_surface = rayphases_surface[:, :3]
    dir_surface = rayphases_surface[:, 3:6]
    
    # === Define pure functions for frame generation ===
    def pure_U_brdf(uv, mu):
        return jnp.array([1.0, 0.0, 1.0, 0.0])
    
    def is_dual_cap_region(uv, semi_ap=0.15):
        return (uv[0] < semi_ap) #| (uv[0] > (1.0 - semi_ap))
    
    def render_frame_at_phase(rot_phase):
        orient = rotation(theta=0.0) @ rotate_about_axis(
            spin_angle=rot_phase, axis=jnp.array([1., 0., 0.])
        )
        
        # Compute Stokes rotation
        pol_rotation = jax.vmap(
            lambda pos, dir: stokes_rotation(magnetic_field_config, orient, pos, dir)
        )(pos_adiabatic, dir_adiabatic)
        
        sphere = put_sphere(radius=ns_radius, orient=orient)
        
        def eval_brdf(pos, dir):
            uv = sphere.uv(pos)
            sn = sphere.sn(pos)
            mu = jnp.vecdot(-normalize(dir), normalize(sn))
            in_cap = is_dual_cap_region(uv, capsize)
            return jax.lax.select(in_cap, pure_U_brdf(uv, mu), jnp.zeros(4))
        
        initial_colors = jax.vmap(eval_brdf)(pos_surface, dir_surface)
        
        # Zero out colors for rays that missed the surface through the is_surface
        initial_colors = initial_colors * is_surface
        
        # Apply rotation
        P_complex = initial_colors[:, 1] + 1j * initial_colors[:, 2]
        P_rotated = P_complex * pol_rotation
        Q_final = jnp.real(P_rotated)
        U_final = jnp.imag(P_rotated)
        I_final = initial_colors[:, 0]
        
        # Calculate total polarization (sum over all pixels that hit the surface)
        total_I = jnp.sum(I_final)
        total_Q = jnp.sum(Q_final)
        total_U = jnp.sum(U_final)
        
        # Polarization degree and angle
        pol_degree = jnp.sqrt(total_Q**2 + total_U**2) / (total_I + 1e-10)
        pol_angle = 0.5 * jnp.arctan2(total_U, total_Q)  # in radians
        
        # Visualize
        phase = 0.5 * jnp.arctan2(U_final, Q_final)
        phase_norm = (phase + 0.5*jnp.pi) / jnp.pi
        pol_mag = jnp.sqrt(Q_final**2 + U_final**2)
        
        from matplotlib import cm
        rgba = cm.hsv(phase_norm)
        rgb = rgba[..., :3] * pol_mag[..., jnp.newaxis]
        
        frame = rgb.reshape(xres, yres, 3)
        
        # Return frame and polarization data
        pol_info = {
            'phase': float(rot_phase),
            'pol_degree': float(pol_degree),
            'pol_angle': float(pol_angle),
            'total_I': float(total_I),
            'total_Q': float(total_Q),
            'total_U': float(total_U)
        }
        
        return frame, pol_info
    
    # Generate frames and collect polarization data
    rot_phases = [float(phase) for phase in jnp.linspace(0.0, jnp.pi/2, num_frames)]
    results = [render_frame_at_phase(phase) for phase in rot_phases]
    
    frames = [result[0] for result in results]
    pol_data_list = [result[1] for result in results]
    
    save_frame_as_gif(frames, outfile)
    
    # Save polarization data
    if save_pol_data:
        pol_data_array = np.array([
            [d['phase'], d['pol_degree'], d['pol_angle'], 
             d['total_I'], d['total_Q'], d['total_U']]
            for d in pol_data_list
        ])
        np.savetxt('out/polarization_vs_phase.dat', pol_data_array,
                   header='phase(rad) pol_degree pol_angle(rad) total_I total_Q total_U',
                   fmt='%.6e')
        print(f"Saved polarization data to out/polarization_vs_phase.dat")
    
    return frames, pol_data_list

def create_varying_adiabatic_radius_gif(num_frames = 8, outfile="out/varying_adiabatic_radius.gif"):
    adiabatic_factors = [float(factor) for factor in jnp.geomspace(1.0, 30.0, num_frames)]
    frames = [
        create_ns_polarization_test(xres=200, yres=200, size=10.0, 
                                    focal_distance=12.0*factor, adiabatic_factor=factor)
        for factor in adiabatic_factors
    ]
    save_frame_as_gif(frames, outfile, duration=450)

def test_render():
    # create_varying_adiabatic_radius_gif(num_frames=20)

    create_rotating_ns_polarization_gif_optimized(num_frames=64, adiabatic_factor=2.1, capsize=jnp.pi/jnp.pi,
                                                  outfile="out/rotating_ns_polarization.gif")

    # create_rotating_bh_gif()