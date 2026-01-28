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
        put_sphere(radius = 5.0),
    )
    shapes = (
        # black hole shadow is 2.0 but event horizon is 1.0
        put_sphere(radius = 2.0, orient = rotation(phi = phi, theta = theta)),
        put_thindisc(inner=3.0, outer=8.0, height=0.1, orient = rotation(phi = phi, theta = theta)),
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
    rayphases, colors = batch_render_by_surface(rayphases, shapes, brdfs, 1e-4, 24.0)

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
        put_sphere(radius = 2.5, orient = rotation(theta = jnp.pi / 3.2, phi = phi)),
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
    rayphases, colors = batch_render_by_surface(rayphases, shapes, brdfs, 1e-4, 24.0)
    frame = colors.reshape(xres, yres, 3)

    # Construct the image for viewing
    save_frame_as_png(frame, filepath="out/image.png")
    return frame

def create_ns_polspec(xres = 200, yres = 200, size = 5.0, focal_distance = 50.0, phi = -jnp.pi/4):
    from sinar.entities.shapes import put_sphere, rotation
    from sinar.io.loaders import load_full_stokes_brdf, read_checked_intensity_file
    from sinar.entities.colors import bb_spectrum
    import numpy as np
    import matplotlib.pyplot as plt

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    orient = rotation(theta = jnp.pi/2.1, phi = phi)
    shapes = (
        put_sphere(radius = 1.2, orient = orient),
    )
    # The associated colors:
    energy_points = read_checked_intensity_file("tests/inten_B13_7T7.dat")[0]
    ulims = jnp.array([0.2, 0.4])
    vlims = jnp.array([0.0, 1.0]) # belt configuration
    # Polarized emission requires an array of output values for each energy point
    brdfs = (
        set_brdf_region(is_cap_region, 0.2,
                       on_brdf = load_full_stokes_brdf("tests/inten_B13_7T7.dat"),
                       off_brdf = lambda uv, mu: jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0])[:, jnp.newaxis],
                       (3,len(energy_points)))
        ),
    )

    rayphases = construct_screen_rays(xres, yres, size, focal_distance)
    # Color each pixel using the batch_render
    _, frame = batch_render_by_surface(rayphases, shapes, brdfs, 1e-4, 100.0)
    save_frame_as_png(frame.reshape(xres, yres, 3, len(energy_points))[:, :, :, 5], filepath="out/image.png")
    from sinar.entities.shapes import put_nested_spheres
    from sinar.entities.harmonics import adiabatic_factor, mag_vector
    magnetic_field_config = jnp.array([5.0 * 1.2**3, 50.0 * 1.2**4, 0.0 * 1.2**5])
    bfield = mag_vector(magnetic_field_config, jnp.eye(3), jnp.array([0.0, 0.0, 1.2]))
    print(jnp.linalg.norm(bfield))
    # The field strength needs to be calculated properly here to get the proper adiabatic radius factor
    radii = adiabatic_factor(energy_points, magnetic_field_config) * 1.2
    print(radii)
    staged_shapes = put_nested_spheres(radii)
    from sinar.entities.harmonics import stokes_rotation
    rot = lambda pos, dir: stokes_rotation(magnetic_field_config, orient, pos, dir)
    rayphases, pol = staged_batch_render_by_rayphase(rayphases, staged_shapes, rot, 1e-4,
                                                     100.0)

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

def create_rotating_ns_gif(num_frames = 36, outfile="out/rotating_ns.gif"):
    phis = [float(phi) for phi in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    frames = [create_ns_frame(xres=200, yres=200, phi = phi) for phi in phis]
    save_frame_as_gif(frames, outfile)

def test_render():
    create_ns_polspec()