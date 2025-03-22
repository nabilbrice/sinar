from sinar.renderers import construct_pixlocs, construct_screen_rays, batch_render_by_surface, batch_render_by_rayphase
from sinar.entities.colors import set_brdf_region , set_brdf_dbb, is_cap_region, is_patch_region, is_chequered_region
from sinar.io.visuals import save_frame_as_png, save_frame_as_gif
import jax.numpy as jnp

# Commonly used configuration for bh marching
def create_bh_frame(xres = 400, yres = 400, size = 10.0,
                    focal_distance = 10.0,
                    theta = jnp.pi/2.3, phi = 0.0):
    from sinar.entities.shapes import put_sphere, put_thindisc, rotation

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 2.0),
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

    screen_rays = construct_screen_rays(xres = xres, yres = yres,
                                        size = size, focal_distance = focal_distance)
    # Color each pixel
    # TODO: shapes is manually written in as a tuple
    colors = batch_render_by_surface(screen_rays, shapes, brdfs)

    # Construct the image for viewing with length 3
    from sinar.rays import batch_normalize
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

    pixlocs = construct_pixlocs(xres, yres)
    # Color each pixel using the batch_render
    frame = batch_render_by_surface(pixlocs, focal_distance, shapes, brdfs).reshape(xres, yres, 3)

    # Construct the image for viewing
    save_frame_as_png(frame, filepath="out/image.png")
    return frame

def create_ns_polspec(xres = 200, yres = 200, size = 10.0, focal_distance = 10.0, phi = -jnp.pi/4):
    from sinar.entities.shapes import put_sphere, rotation
    from sinar.io.loaders import load_full_stokes_brdf, read_checked_intensity_file
    from sinar.entities.colors import bb_spectrum
    import numpy as np
    import matplotlib.pyplot as plt

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    orient = rotation(theta = jnp.pi/3, phi = phi)
    shapes = (
        put_sphere(radius = 2.5, orient = orient),
    )
    # The associated colors:
    energy_points = read_checked_intensity_file("tests/inten_incl_patch0.dat")[0]
    ulims = jnp.array([0.1, 0.2])
    vlims = jnp.array([0.1, 0.2]) # belt configuration
    # Polarized emission requires an array of output values for each energy point
    brdfs = (
        set_brdf_region(is_cap_region, 
                       on_brdf = load_full_stokes_brdf("tests/inten_incl_patch0.dat"),
                       off_brdf = lambda uv, mu: jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0])[:, jnp.newaxis],
                       (3,len(energy_points)))
        ),
    )

    pixlocs = construct_pixlocs(xres, yres, size)
    # Color each pixel using the batch_render
    frame = batch_render_by_surface(pixlocs, focal_distance, shapes, brdfs)
    save_frame_as_png(frame.reshape(xres, yres, 3, len(energy_points))[:, :, :, 5], filepath="out/image.png")

    # TODO: The radius here is the adiabatic radius, which actually depends on the energy...
    pol_shapes = (
        put_sphere(radius = 5.0, orient = orient),
    )
    from sinar.entities.harmonics import stokes_rotation
    rot = lambda pos, dir: stokes_rotation(jnp.array([1.0, 0.0, 0.0]), orient, pos, dir)
    pol = batch_render_by_rayphase(pixlocs, focal_distance, pol_shapes, rot)

    frame_Q = frame[:, 1, :]
    frame_P = frame_Q * pol[:, jnp.newaxis]

    total_P = jnp.mean(frame_P, axis=0)
    total_I = jnp.mean(frame[:, 0, :], axis=0)
    # TODO: This should be implemented as saving the array:
    plt.plot(energy_points, jnp.real(total_P) / total_I)
    plt.plot(energy_points, jnp.imag(total_P) / total_I)
    plt.xscale("log")
    plt.show()

def create_rotating_ns_gif(num_frames = 36, outfile="out/rotating_ns.gif"):
    phis = [float(phi) for phi in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    frames = [create_ns_frame(xres=200, yres=200, phi = phi) for phi in phis]
    save_frame_as_gif(frames, outfile)

def test_render():
    create_bh_frame()