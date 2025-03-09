from ..renderers import construct_pixlocs, batch_render
from ..colors import set_brdf_region , set_brdf_dbb, is_cap_region, is_patch_region, is_chequered_region
from ..io.visuals import save_frame_as_png, save_frame_as_gif
import jax.numpy as jnp

# Commonly used configuration for bh marching
def create_bh_frame(xres = 400, yres = 400, size = 10.0):
    from ..shapes import put_sphere, put_thindisc, rotation

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 2.0),
        put_thindisc(inner=3.0, outer=8.0, height=0.1, orient = rotation(theta = jnp.pi/2.3)),
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

    pixlocs = construct_pixlocs(xres, yres)
    # Color each pixel
    colors = batch_render(shapes, brdfs, pixlocs)

    # Construct the image for viewing with length 3
    frame = colors.reshape(xres, yres, 3)
    save_frame_as_png(frame, filepath="image.png")

def create_ns_frame(xres = 400, yres = 400, size = 10.0, phi = -jnp.pi/8):
    from ..shapes import put_sphere, rotation
    from ..io.loaders import load_checked_fixed_spectrum

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
                       on_brdf = load_checked_fixed_spectrum("tests/inten_incl_patch0.dat", energy_points),
                       off_brdf = set_brdf_region(is_cap_region)
        ),
    )

    pixlocs = construct_pixlocs(xres, yres)
    # Color each pixel using the batch_render
    frame = batch_render(shapes, brdfs, pixlocs).reshape(xres, yres, 3)

    # Construct the image for viewing
    save_frame_as_png(frame, filepath="image.png")
    return frame

def create_ns_spectrum(xres = 400, yres = 400, size = 10.0, phi = -jnp.pi/8):
    from ..shapes import put_sphere, rotation
    from ..io.loaders import load_checked_fixed_spectrum
    from ..colors import bb_spectrum
    import numpy as np
    import matplotlib.pyplot as plt

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 2.5, orient = rotation(theta = jnp.pi / 3.2, phi = phi)),
    )
    # The associated colors:
    energy_points = jnp.linspace(0.2, 8.0, 10)
    ulims = jnp.array([0.1, 0.3])
    vlims = jnp.array([0.1, 0.2]) # belt configuration
    brdfs = (
        set_brdf_region(is_patch_region, ulims, vlims,
                       on_brdf = load_checked_fixed_spectrum("tests/inten_incl_patch0.dat", energy_points),
                       off_brdf = lambda uv, mu: bb_spectrum(0.1, energy_points)
        ),
    )

    pixlocs = construct_pixlocs(xres, yres, size)
    # Color each pixel using the batch_render
    frame = batch_render(shapes, brdfs, pixlocs)
    spectrum = jnp.sum(frame, axis=0)
    plt.plot(energy_points, spectrum)
    plt.show()

def create_rotating_ns_gif(num_frames = 36, outfile="rotating_ns.gif"):
    phis = [float(phi) for phi in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    frames = [create_ns_frame(xres=100, yres=100, phi = phi) for phi in phis]
    save_frame_as_gif(frames, outfile)

def test_render():
    create_ns_spectrum(phi = jnp.pi + jnp.pi/8)