import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from sinar.entities.shapes import put_sphere, put_thindisc, rotation
from sinar.entities.colors import set_brdf_region, set_brdf_dbb, is_chequered_region
from sinar.renderers import construct_screen_rays, batch_render_by_surface
from sinar.io.visuals import save_frame_as_png, save_frame_as_gif
from sinar.rays import batch_normalize


def create_bh_frame(xres = 400, yres = 400, size = 10.0,
                    focal_distance = 20.0,
                    theta = jnp.pi/2.1, phi = 0.0):
    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    bounds = (
        put_sphere(radius = 12.0),
    )
    shapes = (
        # black hole event horizon is 2.0
        put_sphere(radius = 2.0, orient = rotation(phi = phi, theta = theta)),
        put_thindisc(inner=6.0, outer=12.0, height=0.1, orient = rotation(phi = phi, theta = theta)),
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
    rayphases, colors = batch_render_by_surface(rayphases, shapes, brdfs, dtol=1e-4,
                                                timespan=focal_distance * 2.0)

    # Construct the image for viewing with length 3
    colors = batch_normalize(colors)
    frame = colors.reshape(xres, yres, 3)
    save_frame_as_png(frame, filepath="out/image.png")
    return frame


def create_rotating_bh_gif(num_frames = 36, outfile="out/rotating_bh.gif"):
    thetas = [float(theta) for theta in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    frames = [create_bh_frame(xres=300, yres=300, theta = theta) for theta in thetas]
    save_frame_as_gif(frames, outfile)


def create_wobbling_bh_gif(num_frames = 36, outfile="out/wobbling_bh.gif"):
    phis = [float(phi) for phi in jnp.linspace(0.0, 2.0*jnp.pi, num_frames)]
    thetas = [float(jnp.pi/2 - jnp.pi/8 * jnp.cos(phi)) for phi in phis]
    angles = zip(phis, thetas)
    frames = [create_bh_frame(xres=300, yres=300, phi = phi, theta = theta) for phi, theta in angles]
    save_frame_as_gif(frames, outfile)


if __name__ == "__main__":
    # create_rotating_bh_gif()

    create_wobbling_bh_gif()