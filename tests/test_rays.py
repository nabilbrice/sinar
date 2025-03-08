from ..renderers import construct_pixlocs, batch_render
from ..colors import set_brdf_patch, set_brdf_chequered , set_brdf_dbb
import jax.numpy as jnp
import numpy as np

# Commonly used configuration for bh marching
def bh_raymarch(xres = 400, yres = 400, size = 10.0):
    from ..shapes import put_sphere, put_cylinder, put_thindisc, rotation, y_up_mat
    from PIL import Image

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 2.0),
        put_thindisc(inner=3.0, outer=10.0, height=0.1, orient = rotation(jnp.pi/2.3)),
    )
    # The associated colors:
    brdfs = (
        set_brdf_chequered(boxes = jnp.array([6, 12]),
                           bright=lambda mu: jnp.array([1.0, 0.3, 0.0]),
                           dark=lambda mu: jnp.array([0.0, 0.3, 1.0])),
        set_brdf_dbb(1.0),
        #set_brdf_chequered(),
    )

    pixlocs = construct_pixlocs(xres, yres)
    # Color each pixel
    colors = batch_render(shapes, brdfs, pixlocs)

    # Construct the image for viewing
    image = colors.reshape(xres, yres, 4)
    image = np.abs(np.array(image))
    image = (image * 250).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('image.png')

def ns_raymarch(xres = 400, yres = 400, size = 10.0):
    from ..shapes import put_sphere
    from ..loaders import load_checked_interpolators, load_checked_fixed_spectrum
    from PIL import Image
    from functools import partial

    # TODO: Both shapes and brdfs can be encapsulated into a single list of entities
    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 2.5),
    )
    # The associated colors:
    energy_points = jnp.array([0.3, 0.9, 1.2])
    brdfs = (
        load_checked_fixed_spectrum("tests/inten_incl_patch0.dat", energy_points),
        set_brdf_chequered(),
    )

    pixlocs = construct_pixlocs(xres, yres)
    # Color each pixel
    colors = batch_render(shapes, brdfs, pixlocs)

    # Construct the image for viewing
    image = colors.reshape(xres, yres, 4)
    image = np.abs(np.array(image))
    image = (image * 250).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('image.png')

def test_render():
    bh_raymarch()