from ..renderers import construct_pixlocs, batch_render
from ..colors import set_brdf_patch, set_brdf_chequered 
import jax.numpy as jnp
import numpy as np

def test_raymarch():
    pass

def test_render(xres = 400, yres = 400, size = 10.0):
    from ..shapes import put_sphere, put_cylinder, put_thindisc, rotation, y_up_mat
    from PIL import Image

    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 2.0),
    #    put_sphere(location = jnp.array([6.0, 3.0, 0.0]), radius=2.0),
        put_thindisc(inner=3.0, outer=10.0, orient = rotation(jnp.pi/2.3)),
    )
    # The associated colors:
    brdfs = (
        set_brdf_chequered(boxes = jnp.array([6, 12]),
                           bright=lambda mu: jnp.array([1.0, 0.3, 0.0]),
                           dark=lambda mu: jnp.array([0.0, 0.3, 1.0])),
        set_brdf_chequered(boxes = jnp.array([10, 20]),
                           bright=lambda mu: jnp.array([0.5, 0.5, 0.5]),
                           dark=lambda mu: jnp.array([0.2, 0.0, 0.8])),
    )

    pixlocs = construct_pixlocs()
    # Color each pixel
    colors = batch_render(shapes, brdfs, pixlocs)

    # Construct the image for viewing
    image = colors.reshape(xres, yres, 4)
    image = np.abs(np.array(image))
    image = (image * 250).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('image.png')