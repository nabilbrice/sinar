from ..renderers import construct_pixlocs, batch_render
import jax.numpy as jnp
import numpy as np

def test_raymarch():
    pass

def test_render(xres = 400, yres = 400, size = 10.):
    from ..shapes import put_sphere, put_cylinder, put_thindisc, rotation, y_up_mat
    from PIL import Image

    # The scene requires shapes:
    shapes = (
        put_sphere(radius = 2.0),
        #put_thindisc(inner=3.0, outer=8.0, orient = rotation(jnp.pi/3.1)),
    )

    pixlocs = construct_pixlocs()
    # Color each pixel
    colors = batch_render(shapes, pixlocs)

    # Construct the image for viewing
    image = colors.reshape(xres, yres, 4)
    image = np.abs(np.array(image))
    image = (image * 250).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('image.png')