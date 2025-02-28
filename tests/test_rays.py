from ..rays import batch_render, render
import jax.numpy as jnp
import numpy as np

def test_raymarch():
    pass

def test_render(xres = 400, yres = 400):
    from ..shapes import put_sphere, put_cylinder, put_thindisc, rotation, y_up_mat
    from PIL import Image
    # Coordinate grid: the screen
    xs = jnp.linspace(-1., 1., xres)*5.0
    ys = jnp.linspace(1., -1., yres)*5.0 # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    pixlocs = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # The scene requires geoms:
    shapes = (
        put_sphere(),
        #put_thindisc(inner=3.0, outer=8.0, orient = rotation(jnp.pi/2.1)),
    )

    # Color each pixel
    colors = batch_render(shapes, pixlocs)

    # Construct the image for viewing
    image = colors.reshape(xres, yres, 4)
    image = np.abs(np.array(image))
    image = (image * 250).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('image.png')