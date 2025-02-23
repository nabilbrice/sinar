from ..rays import batch_render
import jax.numpy as jnp
import numpy as np

def test_raymarch():
    pass

def test_render(xres = 400, yres = 400):
    from ..geoms import put_sphere
    from PIL import Image
    # Coordinate grid: the screen
    xs = jnp.linspace(-1., 1., xres)*3.0
    ys = jnp.linspace(1., -1., yres)*3.0 # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    pixlocs = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # The scene requires geoms:
    spheres = [
 #       put_sphere(location = jnp.array([1.0,0.0,0.0])), 
        put_sphere(location = jnp.array([0.0,0.0,0.0]))
    ]

    # Color each pixel
    colors = batch_render(spheres, pixlocs)

    # Construct the image for viewing
    image = colors.reshape(xres, yres, 4)
    image = np.abs(np.array(image))
    image = (image * 250).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('raymarched_test.png')