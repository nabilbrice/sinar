from ..rays import raymarch, render
import jax
import jax.numpy as jnp
import numpy as np

def test_raymarch():
    pass

def test_render():
    from ..geoms import new_sphere
    from PIL import Image
    # Coordinate grid: the screen
    xs = jnp.linspace(-1., 1., 800)*3.0
    ys = jnp.linspace(1., -1., 800)*3.0 # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    pixlocs = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # The scene requires geoms:
    scene = new_sphere(location=jnp.array([1.0,0.0,0.0]))

    batch_render = jax.vmap(render, in_axes=(None, 0))

    colors = batch_render(scene, pixlocs)

    image = colors.reshape(800, 800, 4)
    image = np.array(image)
    image = (image * 255).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('raymarched_test.png')