from ..rays import raymarch, render
import jax
import jax.numpy as jnp
import numpy as np

def test_raymarch():
    pass

def test_render():
    from PIL import Image
    # Coordinate grid
    xs = jnp.linspace(-1., 1., 200)
    ys = jnp.linspace(1., -1., 200) # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    pixlocs = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    batch_render = jax.vmap(render)

    colors = batch_render(pixlocs)

    image = colors.reshape(200, 200, 4)
    image = np.array(image)
    image = (image * 255).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('raymarched_test.png')