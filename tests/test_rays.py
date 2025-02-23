from ..rays import batch_render
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

def test_raymarch():
    pass

def test_render():
    from ..geoms import put_sphere, sdmin_scene
    from PIL import Image
    # Coordinate grid: the screen
    xs = jnp.linspace(-1., 1., 800)*3.0
    ys = jnp.linspace(1., -1., 800)*3.0 # coordinate flip!
    X, Y = jnp.meshgrid(xs, ys)

    pixlocs = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # The scene requires geoms:
    spheres = [put_sphere(location = jnp.array([1.0,0.0,0.0])), 
               put_sphere(location = jnp.array([0.0,0.0,0.0]))]
    scene_sd = partial(sdmin_scene, spheres)

    colors = batch_render(scene_sd, pixlocs)

    image = colors.reshape(800, 800, 4)
    image = np.abs(np.array(image))
    image = (image * 250).astype(jnp.uint8)

    im = Image.fromarray(image)
    im.save('raymarched_test.png')