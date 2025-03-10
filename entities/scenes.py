from typing import NamedTuple, Callable
from jax import Array
import jax
import jax.numpy as jnp
from functools import partial

# For an entity collection, the distance field is the combination of all with
# a minimum of some kind.
# This approach loses information.
# Holding a Iterable Tuple of Entity as a scene actually allows
# for generating the minimum distance array and then from that array call
# the appropriate function for the color.

# The signed distance minimum still must compute the sd function
# for each of the entities in the list.
# By taking the minimum of the distance as the key, the color can be obtained.
def sdmin_scene(shapes: list, position: Array):
    return jnp.min(jnp.array(
        [shape.sdf(position) for shape in shapes]
        ))

def sdargmin_scene(shapes: list, position: Array):
    return jnp.argmin(jnp.array(
        [shape.sdf(position) for shape in shapes]
    ), axis=-1)

# For a smoother blending of objects, but it is slower
def sdsmin_scene(shapes: list, position: Array):
    sdistances = jnp.array(
        [entity.sdf(position) for entity in shapes]
        )
    return -jax.nn.logsumexp(-sdistances*16.0)/16
