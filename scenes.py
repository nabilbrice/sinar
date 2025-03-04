from typing import NamedTuple, Callable
from jax import Array
import jax
import jax.numpy as jnp

# For an entity collection, the distance field is the combination of all with
# a minimum of some kind.
# This approach loses information.
# Holding a Iterable Tuple of Entity as a scene actually allows
# for generating the minimum distance array and then from that array call
# the appropriate function for the color.

class Entity(NamedTuple):
    distance_field: Callable
    color_field: Callable

# The signed distance minimum still must compute the sd function
# for each of the entities in the list.
# By taking the minimum of the distance as the key, the color can be obtained.
def sdmin_scene(entities: list, position: Array):
    return jnp.min(jnp.array(
        [entity(position) for entity in entities]
        ))

def sdargmin_scene(entities: list, position: Array):
    return jnp.argmin(jnp.array(
        [entity(position) for entity in entities]
    ))

# For a smoother blending of objects, but it is slower
def sdsmin_scene(entities: list, position: Array):
    sdistances = jnp.array(
        [entity(position) for entity in entities]
        )
    return -jax.nn.logsumexp(-sdistances*16.0)/16
